"""
Core evaluation module for CUDA-Hercules.

Compiles reference and LLM-generated CUDA code, runs both on the same inputs,
checks correctness, and measures performance.
"""

import os
import signal
import sys
import importlib.util
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


UNSUPPORTED_ARCH_PREFIX = "UNSUPPORTED_ARCH:"


class _KernelTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _KernelTimeout("CUDA kernel execution timed out")

from .compiler import compile_cuda_module, get_reference_include_dirs
from .timing import time_cuda_function, TimingStats
from .utils import get_project_root


@dataclass
class KernelExecResult:
    """Results from evaluating a single kernel."""
    task_id: int = -1
    task_name: str = ""

    compiled: bool = False
    ref_compiled: bool = False
    correctness: bool = False
    per_output_correctness: dict = field(default_factory=dict)

    solution_time: Optional[TimingStats] = None
    reference_time: Optional[TimingStats] = None
    speedup: float = 0.0

    compilation_error: str = ""
    runtime_error: str = ""
    metadata: dict = field(default_factory=dict)


def load_task_def(task_dir: str) -> dict:
    """
    Load a task definition (def.py) as a module and return its contents.

    Returns dict with: FUNCTION_SIGNATURE, SCALAR_ARGS, TOLERANCES,
    get_inputs(), get_outputs(), reference_fn(), etc.
    """
    def_path = os.path.join(task_dir, "def.py")
    if not os.path.exists(def_path):
        raise FileNotFoundError(f"Task definition not found: {def_path}")

    spec = importlib.util.spec_from_file_location("task_def", def_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        "FUNCTION_SIGNATURE": getattr(module, "FUNCTION_SIGNATURE"),
        "SCALAR_ARGS": getattr(module, "SCALAR_ARGS", {}),
        "TOLERANCES": getattr(module, "TOLERANCES", {"atol": 1e-4, "rtol": 1e-4}),
        "HAS_CUDA_REFERENCE": getattr(module, "HAS_CUDA_REFERENCE", False),
        "REFERENCE_EXTRA_INCLUDES": getattr(module, "REFERENCE_EXTRA_INCLUDES", []),
        "REFERENCE_EXTRA_CUDA_FLAGS": getattr(module, "REFERENCE_EXTRA_CUDA_FLAGS", []),
        "REFERENCE_EXTRA_LDFLAGS": getattr(module, "REFERENCE_EXTRA_LDFLAGS", []),
        "get_inputs": module.get_inputs,
        "get_outputs": module.get_outputs,
        "reference_fn": getattr(module, "reference_fn", None),
        "DESCRIPTION": getattr(module, "DESCRIPTION", ""),
        "module": module,
    }


def _is_class2_general_task(task_dir: str) -> bool:
    """Return True for class2/general def.py tasks."""
    return "/tasks/class2/general/" in Path(task_dir).resolve().as_posix()


def _is_class2_task(task_dir: str) -> bool:
    """Return True for any class2 def.py task."""
    return "/tasks/class2/" in Path(task_dir).resolve().as_posix()


def _get_device_sm(device: Optional[torch.device] = None) -> int:
    """Get the active CUDA device's SM capability as an integer like 80 or 120."""
    if not torch.cuda.is_available():
        return 0

    if isinstance(device, torch.device):
        device_index = device.index if device.index is not None else torch.cuda.current_device()
    elif device is None:
        device_index = torch.cuda.current_device()
    else:
        device_index = int(device)

    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


def _current_arch_gencode_flags(target_sm: int) -> list[str]:
    """Return nvcc flags targeting exactly the current GPU arch plus PTX fallback."""
    return [
        "-gencode", f"arch=compute_{target_sm},code=sm_{target_sm}",
        "-gencode", f"arch=compute_{target_sm},code=compute_{target_sm}",
    ]


def get_solution_include_dirs(root_dir: str) -> list[str]:
    """Include paths permitted for the LLM solution in class2 builds.

    We shipped CUTLASS in reference_sources/ and most class2 anti-cheat
    configs allow models to `#include <cutlass/...>` (they only forbid
    domain-specific libraries like flash_attn / cufft). Without
    adding the CUTLASS root to the solution include path, such includes
    fail at compile time and the model cannot tell whether the library is
    allowed.

    Each candidate path is checked for existence so fresh clones that skip
    the CUTLASS submodule do not break the build — missing paths are
    silently dropped.
    """
    candidates = [
        os.path.join(root_dir, "reference_sources", "cutlass", "include"),
        os.path.join(root_dir, "reference_sources", "cutlass", "tools", "util", "include"),
        os.path.join(root_dir, "reference_sources", "cutlass", "examples", "common"),
    ]
    return [p for p in candidates if os.path.isdir(p)]


def _retarget_cuda_arch_flags(extra_cuda_flags: list[str], target_sm: int) -> list[str]:
    """Replace any existing -gencode pairs with flags for the current GPU arch."""
    retained_flags = []
    i = 0
    while i < len(extra_cuda_flags):
        flag = extra_cuda_flags[i]
        if flag == "-gencode" and i + 1 < len(extra_cuda_flags):
            i += 2
            continue
        retained_flags.append(flag)
        i += 1
    return retained_flags + _current_arch_gencode_flags(target_sm)


def _is_arch_runtime_error(error_text: str) -> bool:
    """Detect runtime failures that indicate the binary is not runnable on this GPU."""
    text = error_text.lower()
    return any(
        pattern in text for pattern in [
            "no kernel image is available for execution on the device",
            "invalid device function",
        ]
    )


def _build_call_args(task_def: dict, inputs: list, outputs: list):
    """
    Build the argument list for calling the compiled CUDA function.

    The function signature determines argument order. Tensor args come from
    inputs and outputs (in order), scalar args from SCALAR_ARGS.
    """
    from .compiler import parse_function_signature

    parsed = parse_function_signature(task_def["FUNCTION_SIGNATURE"])
    scalar_args = task_def["SCALAR_ARGS"]

    # Build name → tensor maps
    input_tensors = {name: tensor for name, tensor in inputs}
    output_tensors = {name: tensor for name, tensor in outputs}
    all_tensors = {**input_tensors, **output_tensors}

    args = []
    for param in parsed["params"]:
        if param.get("is_stream"):
            continue  # Stream is handled by the wrapper
        elif param["is_pointer"]:
            if param["name"] in all_tensors:
                args.append(all_tensors[param["name"]])
            else:
                raise ValueError(f"No tensor found for parameter '{param['name']}'")
        else:
            if param["name"] in scalar_args:
                args.append(scalar_args[param["name"]])
            else:
                raise ValueError(
                    f"No scalar value found for parameter '{param['name']}'. "
                    f"Available: {list(scalar_args.keys())}"
                )

    return args


def check_correctness(
    ref_outputs: list,
    sol_outputs: list,
    tolerances: dict,
) -> tuple[bool, dict]:
    """
    Check if solution outputs match reference outputs within tolerance.

    Args:
        ref_outputs: List of (name, tensor) from reference.
        sol_outputs: List of (name, tensor) from solution.
        tolerances: Dict with 'atol' and 'rtol'.

    Returns:
        (all_correct, per_output_results)
    """
    atol = tolerances.get("atol", 1e-4)
    rtol = tolerances.get("rtol", 1e-4)

    per_output = {}
    for (ref_name, ref_tensor), (sol_name, sol_tensor) in zip(ref_outputs, sol_outputs):
        name = ref_name

        if ref_tensor.shape != sol_tensor.shape:
            per_output[name] = {
                "correct": False,
                "error": f"Shape mismatch: expected {list(ref_tensor.shape)}, got {list(sol_tensor.shape)}",
            }
            continue

        # Compare in float32
        ref_f = ref_tensor.float()
        sol_f = sol_tensor.float()

        correct = torch.allclose(ref_f, sol_f, atol=atol, rtol=rtol)
        info = {"correct": correct}

        if not correct:
            diff = (ref_f - sol_f).abs()
            info["max_diff"] = diff.max().item()
            info["mean_diff"] = diff.mean().item()
            # Find position of max diff
            flat_idx = diff.argmax().item()
            info["ref_at_max"] = ref_f.flatten()[flat_idx].item()
            info["sol_at_max"] = sol_f.flatten()[flat_idx].item()
            info["num_mismatched"] = int((diff > atol + rtol * ref_f.abs()).sum().item())
            info["total_elements"] = int(ref_f.numel())
            print(f"  Output '{name}': MISMATCH (max_diff={info['max_diff']:.6g}, "
                  f"mean_diff={info['mean_diff']:.6g}, "
                  f"mismatched={info['num_mismatched']}/{info['total_elements']})")

        per_output[name] = info

    all_correct = all(v["correct"] if isinstance(v, dict) else v for v in per_output.values())
    return all_correct, per_output


def eval_kernel(
    task_dir: str,
    solution_cu: str,
    measure_performance: bool = True,
    num_correct_trials: int = 3,
    num_perf_trials: int = 10,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    cached_ref_time_ms: float = 0.0,
) -> KernelExecResult:
    """
    Evaluate a CUDA solution against a task's reference.

    Args:
        task_dir: Path to the task directory (containing def.py and optionally reference.cu).
        solution_cu: Either a file path to a .cu file or CUDA source code as string.
        measure_performance: Whether to time the kernels.
        num_correct_trials: Number of random trials for correctness.
        num_perf_trials: Number of trials for performance measurement.
        atol: Absolute tolerance override (uses task default if None).
        rtol: Relative tolerance override (uses task default if None).
        device: CUDA device.
        verbose: Print progress.

    Returns:
        KernelExecResult with compilation, correctness, and performance data.
    """
    if device is None:
        device = torch.cuda.current_device()

    result = KernelExecResult()
    root_dir = get_project_root()
    is_general_task = _is_class2_general_task(task_dir)
    is_class2_task = _is_class2_task(task_dir)
    target_sm = _get_device_sm(device) if is_general_task else 0
    if target_sm > 0:
        result.metadata["target_sm"] = target_sm
        result.metadata["target_arch"] = f"sm_{target_sm}"
        if verbose and is_general_task:
            print(f"[Eval] General task detected; targeting current GPU arch sm_{target_sm}")

    # Load task definition
    task_def = load_task_def(task_dir)
    sig = task_def["FUNCTION_SIGNATURE"]

    # --- Step 1: Compile solution ---
    if verbose:
        print(f"[Eval] Compiling solution...")

    build_dir = tempfile.mkdtemp(prefix="cuda_hercules_sol_")
    try:
        solution_cuda_flags = _current_arch_gencode_flags(target_sm) if target_sm > 0 else None
        solution_include_dirs = get_solution_include_dirs(root_dir)
        sol_fn = compile_cuda_module(
            cuda_source=solution_cu,
            function_signature=sig,
            extra_include_dirs=solution_include_dirs or None,
            extra_cuda_cflags=solution_cuda_flags,
            interpose_cached_allocators=is_class2_task,
            build_dir=build_dir,
            verbose=verbose,
        )
        result.compiled = True
    except Exception as e:
        result.compilation_error = str(e)
        if verbose:
            print(f"[Eval] Solution compilation FAILED: {e}")
        return result

    # --- Step 2: Compile CUDA reference ---
    ref_cuda_fn = None
    ref_cu_path = os.path.join(task_dir, "reference.cu")
    if task_def["HAS_CUDA_REFERENCE"] and os.path.exists(ref_cu_path):
        if verbose:
            print(f"[Eval] Compiling CUDA reference...")
        task_name = os.path.basename(task_dir.rstrip('/'))
        ref_build_dir = os.path.join(root_dir, ".cache", "ref_builds", task_name)
        os.makedirs(ref_build_dir, exist_ok=True)
        try:
            ref_cuda_flags = list(task_def.get("REFERENCE_EXTRA_CUDA_FLAGS", []))
            if target_sm > 0:
                ref_cuda_flags = _retarget_cuda_arch_flags(ref_cuda_flags, target_sm)

            extra_includes = []
            for inc in task_def["REFERENCE_EXTRA_INCLUDES"]:
                if os.path.isabs(inc):
                    inc_path = inc
                else:
                    inc_path = os.path.join(task_dir, inc)
                    if not os.path.isdir(inc_path):
                        inc_path = os.path.join(root_dir, inc)
                if os.path.isdir(inc_path):
                    extra_includes.append(inc_path)

            ref_cuda_fn = compile_cuda_module(
                cuda_source=ref_cu_path,
                function_signature=sig,
                extra_include_dirs=extra_includes,
                extra_cuda_cflags=ref_cuda_flags,
                extra_ldflags=task_def.get("REFERENCE_EXTRA_LDFLAGS", []),
                interpose_cached_allocators=is_class2_task,
                build_dir=ref_build_dir,
                verbose=verbose,
            )
            result.ref_compiled = True
        except Exception as e:
            if target_sm > 0:
                result.runtime_error = (
                    f"{UNSUPPORTED_ARCH_PREFIX} general reference failed to build for target "
                    f"sm_{target_sm}: {e}"
                )
                if verbose:
                    print(f"[Eval] {result.runtime_error}")
                return result
            if verbose:
                print(f"[Eval] Reference CUDA compilation failed: {e}")

    if ref_cuda_fn is None:
        if not result.runtime_error:
            result.runtime_error = "CUDA reference compilation failed; cannot evaluate"
        return result

    # --- Step 3: Check correctness against CUDA reference ---
    tolerances = {"atol": atol, "rtol": rtol}

    if verbose:
        print(f"[Eval] Checking correctness vs CUDA reference ({num_correct_trials} trials, atol={atol}, rtol={rtol})...")

    all_correct = True
    kernel_timeout_sec = 600  # Kill if a single kernel call hangs > 10 min
    for trial in range(num_correct_trials):
        try:
            torch.manual_seed(42 + trial)
            torch.cuda.manual_seed(42 + trial)

            inputs = task_def["get_inputs"]()

            # Run CUDA reference
            ref_outputs = task_def["get_outputs"](inputs)
            ref_args = _build_call_args(task_def, inputs, ref_outputs)
            try:
                ref_cuda_fn(*ref_args)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                if target_sm > 0 and _is_arch_runtime_error(str(e)):
                    result.runtime_error = (
                        f"{UNSUPPORTED_ARCH_PREFIX} general reference launch failed on target "
                        f"sm_{target_sm}: {e}"
                    )
                else:
                    result.runtime_error = f"Trial {trial + 1}: reference failed: {traceback.format_exc()}"
                all_correct = False
                break

            # Run solution with timeout to catch infinite-loop kernels
            sol_outputs = task_def["get_outputs"](inputs)
            sol_args = _build_call_args(task_def, inputs, sol_outputs)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(kernel_timeout_sec)
            try:
                sol_fn(*sol_args)
                torch.cuda.synchronize(device=device)
            except Exception:
                result.runtime_error = f"Trial {trial + 1}: solution failed: {traceback.format_exc()}"
                all_correct = False
                break
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            # Compare solution vs CUDA reference
            correct, per_output = check_correctness(
                ref_outputs, sol_outputs, tolerances
            )
            result.per_output_correctness = per_output

            if not correct:
                all_correct = False
                if verbose:
                    print(f"  Trial {trial + 1}: FAILED")
                break
            elif verbose:
                print(f"  Trial {trial + 1}: PASSED")

        except _KernelTimeout:
            result.runtime_error = f"Trial {trial + 1}: kernel execution timed out ({kernel_timeout_sec}s)"
            all_correct = False
            if verbose:
                print(f"  Trial {trial + 1}: TIMEOUT")
            break
        except Exception as e:
            result.runtime_error = f"Trial {trial + 1}: {traceback.format_exc()}"
            all_correct = False
            break

    result.correctness = all_correct

    # --- Step 4: Measure performance ---
    if measure_performance and result.correctness:
        if verbose:
            print(f"[Eval] Measuring performance ({num_perf_trials} trials)...")

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        inputs = task_def["get_inputs"]()

        # Time solution (wrap with 10-min SIGALRM — all kernel trials combined)
        sol_outputs = task_def["get_outputs"](inputs)
        sol_args = _build_call_args(task_def, inputs, sol_outputs)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(kernel_timeout_sec)
        try:
            result.solution_time = time_cuda_function(
                sol_fn, sol_args, num_trials=num_perf_trials, verbose=verbose, device=device
            )
        except _KernelTimeout:
            result.runtime_error = f"Performance measurement (solution) timed out ({kernel_timeout_sec}s)"
            if verbose:
                print(f"[Eval] Solution perf timed out after {kernel_timeout_sec}s")
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        # Reference timing: use cached value or measure live
        if cached_ref_time_ms > 0:
            result.reference_time = TimingStats(
                mean_ms=cached_ref_time_ms,
                std_ms=0.0,
                min_ms=cached_ref_time_ms,
                max_ms=cached_ref_time_ms,
                num_trials=1,
                all_times_ms=[cached_ref_time_ms],
            )
            if verbose:
                print(f"[Eval] Using cached ref time: {cached_ref_time_ms:.4f} ms")
        else:
            ref_outputs_tensors = task_def["get_outputs"](inputs)
            ref_args = _build_call_args(task_def, inputs, ref_outputs_tensors)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(kernel_timeout_sec)
            try:
                result.reference_time = time_cuda_function(
                    ref_cuda_fn, ref_args, num_trials=num_perf_trials, verbose=verbose, device=device
                )
            except _KernelTimeout:
                result.runtime_error = f"Performance measurement (reference) timed out ({kernel_timeout_sec}s)"
                if verbose:
                    print(f"[Eval] Reference perf timed out after {kernel_timeout_sec}s")
                return result
            except Exception as e:
                if target_sm > 0 and _is_arch_runtime_error(str(e)):
                    result.runtime_error = (
                        f"{UNSUPPORTED_ARCH_PREFIX} general reference perf launch failed on target "
                        f"sm_{target_sm}: {e}"
                    )
                else:
                    result.runtime_error = f"Performance measurement (reference) failed: {traceback.format_exc()}"
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        if result.reference_time.mean_ms > 0:
            result.speedup = result.reference_time.mean_ms / result.solution_time.mean_ms

    return result
