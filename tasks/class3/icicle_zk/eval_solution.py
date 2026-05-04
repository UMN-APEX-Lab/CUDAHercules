#!/usr/bin/env python3
"""
Evaluate an Icicle ZK custom backend candidate.

Primary mode:
    model-generated backend.cu -> fixed nvcc build -> benchmark

Debug mode:
    prebuilt shared library -> stage -> benchmark
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TASK_DIR, "..", "..", ".."))
ICICLE_SRC_DIR = os.path.join(PROJECT_ROOT, "reference_sources", "icicle")
ICICLE_BUILD_DIR = os.path.join(ICICLE_SRC_DIR, "build")
EXPECTED_SO_NAME = "libkh_custom_backend.so"

SOURCE_EXTS = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".md", ".txt"}

IGNORE_PATTERNS = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "custom_cuda_backend",
    "src/build",
    "build",
)


def _json_fail(message: str, compiled: bool = False, output: str = "", mode: str = "") -> int:
    print(
        json.dumps(
            {
                "compiled": compiled,
                "correct": False,
                "mode": mode,
                "kernel_time_ms": -1,
                "ref_time_ms": -1,
                "gpu_ref_time_ms": -1,
                "baseline_time_ms": -1,
                "cpu_time_ms": -1,
                "speedup": 0.0,
                "speedup_vs_gpu_ref": 0.0,
                "output": output or message,
                "error": message,
            }
        )
    )
    return 1


def _parse_section_time_ms(output: str, header: str) -> float:
    pattern = rf"{re.escape(header)}.*?E2E:\s*([0-9.]+)\s*ms"
    match = re.search(pattern, output, re.DOTALL)
    return float(match.group(1)) if match else -1.0


def _custom_backend_passed(output: str) -> bool:
    if "--- Custom CUDA Backend (Solution) ---" not in output:
        return False
    if "Custom CUDA backend FAILED correctness check" in output:
        return False
    return _parse_section_time_ms(output, "--- Custom CUDA Backend (Solution) ---") > 0


def _normalize_relpath(relpath: str) -> str:
    relpath = relpath.replace("\\", "/")
    prefixes = ("custom_cuda_backend/", "./custom_cuda_backend/")
    for prefix in prefixes:
        if relpath.startswith(prefix):
            return relpath[len(prefix):]
    return relpath


def _collect_dir_files(root: str) -> tuple[dict[str, str], dict[str, str]]:
    sources: dict[str, str] = {}
    shared_libs: dict[str, str] = {}

    for dirpath, _, files in os.walk(root):
        for name in files:
            src = os.path.join(dirpath, name)
            rel = _normalize_relpath(os.path.relpath(src, root))
            ext = os.path.splitext(name)[1]
            if name.endswith(".so"):
                shared_libs[rel] = src
            elif ext in SOURCE_EXTS:
                sources[rel] = src

    return sources, shared_libs


def _collect_inputs(inputs: list[str]) -> tuple[str, dict[str, str], str]:
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        root = os.path.abspath(inputs[0])
        sources, shared_libs = _collect_dir_files(root)
        if shared_libs and not sources:
            return "prebuilt_so", shared_libs, ""
        if sources:
            return "source", sources, ""
        return "", {}, f"No source files or .so files found in {root}"

    sources: dict[str, str] = {}
    shared_libs: dict[str, str] = {}
    saw_source = False
    saw_so = False

    for item in inputs:
        src = os.path.abspath(item)
        if not os.path.isfile(src):
            return "", {}, f"Solution file not found: {src}"

        basename = os.path.basename(src)
        ext = os.path.splitext(basename)[1]

        if src.endswith(".so"):
            saw_so = True
            shared_libs[basename] = src
        elif ext in SOURCE_EXTS:
            saw_source = True
            sources[basename] = src
        else:
            return "", {}, f"Unsupported solution file type: {src}"

    if saw_source and saw_so:
        return "", {}, "Do not mix source files and prebuilt .so files in one evaluation"
    if saw_so:
        return "prebuilt_so", shared_libs, ""
    if saw_source:
        return "source", sources, ""
    return "", {}, "No candidate files were provided"


def _stage_files(backend_dir: str, files: dict[str, str]) -> list[str]:
    staged: list[str] = []
    os.makedirs(backend_dir, exist_ok=True)
    for rel_dst, src in files.items():
        dst = os.path.join(backend_dir, rel_dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        staged.append(rel_dst)
    return sorted(staged)


def _write_support_headers(backend_dir: str) -> None:
    gpu_utils_dir = os.path.join(backend_dir, "gpu-utils")
    os.makedirs(gpu_utils_dir, exist_ok=True)

    sharedmem_path = os.path.join(gpu_utils_dir, "sharedmem.h")
    if not os.path.isfile(sharedmem_path):
        with open(sharedmem_path, "w") as f:
            f.write("#pragma once\n\ntemplate <typename T>\nstruct SharedMemory;\n")

    cuda_math_path = os.path.join(backend_dir, "cuda_math.h")
    if not os.path.isfile(cuda_math_path):
        with open(cuda_math_path, "w") as f:
            f.write('#pragma once\n#include "icicle/math/host_math.h"\nnamespace cuda_math = host_math;\n')


def _ensure_icicle_core(timeout: int) -> tuple[bool, str]:
    try:
        subprocess.run(
            [sys.executable, "-c", "import run; run.build_icicle()"],
            cwd=TASK_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"Icicle core build failed:\n{(e.stderr or e.stdout)[-4000:]}"
    except subprocess.TimeoutExpired:
        return False, f"Icicle core build timed out after {timeout}s"


def _fixed_build_output_path(backend_dir: str) -> str:
    return os.path.join(backend_dir, EXPECTED_SO_NAME)


def _run_fixed_backend_build(backend_dir: str, env: dict[str, str], timeout: int) -> tuple[bool, str]:
    if not os.path.isfile(os.path.join(backend_dir, "backend.cu")):
        return False, "backend.cu not found"

    nvcc = os.path.join(env.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
    output_so = _fixed_build_output_path(backend_dir)

    cmd = [
        nvcc,
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-std=c++17",
        "-O3",
        "-use_fast_math",
        "-I.",
        f"-I{TASK_DIR}",
        f"-I{env['ICICLE_SRC_DIR']}/include",
        f"-I{env['CUDA_HOME']}/include",
        f"-I{env['CUDA_HOME']}/targets/x86_64-linux/include",
        "backend.cu",
        f"-L{env['ICICLE_BUILD_DIR']}",
        "-licicle_device",
        "-licicle_field_bn254",
        "-licicle_curve_bn254",
        "-lcudart",
        "-lcuda",
        "-Xlinker",
        f"-rpath={env['ICICLE_BUILD_DIR']}",
        "-o",
        output_so,
    ]
    if env.get("CUDA_GENCODE"):
        cmd[6:6] = shlex.split(env["CUDA_GENCODE"])

    proc = subprocess.run(
        cmd,
        cwd=backend_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        return False, f"Fixed backend build failed:\n{(proc.stderr or proc.stdout)[-4000:]}"
    return True, ""


def _has_backend_artifact(backend_dir: str) -> bool:
    return os.path.isfile(_fixed_build_output_path(backend_dir))


def _build_source_backend(task_copy: str, timeout: int) -> tuple[bool, str]:
    ok, err = _ensure_icicle_core(timeout)
    if not ok:
        return False, err

    backend_dir = os.path.join(task_copy, "custom_cuda_backend")
    _write_support_headers(backend_dir)
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env["ICICLE_SRC_DIR"] = ICICLE_SRC_DIR
    env["ICICLE_BUILD_DIR"] = ICICLE_BUILD_DIR
    if "CUDA_GENCODE" in os.environ:
        env["CUDA_GENCODE"] = os.environ["CUDA_GENCODE"]

    ok, err = _run_fixed_backend_build(backend_dir, env, timeout)
    if not ok:
        return False, err
    if not _has_backend_artifact(backend_dir):
        return False, f"Build completed but produced no {EXPECTED_SO_NAME}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an Icicle ZK backend candidate")
    parser.add_argument(
        "solutions",
        nargs="+",
        help="A directory or files containing model-generated backend sources or prebuilt .so files",
    )
    parser.add_argument("--timeout", type=int, default=1200, help="Per-step timeout in seconds")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the temporary task copy")
    args = parser.parse_args()

    mode, files, err = _collect_inputs(args.solutions)
    if err:
        return _json_fail(err)

    scratch_root = os.path.join(PROJECT_ROOT, ".tmp")
    os.makedirs(scratch_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="kh_icicle_zk_eval_", dir=scratch_root)
    task_copy = os.path.join(tmp_dir, "task")

    try:
        shutil.copytree(TASK_DIR, task_copy, ignore=IGNORE_PATTERNS)

        backend_dir = os.path.join(task_copy, "custom_cuda_backend")
        staged_files = _stage_files(backend_dir, files)
        if not staged_files:
            return _json_fail("No candidate files were staged", mode=mode)

        compiled = mode == "prebuilt_so"
        if mode == "source":
            compiled, build_error = _build_source_backend(task_copy, args.timeout)
            if not compiled:
                return _json_fail(build_error, mode=mode)

        proc = subprocess.run(
            [sys.executable, "run.py"],
            cwd=task_copy,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
        output = proc.stdout + proc.stderr

        cpu_time_ms = _parse_section_time_ms(output, "--- CPU Reference ---")
        baseline_time_ms = _parse_section_time_ms(output, "--- CUDA Baseline (Icicle) ---")
        solution_time_ms = _parse_section_time_ms(output, "--- Custom CUDA Backend (Solution) ---")
        custom_passed = _custom_backend_passed(output)

        gpu_ref_time_ms = baseline_time_ms
        ref_time_ms = gpu_ref_time_ms if gpu_ref_time_ms > 0 else cpu_time_ms
        speedup = (
            ref_time_ms / solution_time_ms
            if (custom_passed and ref_time_ms > 0 and solution_time_ms > 0)
            else 0.0
        )

        result = {
            "compiled": compiled,
            "correct": custom_passed,
            "mode": mode,
            "kernel_time_ms": solution_time_ms if custom_passed else -1,
            "ref_time_ms": ref_time_ms,
            "gpu_ref_time_ms": gpu_ref_time_ms,
            "baseline_time_ms": baseline_time_ms,
            "cpu_time_ms": cpu_time_ms,
            "speedup": speedup,
            "speedup_vs_gpu_ref": speedup,
            "staged_files": staged_files,
            "output": output[-4000:],
        }
        if not custom_passed:
            result["error"] = "Custom backend failed correctness or did not produce a valid timing result"

        print(json.dumps(result))
        return 0 if custom_passed else 1

    except subprocess.TimeoutExpired:
        return _json_fail(
            f"Evaluation timed out after {args.timeout}s",
            compiled=(mode == "prebuilt_so"),
            mode=mode,
        )
    finally:
        if args.keep_tmp:
            print(f"Temp directory kept at: {tmp_dir}", file=sys.stderr)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
