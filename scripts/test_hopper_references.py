#!/usr/bin/env python3
"""Test all Hopper reference implementations (Class 1 + Class 2)."""

import os
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_class1_hopper():
    """Test Class 1 Hopper tasks by running `make ref` and executing."""
    hopper_dir = PROJECT_ROOT / "tasks" / "class1" / "hopper"
    if not hopper_dir.exists():
        print("No Class 1 Hopper tasks found")
        return [], []

    tasks = sorted(d for d in hopper_dir.iterdir() if d.is_dir())
    passed, failed = [], []

    for task_dir in tasks:
        makefile = task_dir / "Makefile"
        if not makefile.exists():
            continue

        name = task_dir.name
        print(f"\n{'='*60}")
        print(f"[Class 1] {name}")
        print(f"{'='*60}")

        # Clean first
        subprocess.run(["make", "clean"], cwd=task_dir, capture_output=True)

        # Build reference
        t0 = time.time()
        result = subprocess.run(
            ["make", "ref"],
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        build_time = time.time() - t0

        if result.returncode != 0:
            print(f"  BUILD FAILED ({build_time:.1f}s)")
            print(f"  stderr: {result.stderr[-500:]}")
            failed.append((name, "build", result.stderr[-200:]))
            continue

        print(f"  Build OK ({build_time:.1f}s)")

        # Run reference
        ref_binary = task_dir / "ref"
        if not ref_binary.exists():
            failed.append((name, "no binary", ""))
            continue

        try:
            t0 = time.time()
            result = subprocess.run(
                ["./ref"],
                cwd=task_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            run_time = time.time() - t0

            if result.returncode != 0:
                print(f"  RUN FAILED ({run_time:.1f}s)")
                print(f"  stderr: {result.stderr[-500:]}")
                failed.append((name, "runtime", result.stderr[-200:]))
            else:
                print(f"  Run OK ({run_time:.1f}s)")
                for line in result.stdout.splitlines():
                    if any(kw in line.lower() for kw in ["time", "gflops", "tflops", "passed"]):
                        print(f"  {line.strip()}")
                passed.append(name)
        except subprocess.TimeoutExpired:
            print(f"  RUN TIMEOUT (>600s)")
            failed.append((name, "timeout", "exceeded 600s"))

        # Cleanup binary
        subprocess.run(["make", "clean"], cwd=task_dir, capture_output=True)

    return passed, failed


def test_class2_hopper():
    """Test Class 2 Hopper tasks by compiling and running CUDA references."""
    hopper_dir = PROJECT_ROOT / "tasks" / "class2" / "hopper"
    if not hopper_dir.exists():
        print("No Class 2 Hopper tasks found")
        return [], []

    # Add project to path
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cuda_hercules.eval import load_task_def
    from cuda_hercules.compiler import compile_cuda_module
    from cuda_hercules.utils import get_project_root
    import torch

    tasks = sorted(d for d in hopper_dir.iterdir() if d.is_dir())
    passed, failed = [], []
    root_dir = get_project_root()

    # Handle Makefile-based tasks (e.g. TK tasks with backend: class1_make)
    for task_dir in tasks:
        if not (task_dir / "Makefile").exists() or (task_dir / "def.py").exists():
            continue

        name = task_dir.name
        print(f"\n{'='*60}")
        print(f"[Class 2 / Makefile] {name}")
        print(f"{'='*60}")

        subprocess.run(["make", "clean"], cwd=task_dir, capture_output=True)
        t0 = time.time()
        result = subprocess.run(
            ["make", "ref"], cwd=task_dir, capture_output=True, text=True, timeout=300,
        )
        build_time = time.time() - t0
        if result.returncode != 0:
            print(f"  BUILD FAILED ({build_time:.1f}s)")
            print(f"  stderr: {result.stderr[-500:]}")
            failed.append((name, "build", result.stderr[-200:]))
            continue
        print(f"  Build OK ({build_time:.1f}s)")

        ref_binary = task_dir / "ref"
        if not ref_binary.exists():
            failed.append((name, "no binary", ""))
            continue
        try:
            t0 = time.time()
            result = subprocess.run(
                ["./ref"], cwd=task_dir, capture_output=True, text=True, timeout=600,
            )
            run_time = time.time() - t0
            if result.returncode != 0:
                print(f"  RUN FAILED ({run_time:.1f}s)")
                print(f"  stderr: {result.stderr[-500:]}")
                failed.append((name, "runtime", result.stderr[-200:]))
            else:
                print(f"  Run OK ({run_time:.1f}s)")
                for line in result.stdout.splitlines():
                    if any(kw in line.lower() for kw in ["time", "gflops", "tflops", "passed"]):
                        print(f"  {line.strip()}")
                passed.append(name)
        except subprocess.TimeoutExpired:
            print(f"  RUN TIMEOUT (>600s)")
            failed.append((name, "timeout", "exceeded 600s"))
        subprocess.run(["make", "clean"], cwd=task_dir, capture_output=True)

    # Handle def.py-based tasks (Class 2 pybind11 pipeline)
    for task_dir in tasks:
        def_py = task_dir / "def.py"
        ref_cu = task_dir / "reference.cu"
        if not def_py.exists():
            continue

        name = task_dir.name
        print(f"\n{'='*60}")
        print(f"[Class 2] {name}")
        print(f"{'='*60}")

        try:
            task_def = load_task_def(str(task_dir))
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            failed.append((name, "load", str(e)[:200]))
            continue

        if not task_def["HAS_CUDA_REFERENCE"] or not ref_cu.exists():
            print(f"  SKIP: no CUDA reference")
            continue

        # Compile reference
        t0 = time.time()
        try:
            extra_includes = []
            for inc in task_def["REFERENCE_EXTRA_INCLUDES"]:
                if os.path.isabs(inc):
                    inc_path = inc
                else:
                    inc_path = os.path.join(str(task_dir), inc)
                    if not os.path.isdir(inc_path):
                        inc_path = os.path.join(root_dir, inc)
                if os.path.isdir(inc_path):
                    extra_includes.append(inc_path)

            build_dir = os.path.join(root_dir, ".cache", "ref_builds", name)
            os.makedirs(build_dir, exist_ok=True)

            ref_fn = compile_cuda_module(
                cuda_source=str(ref_cu),
                function_signature=task_def["FUNCTION_SIGNATURE"],
                extra_include_dirs=extra_includes,
                extra_cuda_cflags=task_def.get("REFERENCE_EXTRA_CUDA_FLAGS", []),
                extra_ldflags=task_def.get("REFERENCE_EXTRA_LDFLAGS", []),
                build_dir=build_dir,
                verbose=False,
            )
            build_time = time.time() - t0
            print(f"  Compile OK ({build_time:.1f}s)")
        except Exception as e:
            build_time = time.time() - t0
            print(f"  COMPILE FAILED ({build_time:.1f}s): {e}")
            failed.append((name, "compile", str(e)[:200]))
            continue

        # Run reference with test inputs
        try:
            from cuda_hercules.eval import _build_call_args

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            inputs = task_def["get_inputs"]()
            outputs = task_def["get_outputs"](inputs)
            args = _build_call_args(task_def, inputs, outputs)
            ref_fn(*args)
            torch.cuda.synchronize()
            print(f"  Run OK")
            passed.append(name)
        except Exception as e:
            print(f"  RUN FAILED: {e}")
            failed.append((name, "runtime", str(e)[:200]))

    return passed, failed


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hopper Reference Implementations")
    print("=" * 60)

    c1_passed, c1_failed = test_class1_hopper()
    c2_passed, c2_failed = test_class2_hopper()

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_passed = len(c1_passed) + len(c2_passed)
    total_failed = len(c1_failed) + len(c2_failed)

    print(f"\nClass 1 Hopper: {len(c1_passed)} passed, {len(c1_failed)} failed")
    print(f"Class 2 Hopper: {len(c2_passed)} passed, {len(c2_failed)} failed")
    print(f"Total: {total_passed} passed, {total_failed} failed")

    if c1_failed:
        print(f"\nClass 1 failures:")
        for name, stage, err in c1_failed:
            print(f"  {name} [{stage}]: {err[:100]}")

    if c2_failed:
        print(f"\nClass 2 failures:")
        for name, stage, err in c2_failed:
            print(f"  {name} [{stage}]: {err[:100]}")

    sys.exit(1 if total_failed > 0 else 0)
