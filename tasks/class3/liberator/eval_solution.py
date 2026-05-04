#!/usr/bin/env python3
"""
Evaluate a black-box Liberator backend candidate.

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
EXPECTED_SO_NAME = "libkh_liberator_backend.so"
SOURCE_EXTS = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".md", ".txt"}

IGNORE_PATTERNS = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "custom_cuda_backend",
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
                "speedup": 0.0,
                "output": output or message,
                "error": message,
            }
        )
    )
    return 1


def _parse_total_ms(output: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.]+)\s*ms", output)
    return float(match.group(1)) if match else -1.0


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
    staged = []
    os.makedirs(backend_dir, exist_ok=True)
    for rel_dst, src in files.items():
        dst = os.path.join(backend_dir, rel_dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        staged.append(rel_dst)
    return sorted(staged)


def _fixed_build_output_path(backend_dir: str) -> str:
    return os.path.join(backend_dir, EXPECTED_SO_NAME)


def _run_fixed_backend_build(backend_dir: str, timeout: int) -> tuple[bool, str]:
    if not os.path.isfile(os.path.join(backend_dir, "backend.cu")):
        return False, "backend.cu not found"

    env = os.environ.copy()
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    output_so = _fixed_build_output_path(backend_dir)

    cmd = [
        nvcc,
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-std=c++17",
        "-O3",
        "-I.",
        f"-I{TASK_DIR}",
        f"-I{cuda_home}/include",
        f"-I{cuda_home}/targets/x86_64-linux/include",
        "backend.cu",
        "-lcudart",
        "-lcuda",
        "-o",
        output_so,
    ]
    if "CUDA_GENCODE" in env:
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


def _prune_task_copy(task_copy: str) -> None:
    # copytree(ignore=...) matches basenames in each directory listing, so a
    # nested path like src/cmake-build-release will still get copied. Remove
    # known build artifacts explicitly after staging the task copy.
    for relpath in (
        os.path.join("src", "cmake-build-release"),
        os.path.join("src", "build"),
        "build",
    ):
        shutil.rmtree(os.path.join(task_copy, relpath), ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Liberator backend candidate")
    parser.add_argument("solutions", nargs="+", help="Candidate source files or a directory")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-step timeout in seconds")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    mode, files, err = _collect_inputs(args.solutions)
    if err:
        return _json_fail(err)

    scratch_root = os.path.join(PROJECT_ROOT, ".tmp")
    os.makedirs(scratch_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="kh_liberator_eval_", dir=scratch_root)
    task_copy = os.path.join(tmp_dir, "task")

    try:
        shutil.copytree(TASK_DIR, task_copy, ignore=IGNORE_PATTERNS)
        _prune_task_copy(task_copy)
        backend_dir = os.path.join(task_copy, "custom_cuda_backend")
        staged_files = _stage_files(backend_dir, files)
        if not staged_files:
            return _json_fail("No candidate files were staged", mode=mode)

        compiled = mode == "prebuilt_so"
        if mode == "source":
            compiled, build_error = _run_fixed_backend_build(backend_dir, args.timeout)
            if not compiled:
                return _json_fail(build_error, mode=mode)
            if not _has_backend_artifact(backend_dir):
                return _json_fail(f"Build completed but produced no {EXPECTED_SO_NAME}", mode=mode)

        proc = subprocess.run(
            [sys.executable, "run.py"],
            cwd=task_copy,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
        output = proc.stdout + proc.stderr
        baseline_ms = _parse_total_ms(output, "Baseline total")
        kernel_time_ms = _parse_total_ms(output, "Solution total")
        correct = "--- Custom CUDA Backend ---" in output and "FAILED correctness check" not in output and proc.returncode == 0
        speedup = baseline_ms / kernel_time_ms if correct and baseline_ms > 0 and kernel_time_ms > 0 else 0.0

        result = {
            "compiled": compiled,
            "correct": correct,
            "mode": mode,
            "kernel_time_ms": kernel_time_ms if correct else -1,
            "ref_time_ms": baseline_ms,
            "speedup": speedup,
            "staged_files": staged_files,
            "output": output[-4000:],
        }
        if not correct:
            result["error"] = output[-4000:] or "Custom backend failed correctness or timing"

        print(json.dumps(result))
        return 0 if correct else 1

    except subprocess.TimeoutExpired:
        return _json_fail(f"Evaluation timed out after {args.timeout}s", compiled=(mode == "prebuilt_so"), mode=mode)
    finally:
        if args.keep_tmp:
            print(f"Temp directory kept at: {tmp_dir}", file=sys.stderr)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
