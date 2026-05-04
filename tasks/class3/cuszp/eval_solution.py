#!/usr/bin/env python3
"""
Evaluate an LLM-generated solution for the cuSZp task.

Usage:
    # Single file (backwards compatible):
    python eval_solution.py <path/to/cuSZp_kernels_1D_f32.cu>

    # Directory containing all 6 kernel files:
    python eval_solution.py <path/to/solution_dir/>

Creates a temp copy of the task, replaces solution files, builds,
runs the benchmark, and prints a JSON result with per-kernel metrics.

Strict mode: the on-disk `src/cuSZp_kernels_*.cu` files start as `#error`
placeholders so candidate solutions cannot silently inherit the SC'23
reference. This evaluator does NOT touch `reference/` — any solution_file
the candidate omits stays as a placeholder, and `make` will hard-fail with
a `CUDA_HERCULES_PLACEHOLDER_*` directive (see the `compiled=False`
branch). The SC'23 reference baseline is measured separately by
`scripts/eval_cuszp_toolaug.py` / `scripts/replay_cuszp_planc.py` in an
isolated tmp dir; this evaluator only times candidate solutions.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

TASK_DIR = os.path.dirname(os.path.abspath(__file__))

SOLUTION_FILES = [
    "cuSZp_kernels_1D_f32.cu",
    "cuSZp_kernels_1D_f64.cu",
    "cuSZp_kernels_2D_f32.cu",
    "cuSZp_kernels_2D_f64.cu",
    "cuSZp_kernels_3D_f32.cu",
    "cuSZp_kernels_3D_f64.cu",
]

# Map kernel file -> which variant/modes it covers
KERNEL_FILE_VARIANTS = {
    "cuSZp_kernels_1D_f32.cu": ["1D_f32"],
    "cuSZp_kernels_1D_f64.cu": ["1D_f64"],
    "cuSZp_kernels_2D_f32.cu": ["2D_f32"],
    "cuSZp_kernels_2D_f64.cu": ["2D_f64"],
    "cuSZp_kernels_3D_f32.cu": ["3D_f32"],
    "cuSZp_kernels_3D_f64.cu": ["3D_f64"],
}

IGNORE_PATTERNS = shutil.ignore_patterns(
    "build", "cmake-build-release", "cmake-build-*", "__pycache__",
    "*.o", "*.so", "*.a",
    # Never copy reference/ into a candidate work_dir — that would defeat
    # the point of the placeholders. Baseline measurement is the harness's
    # job, not this evaluator's.
    "reference",
)

# Regex to parse structured KERNEL lines from main.cu output
# Format: KERNEL <variant> <mode>: correct=<0|1> errors=<N> max_error=<f> error_bound=<f>
#         cmp_ms=<f> dec_ms=<f> ratio=<f> cmp_gbps=<f> dec_gbps=<f> nbEle=<N>
# Format: KERNEL <variant> <mode> eb=<rel_eb>: correct=... errors=... ...
KERNEL_RE = re.compile(
    r"KERNEL\s+(\S+)\s+(\S+)\s+eb=(\S+):\s+"
    r"correct=(\d+)\s+"
    r"errors=(\d+)\s+"
    r"max_error=([0-9.eE+\-]+)\s+"
    r"error_bound=([0-9.eE+\-]+)\s+"
    r"err_ratio=([0-9.]+)\s+"
    r"cmp_ms=([0-9.]+)\s+"
    r"dec_ms=([0-9.]+)\s+"
    r"ratio=([0-9.]+)\s+"
    r"cmp_gbps=([0-9.]+)\s+"
    r"dec_gbps=([0-9.]+)\s+"
    r"nbEle=(\d+)"
)


def parse_kernel_results(output):
    """Parse per-kernel structured output lines into a dict."""
    kernels = {}
    for m in KERNEL_RE.finditer(output):
        variant = m.group(1)    # e.g. "1D_f32"
        mode = m.group(2)       # e.g. "fixed"
        rel_eb = m.group(3)     # e.g. "1E-2"
        key = f"{variant}_{mode}_eb{rel_eb}"
        kernels[key] = {
            "variant": variant,
            "mode": mode,
            "rel_error_bound": rel_eb,
            "correct": int(m.group(4)) == 1,
            "error_count": int(m.group(5)),
            "max_error": float(m.group(6)),
            "error_bound": float(m.group(7)),
            "error_ratio": float(m.group(8)),
            "compress_ms": float(m.group(9)),
            "decompress_ms": float(m.group(10)),
            "compression_ratio": float(m.group(11)),
            "compress_gbps": float(m.group(12)),
            "decompress_gbps": float(m.group(13)),
            "num_elements": int(m.group(14)),
        }
    return kernels


def main():
    parser = argparse.ArgumentParser(description="Evaluate cuSZp solution")
    parser.add_argument("solution", help="Path to solution file or directory")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Execution timeout in seconds")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Do not delete temp directory after evaluation")
    args = parser.parse_args()

    solution_path = os.path.abspath(args.solution)

    # Determine if solution is a directory (multi-file) or single file
    if os.path.isdir(solution_path):
        sol_files = {}
        for f in SOLUTION_FILES:
            fp = os.path.join(solution_path, f)
            if os.path.isfile(fp):
                sol_files[f] = fp
        if not sol_files:
            print(json.dumps({"compiled": False, "correct": False,
                              "kernel_time_ms": -1, "kernels": {},
                              "output": f"No solution files found in {solution_path}"}))
            sys.exit(1)
    elif os.path.isfile(solution_path):
        basename = os.path.basename(solution_path)
        if basename in SOLUTION_FILES:
            sol_files = {basename: solution_path}
        else:
            sol_files = {"cuSZp_kernels_1D_f32.cu": solution_path}
    else:
        print(json.dumps({"compiled": False, "correct": False,
                          "kernel_time_ms": -1, "kernels": {},
                          "output": f"Solution not found: {solution_path}"}))
        sys.exit(1)

    # Determine which variants are being tested (based on which files were replaced)
    replaced_variants = set()
    for f in sol_files:
        for v in KERNEL_FILE_VARIANTS.get(f, []):
            replaced_variants.add(v)

    tmp_dir = tempfile.mkdtemp(prefix="kh_cuszp_eval_")
    task_copy = os.path.join(tmp_dir, "task")

    try:
        # 1. Copy task directory
        shutil.copytree(TASK_DIR, task_copy, ignore=IGNORE_PATTERNS)

        # 2. Replace solution files
        for filename, src_path in sol_files.items():
            dst = os.path.join(task_copy, "src", filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src_path, dst)

        # 3. Build
        build_dir = os.path.join(task_copy, "src", "cmake-build-release")
        os.makedirs(build_dir, exist_ok=True)

        try:
            subprocess.check_call(
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                cwd=build_dir,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            result_make = subprocess.run(
                ["make", f"-j{os.cpu_count()}"],
                cwd=build_dir,
                capture_output=True, text=True,
            )
            if result_make.returncode != 0:
                print(json.dumps({
                    "compiled": False, "correct": False,
                    "kernel_time_ms": -1, "kernels": {},
                    "output": f"Build failed:\n{result_make.stderr[-2000:]}",
                }))
                sys.exit(1)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(json.dumps({
                "compiled": False, "correct": False,
                "kernel_time_ms": -1, "kernels": {},
                "output": f"Build error: {e}",
            }))
            sys.exit(1)

        compiled = True

        # 4. Run benchmark
        executable = os.path.join(build_dir, "cuSZp_bench")
        if not os.path.isfile(executable):
            print(json.dumps({
                "compiled": False, "correct": False,
                "kernel_time_ms": -1, "kernels": {},
                "output": "Executable not found after build",
            }))
            sys.exit(1)

        env = os.environ.copy()
        try:
            proc = subprocess.run(
                [executable],
                capture_output=True, text=True,
                env=env, timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            print(json.dumps({
                "compiled": True, "correct": False,
                "kernel_time_ms": -1, "kernels": {},
                "output": f"Execution timed out ({args.timeout}s)",
            }))
            sys.exit(1)

        output = proc.stdout + proc.stderr

        # 5. Parse per-kernel results
        kernels = parse_kernel_results(output)

        # 6. Compute overall metrics
        # Only check correctness for variants whose kernel file was replaced
        all_correct = True
        total_kernel_ms = 0.0
        num_passed = 0
        num_tested = 0

        for key, k in kernels.items():
            total_kernel_ms += k["compress_ms"] + k["decompress_ms"]
            # Only count replaced variants towards correctness
            if k["variant"] in replaced_variants:
                num_tested += 1
                if k["correct"]:
                    num_passed += 1
                else:
                    all_correct = False

        # If no replaced variants matched any kernel output, fall back to overall check
        if num_tested == 0:
            all_correct = "Passed" in output and proc.returncode == 0

        # Also parse total kernel time from summary line
        m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", output)
        kernel_time_ms = float(m.group(1)) if m else total_kernel_ms

        result = {
            "compiled": compiled,
            "correct": all_correct,
            "kernel_time_ms": kernel_time_ms,
            "num_kernels_tested": num_tested,
            "num_kernels_passed": num_passed,
            "replaced_files": list(sol_files.keys()),
            "kernels": kernels,
            "output": output[-3000:],
        }

        print(json.dumps(result))
        sys.exit(0 if all_correct else 1)

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"Temp directory kept at: {tmp_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
