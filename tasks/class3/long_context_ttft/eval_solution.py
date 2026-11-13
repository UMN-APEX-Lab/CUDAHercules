#!/usr/bin/env python3
"""Evaluate a solution of the long-context TTFT task.

    python eval_solution.py <solution_dir>     # the directory holding hercules_plugin/ (the task's src/)

Checks the sources, serves the model with the solution's plugin (run.py candidate), measures the
reference under the same conditions (run.py reference, or the JSON in $HERCULES_LCT_REFERENCE_JSON),
and prints one JSON object: compiled, correct, speedup (geometric mean over the three prompt lengths
of reference TTFT / solution TTFT), per-length times and the quality report.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_EXT = (".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp")
# New compute must be CUDA C++ written by the solution: no Python kernel DSLs, no third-party
# attention/GEMM kernels, no external solver libraries in the plugin.
BLOCKED = [
    (r"^\s*(import|from)\s+(triton|cutlass|tilelang|numba|cupy|taichi|warp|thunderkittens)\b", "Python kernel DSL / JIT"),
    (r"\bcute\.(compile|kernel|jit)\b", "CuTe DSL kernel"),
    (r"^\s*(import|from)\s+(flashinfer|flash_attn|xformers|sageattention|sgl_kernel)\b", "third-party attention kernels"),
    (r"\b(scaled_dot_product_attention|flex_attention)\b", "PyTorch attention kernels"),
    (r"#include\s*[<\"](cublas|cudnn|cutlass/(gemm|conv)/(device|kernel|collective))", "prebuilt GEMM/attention library"),
    (r"\bcublas[A-Z]\w*\s*\(|\bcudnn[A-Z]\w*\s*\(", "prebuilt GEMM/attention library"),
    (r"\blcp_ref\b|\bLCP_SPARSE", "reference implementation"),
]


def static_check(root):
    for dp, _, files in os.walk(root):
        for f in files:
            if not f.endswith(SOURCE_EXT):
                continue
            p = os.path.join(dp, f)
            text = open(p, errors="replace").read()
            for pat, what in BLOCKED:
                if re.search(pat, text, re.M):
                    return f"{os.path.relpath(p, root)}: {what} ({pat})"
    return ""


def run(cfg, out, extra=()):
    cmd = [sys.executable, os.path.join(TASK_DIR, "run.py"), cfg, "--output", out, *extra]
    p = subprocess.run(cmd, cwd=TASK_DIR, capture_output=True, text=True)
    try:
        return json.load(open(out)), p.stdout[-3000:] + p.stderr[-3000:]
    except (OSError, json.JSONDecodeError):
        return {}, p.stdout[-3000:] + p.stderr[-3000:]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sol = os.path.abspath(sys.argv[1])
    res = {"compiled": False, "correct": False, "speedup": 0.0}
    if not os.path.isdir(os.path.join(sol, "hercules_plugin")):
        res["error"] = "the solution directory has no hercules_plugin/ package"
        print(json.dumps(res))
        return
    err = static_check(sol)
    if err:
        res["error"] = "blocked: " + err
        print(json.dumps(res))
        return
    work = tempfile.mkdtemp(prefix="lct_eval_")
    try:
        shutil.copytree(sol, os.path.join(work, "plugin"))
        cand, log = run("candidate", os.path.join(work, "cand.json"),
                        ["--plugin", os.path.join(work, "plugin"), "--log-dir", work])
        if not cand.get("started") or "ttft" not in cand:
            res["error"] = cand.get("error") or log
            print(json.dumps(res))
            return
        res["compiled"] = True
        res["quality"] = cand.get("quality", {})
        ref_json = os.environ.get("HERCULES_LCT_REFERENCE_JSON")
        if ref_json and os.path.isfile(ref_json):
            ref = json.load(open(ref_json))
        else:
            ref, log = run("reference", os.path.join(work, "ref.json"), ["--skip-quality", "--log-dir", work])
        if "ttft" not in ref:
            res["error"] = "reference run failed: " + (ref.get("error") or log)
            print(json.dumps(res))
            return
        per = {}
        for L, c in cand["ttft"].items():
            r = ref["ttft"][L]["min_s"]
            per[L] = {"ref_s": r, "solution_s": c["min_s"], "speedup": r / c["min_s"]}
        res["per_length"] = per
        res["ref_time_ms"] = 1e3 * sum(v["ref_s"] for v in per.values())
        res["kernel_time_ms"] = 1e3 * sum(v["solution_s"] for v in per.values())
        res["correct"] = bool(res["quality"].get("passed"))
        geo = math.exp(sum(math.log(v["speedup"]) for v in per.values()) / len(per))
        res["speedup"] = geo if res["correct"] else 0.0
        if not res["correct"]:
            res["error"] = "quality gate failed"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
