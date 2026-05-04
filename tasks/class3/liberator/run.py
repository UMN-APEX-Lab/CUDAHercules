#!/usr/bin/env python3
"""
Liberator Black-Box Benchmark -- CUDA-Hercules Class 3

Reference baseline:
  Internal Liberator executable built from the task's private source tree.

Custom solution:
  A black-box CUDA shared library implementing the fixed ABI declared in
  `custom_backend_api.h`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("LIBERATOR_DATA", os.path.join(TASK_DIR, "data"))
MEMORY_LIMITS_GB = [8, 10, 12, 15]
BASELINE_REPEATS = int(os.environ.get("KH_BENCHMARK", "3") or "3")
SOURCE_NODE = 101
PR_MAX_ITERATIONS = 1000
PR_DAMPING = 0.85
PR_TOLERANCE = 0.001
CUDA_MARGIN_BYTES = 512 * 1024 * 1024

LIBERATOR_SRC = os.path.join(TASK_DIR, "src")
BUILD_DIR = os.path.join(LIBERATOR_SRC, "cmake-build-release")
EXECUTABLE = os.path.join(BUILD_DIR, "Liberator")
CUSTOM_CUDA_BACKEND = os.path.join(TASK_DIR, "custom_cuda_backend")
REF_CHECKSUMS_FILE = os.path.join(TASK_DIR, "data", "ref_checksums.json")
CUDA_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "")

BCSR_FILE = os.path.join(DATA_DIR, "friendster.bcsr")
BCSC_FILE = os.path.join(DATA_DIR, "friendster.bcsc")
BWCSR_FILE = os.path.join(DATA_DIR, "friendster.bwcsr")

REF_CHECKSUMS = {
    "description": "Reference checksums for Liberator correctness verification (Friendster graph, source=101)",
    "bfs": {
        "reachable": 64768516,
        "value_sum": 411421572,
    },
    "cc": {
        "num_components": 59875151,
        "largest_component": 64768516,
    },
    "sssp": {
        "reachable": 64546552,
        "value_sum": 501358757,
        "value_sum_tolerance": 0.001,
    },
    "pr": {
        "pr_sum": 22765909.0623643212,
        "pr_sum_tolerance": 1e-6,
        "nonzero": 124836180,
    },
}


class KhWeightedEdge(ctypes.Structure):
    _fields_ = [("to_node", ctypes.c_uint32), ("weight", ctypes.c_uint32)]


WEIGHTED_EDGE_DTYPE = np.dtype([("to_node", np.uint32), ("weight", np.uint32)])


@dataclass
class AlgoResult:
    total_ms: float
    checksum_ok: bool
    checksum_detail: str
    error: str = ""


class CudaRuntime:
    def __init__(self) -> None:
        names = []
        found = ctypes.util.find_library("cudart")
        if found:
            names.append(found)
        names.extend(
            [
                "libcudart.so",
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
            ]
        )

        lib = None
        for name in names:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("Could not load libcudart")

        self.lib = lib
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.cudaMemGetInfo.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int

        self._check(self.lib.cudaSetDevice(0), "cudaSetDevice")
        dummy = ctypes.c_void_p()
        self.lib.cudaFree(dummy)

    @staticmethod
    def _check(status: int, name: str) -> None:
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    def mem_get_info(self) -> tuple[int, int]:
        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        self._check(self.lib.cudaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)), "cudaMemGetInfo")
        return int(free_bytes.value), int(total_bytes.value)

    def malloc(self, size_bytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes)), "cudaMalloc")
        return ptr

    def free(self, ptr: ctypes.c_void_p | None) -> None:
        if ptr:
            self._check(self.lib.cudaFree(ptr), "cudaFree")

    def synchronize(self) -> None:
        self._check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


class DeviceMemoryGuard:
    def __init__(self, cuda: CudaRuntime, budget_bytes: int, margin_bytes: int = CUDA_MARGIN_BYTES) -> None:
        self.cuda = cuda
        self.budget_bytes = budget_bytes
        self.margin_bytes = margin_bytes
        self.ptr: ctypes.c_void_p | None = None
        self.allocated_bytes = 0

    def __enter__(self) -> "DeviceMemoryGuard":
        free_bytes, _ = self.cuda.mem_get_info()
        if free_bytes <= self.budget_bytes:
            raise RuntimeError(
                f"Only {free_bytes / 1024**3:.2f} GB free on device, below requested budget "
                f"{self.budget_bytes / 1024**3:.2f} GB"
            )

        target_guard = max(0, free_bytes - self.budget_bytes - self.margin_bytes)
        step = 256 * 1024 * 1024
        while target_guard > 0:
            try:
                self.ptr = self.cuda.malloc(target_guard)
                self.allocated_bytes = target_guard
                break
            except RuntimeError:
                target_guard -= step
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.ptr is not None:
            self.cuda.free(self.ptr)
            self.ptr = None
            self.allocated_bytes = 0


def find_custom_backend_so(path: str) -> str:
    if not os.path.isdir(path):
        return ""
    for root, _, files in os.walk(path):
        for name in sorted(files):
            if name == "libkh_liberator_backend.so":
                return os.path.join(root, name)
    return ""


def build_liberator() -> None:
    print("Building Liberator baseline...", flush=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    subprocess.check_call(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=BUILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["make", f"-j{os.cpu_count()}"],
        cwd=BUILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print("Baseline build complete.", flush=True)


def load_ref_checksums() -> dict[str, Any]:
    if not os.path.isfile(REF_CHECKSUMS_FILE):
        return REF_CHECKSUMS
    with open(REF_CHECKSUMS_FILE) as f:
        return json.load(f)


def run_baseline_algorithm(algo: str, data_file: str, memory_gb: int, source: int | None = None) -> dict[str, Any]:
    cmd = [
        EXECUTABLE,
        "--input",
        data_file,
        "--type",
        algo,
        "--model",
        "7",
        "--testTime",
        str(BASELINE_REPEATS),
        "--timing",
        "1",
        "--memory",
        str(memory_gb),
    ]
    if source is not None:
        cmd += ["--source", str(source)]

    env = os.environ.copy()
    if CUDA_DEVICE:
        env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    output = result.stdout + result.stderr

    total_ms = -1.0
    for pattern in (
        r"average total process time:\s*(\d+)\s*ms",
        r"Average Total time:\s*(\d+)",
        r"totalProcess time is\s*(\d+)\s*ms",
    ):
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            total_ms = float(matches[-1])
            break

    checksum = {}
    m = re.search(r"RESULT_CHECK\s+(\w+):\s+(.*)", output)
    if m:
        for kv in m.group(2).split():
            key, value = kv.split("=", 1)
            try:
                checksum[key] = int(value)
            except ValueError:
                checksum[key] = float(value)

    return {
        "returncode": result.returncode,
        "total_ms": total_ms,
        "checksum": checksum,
        "output": output,
    }


def verify_baseline_checksum(algo: str, checksum: dict[str, Any], ref: dict[str, Any]) -> tuple[bool, str]:
    if not ref or algo not in ref:
        return True, "no reference"

    expected = ref[algo]
    errors = []
    if algo == "bfs":
        if checksum.get("reachable") != expected["reachable"]:
            errors.append(f"reachable={checksum.get('reachable')} != {expected['reachable']}")
        if checksum.get("value_sum") != expected["value_sum"]:
            errors.append(f"value_sum={checksum.get('value_sum')} != {expected['value_sum']}")
    elif algo == "cc":
        if checksum.get("num_components") != expected["num_components"]:
            errors.append(f"num_components={checksum.get('num_components')} != {expected['num_components']}")
        if checksum.get("largest_component") != expected["largest_component"]:
            errors.append(f"largest_component={checksum.get('largest_component')} != {expected['largest_component']}")
    elif algo == "sssp":
        if checksum.get("reachable") != expected["reachable"]:
            errors.append(f"reachable={checksum.get('reachable')} != {expected['reachable']}")
        ref_sum = float(expected["value_sum"])
        got_sum = float(checksum.get("value_sum", 0))
        tol = float(expected.get("value_sum_tolerance", 0.001))
        if ref_sum > 0 and abs(got_sum - ref_sum) / ref_sum > tol:
            errors.append(f"value_sum={got_sum} vs {ref_sum} (> {tol:.4f} rel err)")
    elif algo == "pr":
        ref_sum = float(expected["pr_sum"])
        got_sum = float(checksum.get("pr_sum", 0.0))
        tol = float(expected.get("pr_sum_tolerance", 1e-6))
        if ref_sum > 0 and abs(got_sum - ref_sum) / ref_sum > tol:
            errors.append(f"pr_sum={got_sum} vs {ref_sum} (> {tol:.2e} rel err)")
        if checksum.get("nonzero") != expected["nonzero"]:
            errors.append(f"nonzero={checksum.get('nonzero')} != {expected['nonzero']}")

    return (not errors), ("OK" if not errors else "; ".join(errors))


def _read_header(path: str) -> tuple[int, int]:
    header = np.memmap(path, dtype=np.uint64, mode="r", shape=(2,))
    return int(header[0]), int(header[1])


def load_bcsr_graph() -> tuple[int, int, np.memmap, np.memmap]:
    n, m = _read_header(BCSR_FILE)
    row_offsets = np.memmap(BCSR_FILE, dtype=np.uint64, mode="r", offset=16, shape=(n,))
    col_indices = np.memmap(BCSR_FILE, dtype=np.uint32, mode="r", offset=16 + n * 8, shape=(m,))
    return n, m, row_offsets, col_indices


def load_bwcsr_graph() -> tuple[int, int, np.memmap, np.memmap]:
    n, m = _read_header(BWCSR_FILE)
    row_offsets = np.memmap(BWCSR_FILE, dtype=np.uint64, mode="r", offset=16, shape=(n,))
    edges = np.memmap(BWCSR_FILE, dtype=WEIGHTED_EDGE_DTYPE, mode="r", offset=16 + n * 8, shape=(m,))
    return n, m, row_offsets, edges


def load_bcsc_graph() -> tuple[int, int, np.memmap, np.memmap, np.memmap]:
    n, m = _read_header(BCSC_FILE)
    out_degree = np.memmap(BCSC_FILE, dtype=np.uint32, mode="r", offset=16, shape=(n,))
    col_offsets = np.memmap(BCSC_FILE, dtype=np.uint64, mode="r", offset=16 + n * 4, shape=(n,))
    row_indices = np.memmap(BCSC_FILE, dtype=np.uint32, mode="r", offset=16 + n * 4 + n * 8, shape=(m,))
    return n, m, out_degree, col_offsets, row_indices


def bfs_checksum(levels: np.ndarray) -> dict[str, int]:
    reachable_mask = levels != np.uint32(0xFFFFFFFF)
    reachable = int(np.count_nonzero(reachable_mask))
    value_sum = int(levels[reachable_mask].sum(dtype=np.uint64))
    return {"reachable": reachable, "value_sum": value_sum}


def sssp_checksum(distances: np.ndarray) -> dict[str, int]:
    reachable_mask = distances != np.uint32(0xFFFFFFFF)
    reachable = int(np.count_nonzero(reachable_mask))
    value_sum = int(distances[reachable_mask].sum(dtype=np.uint64))
    return {"reachable": reachable, "value_sum": value_sum}


def cc_checksum(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.uint32)
    labels.sort()
    if labels.size == 0:
        return {"num_components": 0, "largest_component": 0}
    boundaries = np.nonzero(labels[1:] != labels[:-1])[0] + 1
    counts = np.diff(np.concatenate(([0], boundaries, [labels.size])))
    return {
        "num_components": int(counts.size),
        "largest_component": int(counts.max(initial=0)),
    }


def pr_checksum(ranks: np.ndarray) -> dict[str, float | int]:
    pr_sum = float(ranks.sum(dtype=np.float64))
    nonzero = int(np.count_nonzero(ranks > 1e-15))
    return {"pr_sum": pr_sum, "nonzero": nonzero}


def load_custom_backend(path: str):
    lib = ctypes.CDLL(path)
    lib.kh_liberator_backend_init.argtypes = [ctypes.c_uint64]
    lib.kh_liberator_backend_init.restype = ctypes.c_int
    lib.kh_liberator_backend_shutdown.argtypes = []
    lib.kh_liberator_backend_shutdown.restype = None
    lib.kh_liberator_backend_last_error.argtypes = []
    lib.kh_liberator_backend_last_error.restype = ctypes.c_char_p

    lib.kh_liberator_bfs.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.kh_liberator_bfs.restype = ctypes.c_int

    lib.kh_liberator_cc.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.kh_liberator_cc.restype = ctypes.c_int

    lib.kh_liberator_sssp.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(KhWeightedEdge),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.kh_liberator_sssp.restype = ctypes.c_int

    lib.kh_liberator_pr.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.kh_liberator_pr.restype = ctypes.c_int
    return lib


def last_custom_error(lib) -> str:
    msg = lib.kh_liberator_backend_last_error()
    return msg.decode("utf-8", errors="replace") if msg else ""


def _run_custom_call(fn, cuda: CudaRuntime, *args) -> tuple[int, float]:
    t0 = time.perf_counter()
    rc = fn(*args)
    cuda.synchronize()
    t1 = time.perf_counter()
    return rc, (t1 - t0) * 1000.0


def verify_custom_checksum(algo: str, checksum: dict[str, Any], ref: dict[str, Any]) -> tuple[bool, str]:
    return verify_baseline_checksum(algo, checksum, ref)


def run_custom_at_memory_limit(custom_so: str, memory_gb: int, ref_checksums: dict[str, Any]) -> tuple[bool, dict[str, AlgoResult], float]:
    cuda = CudaRuntime()
    budget_bytes = memory_gb * 1024**3
    lib = load_custom_backend(custom_so)

    per_algo: dict[str, AlgoResult] = {}
    total_ms = 0.0

    with DeviceMemoryGuard(cuda, budget_bytes):
        if lib.kh_liberator_backend_init(ctypes.c_uint64(budget_bytes)) != 0:
            detail = last_custom_error(lib) or "backend init failed"
            return False, {"init": AlgoResult(0.0, False, "", detail)}, 0.0

        try:
            n, m, row_offsets, col_indices = load_bcsr_graph()

            bfs_out = np.empty(n, dtype=np.uint32)
            rc, elapsed = _run_custom_call(
                lib.kh_liberator_bfs,
                cuda,
                ctypes.c_uint64(n),
                ctypes.c_uint64(m),
                row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                col_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_uint32(SOURCE_NODE),
                bfs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
            if rc != 0:
                return False, {"bfs": AlgoResult(elapsed, False, "", last_custom_error(lib) or "bfs failed")}, total_ms
            checksum = bfs_checksum(bfs_out)
            ok, detail = verify_custom_checksum("bfs", checksum, ref_checksums)
            per_algo["bfs"] = AlgoResult(elapsed, ok, detail)
            total_ms += elapsed
            del bfs_out
            gc.collect()
            if not ok:
                return False, per_algo, total_ms

            cc_out = np.empty(n, dtype=np.uint32)
            rc, elapsed = _run_custom_call(
                lib.kh_liberator_cc,
                cuda,
                ctypes.c_uint64(n),
                ctypes.c_uint64(m),
                row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                col_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                cc_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
            if rc != 0:
                return False, {"cc": AlgoResult(elapsed, False, "", last_custom_error(lib) or "cc failed")}, total_ms
            checksum = cc_checksum(cc_out)
            ok, detail = verify_custom_checksum("cc", checksum, ref_checksums)
            per_algo["cc"] = AlgoResult(elapsed, ok, detail)
            total_ms += elapsed
            del cc_out
            gc.collect()
            if not ok:
                return False, per_algo, total_ms

            n_w, m_w, w_row_offsets, weighted_edges = load_bwcsr_graph()
            sssp_out = np.empty(n_w, dtype=np.uint32)
            rc, elapsed = _run_custom_call(
                lib.kh_liberator_sssp,
                cuda,
                ctypes.c_uint64(n_w),
                ctypes.c_uint64(m_w),
                w_row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                weighted_edges.ctypes.data_as(ctypes.POINTER(KhWeightedEdge)),
                ctypes.c_uint32(SOURCE_NODE),
                sssp_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
            if rc != 0:
                return False, {"sssp": AlgoResult(elapsed, False, "", last_custom_error(lib) or "sssp failed")}, total_ms
            checksum = sssp_checksum(sssp_out)
            ok, detail = verify_custom_checksum("sssp", checksum, ref_checksums)
            per_algo["sssp"] = AlgoResult(elapsed, ok, detail)
            total_ms += elapsed
            del sssp_out
            gc.collect()
            if not ok:
                return False, per_algo, total_ms

            n_c, m_c, out_degree, col_offsets, row_indices = load_bcsc_graph()
            pr_out = np.empty(n_c, dtype=np.float64)
            rc, elapsed = _run_custom_call(
                lib.kh_liberator_pr,
                cuda,
                ctypes.c_uint64(n_c),
                ctypes.c_uint64(m_c),
                out_degree.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                col_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                row_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_uint32(PR_MAX_ITERATIONS),
                ctypes.c_double(PR_DAMPING),
                ctypes.c_double(PR_TOLERANCE),
                pr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            if rc != 0:
                return False, {"pr": AlgoResult(elapsed, False, "", last_custom_error(lib) or "pr failed")}, total_ms
            checksum = pr_checksum(pr_out)
            ok, detail = verify_custom_checksum("pr", checksum, ref_checksums)
            per_algo["pr"] = AlgoResult(elapsed, ok, detail)
            total_ms += elapsed
            del pr_out
            gc.collect()
            if not ok:
                return False, per_algo, total_ms
        finally:
            lib.kh_liberator_backend_shutdown()

    return True, per_algo, total_ms


def main() -> None:
    missing = []
    for path, desc in (
        (BCSR_FILE, "friendster.bcsr"),
        (BCSC_FILE, "friendster.bcsc"),
        (BWCSR_FILE, "friendster.bwcsr"),
    ):
        if not os.path.isfile(path):
            missing.append(f"{desc}: {path}")
    if missing:
        print("ERROR: Missing data files:")
        for item in missing:
            print(f"  {item}")
        print(f"\nRun: bash prepare_data.sh {DATA_DIR}")
        sys.exit(1)

    build_liberator()
    ref_checksums = load_ref_checksums()

    print("\n=== Liberator Black-Box Benchmark ===\n", flush=True)
    print(f"Dataset:       {DATA_DIR}")
    print(f"Memory limits: {MEMORY_LIMITS_GB} GB")
    print(f"Baseline reps: {BASELINE_REPEATS}")
    print(f"GPU visible:   {CUDA_DEVICE or 'default'}")

    baseline_total = 0.0
    baseline_passed = True
    for memory_gb in MEMORY_LIMITS_GB:
        print(f"\n--- Baseline @ {memory_gb} GB ---", flush=True)
        for algo, data_file, source in (
            ("bfs", BCSR_FILE, SOURCE_NODE),
            ("cc", BCSR_FILE, None),
            ("sssp", BWCSR_FILE, SOURCE_NODE),
            ("pr", BCSC_FILE, None),
        ):
            result = run_baseline_algorithm(algo, data_file, memory_gb, source=source)
            ok, detail = verify_baseline_checksum(algo, result["checksum"], ref_checksums)
            status = "PASS" if ok and result["returncode"] == 0 and result["total_ms"] > 0 else "FAIL"
            print(f"  {algo.upper():>5}: {result['total_ms']:.2f} ms [{status}] {detail}")
            if status != "PASS":
                baseline_passed = False
            baseline_total += max(0.0, result["total_ms"])

    solution = None
    custom_so = find_custom_backend_so(CUSTOM_CUDA_BACKEND)
    if custom_so:
        print(f"\n--- Custom CUDA Backend ---\n  SO: {custom_so}", flush=True)
        custom_passed = True
        custom_total = 0.0
        for memory_gb in MEMORY_LIMITS_GB:
            print(f"\n  Budget {memory_gb} GB", flush=True)
            ok, per_algo, total_ms = run_custom_at_memory_limit(custom_so, memory_gb, ref_checksums)
            for algo in ("bfs", "cc", "sssp", "pr"):
                if algo in per_algo:
                    r = per_algo[algo]
                    status = "PASS" if r.checksum_ok and not r.error else "FAIL"
                    extra = r.error or r.checksum_detail
                    print(f"    {algo.upper():>5}: {r.total_ms:.2f} ms [{status}] {extra}")
            custom_total += total_ms
            if not ok:
                custom_passed = False
                break
        if custom_passed:
            solution = {"passed": True, "kernel_time": custom_total}
        else:
            solution = None
            print("  Custom CUDA backend FAILED correctness check")
    else:
        print("\n--- Custom CUDA backend not available (no custom backend .so found) ---")

    primary_passed = bool(solution and solution["passed"]) or baseline_passed
    primary_time = solution["kernel_time"] if solution and solution["passed"] else baseline_total

    print("\n=== Summary ===")
    print("Passed" if primary_passed else "FAILED")
    print(f"Kernel time: {primary_time:.4f} ms")
    print(f"  Baseline total: {baseline_total:.2f} ms")
    if solution and solution["passed"]:
        print(f"  Solution total: {solution['kernel_time']:.2f} ms")
        if baseline_total > 0:
            print(f"  Speedup vs baseline: {baseline_total / solution['kernel_time']:.2f}x")

    if not primary_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
