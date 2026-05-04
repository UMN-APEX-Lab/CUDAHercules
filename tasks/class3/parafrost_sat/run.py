#!/usr/bin/env python3
"""ParaFROST GPU SAT Solver Benchmark -- CUDA-Hercules Class 3

Runs GPU-accelerated SAT solving on a curated 10-instance subset of the
Zenodo j2021 SAT benchmark suite (record 5138008). The exact instances
are pinned in prepare_data.py.

Measures total solving time (simplifier + solver). No timeout limit.
"""
import os, sys, re, subprocess
from prepare_data import SUBSET_MANIFEST

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, 'src')
EXECUTABLE = os.path.join(SRC_DIR, 'parafrost')
DATA_DIR = os.path.join(TASK_DIR, 'data')
CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', '')


def _load_tests():
    """Load TESTS = [(filename, description)] from the embedded subset manifest."""
    m = SUBSET_MANIFEST
    out = []
    for inst in m['instances']:
        name = inst['name']
        tier = inst.get('tier', '?')
        verdict = inst.get('reference_verdict', '?')
        ref_t = inst.get('reference_time_sec', -1)
        desc = f"Zenodo j2021 [{tier}] verdict={verdict} ref={ref_t:.1f}s"
        out.append((name + '.cnf', desc))
    return out


# Test instances pinned by prepare_data.py.
TESTS = _load_tests()

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

def strip_ansi(text):
    return ANSI_ESCAPE.sub('', text)

def prepare_data():
    needed = [t[0] for t in TESTS]
    if all(os.path.isfile(os.path.join(DATA_DIR, f)) for f in needed):
        return
    print("Preparing benchmark data...", flush=True)
    subprocess.check_call(['bash', os.path.join(TASK_DIR, 'prepare_data.sh'), DATA_DIR])

def build():
    if os.path.isfile(EXECUTABLE): return
    print("Building ParaFROST...", flush=True)
    env = os.environ.copy()
    if CUDA_DEVICE: env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
    subprocess.check_call(['make', f'-j{os.cpu_count()}'], cwd=SRC_DIR, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def run_instance(cnf_path):
    env = os.environ.copy()
    if CUDA_DEVICE: env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
    r = subprocess.run([EXECUTABLE, cnf_path], capture_output=True, text=True, env=env)
    output = strip_ansi(r.stdout + r.stderr)

    sat = 'SATISFIABLE' in output and 'UNSATISFIABLE' not in output
    unsat = 'UNSATISFIABLE' in output
    simp = float(m.group(1)) if (m := re.search(r'Simplifier time\s*:\s*([0-9.]+)', output)) else 0
    solve = float(m.group(1)) if (m := re.search(r'Solver time\s*:\s*([0-9.]+)', output)) else 0
    n_vars = int(m.group(1)) if (m := re.search(r'(\d+)\s+Variables', output)) else 0
    n_cls = int(m.group(1)) if (m := re.search(r'(\d+)\s+Clauses', output)) else 0

    return {
        'sat': sat, 'unsat': unsat, 'solved': sat or unsat,
        'simp_sec': simp, 'solve_sec': solve, 'total_sec': simp + solve,
        'n_vars': n_vars, 'n_clauses': n_cls,
        'returncode': r.returncode
    }

def main():
    prepare_data()
    build()

    print("\n=== ParaFROST GPU SAT Solver Benchmark ===\n", flush=True)

    all_passed = True
    total_time = 0

    for cnf_file, desc in TESTS:
        cnf_path = os.path.join(DATA_DIR, cnf_file)
        print(f"--- {cnf_file} ({desc}) ---", flush=True)
        r = run_instance(cnf_path)
        if r['solved']:
            print(f"  Result: {'SAT' if r['sat'] else 'UNSAT'} "
                  f"({r['n_vars']} vars, {r['n_clauses']} clauses)")
            print(f"  Simplifier: {r['simp_sec']:.3f}s | "
                  f"Solver: {r['solve_sec']:.3f}s | "
                  f"Total: {r['total_sec']:.3f}s")
            total_time += r['total_sec']
        else:
            print(f"  FAILED (no verdict, exit={r['returncode']})")
            all_passed = False

    kernel_time_ms = total_time * 1000

    print(f"\n=== Summary ===")
    if all_passed:
        print("Passed")
    else:
        print("FAILED")
    print(f"Kernel time: {kernel_time_ms:.4f} ms")

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
