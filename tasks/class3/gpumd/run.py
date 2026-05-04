#!/usr/bin/env python3
"""GPU Molecular Dynamics Benchmark (3 Materials) -- CUDA-Hercules Class 3"""
import os, sys, re, subprocess
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE = os.path.join(TASK_DIR, 'src', 'md_bench')
CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', '')

def generate_data():
    data_dir = os.path.join(TASK_DIR, 'data')
    needed = ['Ar_256000.xyz', 'Ar_500000.xyz', 'Si_27000.xyz', 'Si_64000.xyz',
              'NaCl_8000.xyz', 'NaCl_32768.xyz']
    if all(os.path.isfile(os.path.join(data_dir, f)) for f in needed):
        return
    print("Generating structure data...", flush=True)
    subprocess.check_call([sys.executable, os.path.join(TASK_DIR, 'generate_data.py')])

def build():
    if os.path.isfile(EXECUTABLE): return
    print("Building...", flush=True)
    env = os.environ.copy()
    if CUDA_DEVICE: env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
    subprocess.check_call(['make', 'md_bench'], cwd=os.path.join(TASK_DIR, 'src'), env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def main():
    generate_data()
    build()
    print("\n=== GPU MD Benchmark (3 Materials) ===\n", flush=True)
    env = os.environ.copy()
    if CUDA_DEVICE: env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
    data_dir = os.path.join(TASK_DIR, 'data')
    r = subprocess.run([EXECUTABLE, data_dir], capture_output=True, text=True, env=env, timeout=600)
    print(r.stdout + r.stderr)
    kt = float(m.group(1)) if (m := re.search(r'Kernel time:\s+([0-9.]+)', r.stdout)) else -1
    passed = 'Passed' in r.stdout and r.returncode == 0
    print(f"\n=== Summary ===")
    print("Passed" if passed else "FAILED")
    print(f"Kernel time: {kt:.4f} ms")
    if not passed: sys.exit(1)

if __name__ == '__main__': main()
