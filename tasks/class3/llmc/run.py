#!/usr/bin/env python3
"""
llm.c GPT-2 Training Benchmark -- CUDA-Hercules Class 3

Builds llm.c from source and trains GPT-2 (124M) on TinyShakespeare,
measuring training throughput and loss convergence.

LLM optimizes: llmc/attention.cuh (attention forward/backward kernels)
Reference: llm.c (Karpathy) — pure CUDA GPT-2 training.
"""

import os
import sys
import re
import subprocess

# ── Configuration ──────────────────────────────────────────────────────

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TASK_DIR, 'src')
DATA_DIR = os.path.join(TASK_DIR, 'data')
EXECUTABLE = os.path.join(SRC_DIR, 'train_gpt2cu')

CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', '')

# Training parameters
BATCH_SIZE = 16
SEQ_LEN = 256
NUM_STEPS = 20
LEARNING_RATE = 3e-4
MODEL_DESC = 'gpt2:d36'  # GPT-2 large (774M), random init
VAL_EVERY = 10
OVERFIT_SINGLE_BATCH = 1  # overfit one batch for correctness

# ── Build ──────────────────────────────────────────────────────────────

def build():
    """Build llm.c training binary."""
    if os.path.isfile(EXECUTABLE):
        print(f"Binary exists: {EXECUTABLE}")
        return

    print("Building llm.c...", flush=True)
    env = os.environ.copy()
    env['NO_MULTI_GPU'] = '1'
    if CUDA_DEVICE:
        env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
    subprocess.check_call(
        ['make', '-j', 'train_gpt2cu'],
        cwd=SRC_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print("Build complete.", flush=True)

# ── Data ───────────────────────────────────────────────────────────────

def prepare_data():
    """Prepare TinyShakespeare data if not present."""
    train_file = os.path.join(DATA_DIR, 'tiny_shakespeare_train.bin')
    val_file = os.path.join(DATA_DIR, 'tiny_shakespeare_val.bin')
    if os.path.isfile(train_file) and os.path.isfile(val_file):
        return
    print("Preparing data...", flush=True)
    subprocess.check_call(
        ['bash', os.path.join(TASK_DIR, 'prepare_data.sh'), DATA_DIR])

# ── Run ────────────────────────────────────────────────────────────────

def run():
    """Run training and parse results."""
    train_file = os.path.join(DATA_DIR, 'tiny_shakespeare_train.bin')
    val_file = os.path.join(DATA_DIR, 'tiny_shakespeare_val.bin')

    cmd = [
        EXECUTABLE,
        '-i', train_file,
        '-j', val_file,
        '-e', MODEL_DESC,
        '-b', str(BATCH_SIZE),
        '-t', str(SEQ_LEN),
        '-x', str(NUM_STEPS),
        '-l', str(LEARNING_RATE),
        '-v', str(VAL_EVERY),
        '-s', '0',               # no sampling (no tokenizer needed)
        '-a', str(OVERFIT_SINGLE_BATCH),
    ]

    env = os.environ.copy()
    if CUDA_DEVICE:
        env['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    output = result.stdout + result.stderr
    print(output)

    # Parse step losses
    step_losses = re.findall(r'step\s+(\d+)/\d+\s+\|\s+loss\s+([0-9.]+)', output)
    all_losses = [float(s[1]) for s in step_losses]
    first_loss = all_losses[0] if all_losses else -1
    last_loss = all_losses[-1] if all_losses else -1

    # Loss at convergence checkpoints (25%, 50%, 75%, 100% of steps)
    loss_checkpoints = {}
    if all_losses:
        n = len(all_losses)
        for p in [25, 50, 75, 100]:
            idx = max(0, n * p // 100 - 1)
            loss_checkpoints[p] = all_losses[idx]

    # Parse val losses
    val_losses = re.findall(r'val loss\s+([0-9.]+)', output)
    first_val = float(val_losses[0]) if val_losses else -1
    last_val = float(val_losses[-1]) if val_losses else -1

    # Parse total average iteration time
    m = re.search(r'total average iteration time:\s+([0-9.]+)\s+ms', output)
    avg_iter_ms = float(m.group(1)) if m else -1

    # Parse tok/s from last step
    tok_matches = re.findall(r'([0-9.]+)\s+tok/s', output)
    final_toks = float(tok_matches[-1]) if tok_matches else -1

    # Parse model parameter count (llm.c prints 'num_parameters: N => bytes: M')
    m_params = re.search(r'num_parameters:\s*(\d+)', output)
    model_params = int(m_params.group(1)) if m_params else -1

    return {
        'returncode': result.returncode,
        'first_loss': first_loss,
        'last_loss': last_loss,
        'first_val': first_val,
        'last_val': last_val,
        'loss_checkpoints': loss_checkpoints,
        'all_losses': all_losses,
        'avg_iter_ms': avg_iter_ms,
        'tok_per_sec': final_toks,
        'model_params': model_params,
        'loss_decreased': last_loss < first_loss,
    }

# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-time-ms', type=float, default=None,
                        help='Reference kernel time in ms (enables Ref time / Speedup output)')
    args = parser.parse_args()

    prepare_data()
    build()

    print(f"\n=== llm.c Training Benchmark: GPT-2 124M, TinyShakespeare ===\n", flush=True)

    r = run()

    # Correctness: loss must decrease (gradient descent works correctly)
    passed = r['returncode'] == 0 and r['loss_decreased'] and r['last_loss'] < 8.0

    print(f"\n=== Summary ===")
    if passed:
        print("Passed")
    else:
        print("FAILED")

    # Kernel time = total training time (avg iteration × num steps)
    kernel_time = r['avg_iter_ms'] * (NUM_STEPS - 1)  # exclude warmup step 0

    if args.ref_time_ms is not None:
        print(f"Ref time: {args.ref_time_ms:.4f} ms")
    print(f"Kernel time: {kernel_time:.4f} ms")
    if args.ref_time_ms is not None and kernel_time > 0:
        speedup = args.ref_time_ms / kernel_time
        print(f"Speedup: {speedup:.4f}x")

    print(f"  avg iter: {r['avg_iter_ms']:.2f} ms")
    print(f"  tok/s: {r['tok_per_sec']:.0f}")
    print(f"Solution loss: {r['first_loss']:.4f} -> {r['last_loss']:.4f}")
    print(f"  val loss: {r['first_val']:.4f} -> {r['last_val']:.4f}")

    # Loss convergence checkpoints
    if r['loss_checkpoints']:
        print(f"Solution loss checkpoints: " + " | ".join(
            f"{p}%={r['loss_checkpoints'][p]:.4f}" for p in [25, 50, 75, 100]
            if p in r['loss_checkpoints']))

    # Per-step loss trajectory — used by eval_solution.py for anti-cheat
    # (checks solution's per-step loss has natural training noise and roughly
    # matches the reference's trajectory).
    if r['all_losses']:
        print("Loss per step: " + ",".join(f"{v:.4f}" for v in r['all_losses']))

    # Model parameter count — used by eval_solution.py to gate out shrunken models
    if r['model_params'] > 0:
        print(f"Model params: {r['model_params']}")

    if not passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
