"""
CUDA kernel timing infrastructure.

Provides CUDA event-based timing with L2 cache clearing for cold-cache measurements.
Ported from KernelBench's timing.py with adaptations for CUDA-Hercules's function-based
(rather than class-based) kernel interface.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TimingStats:
    """Timing statistics from kernel profiling."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_trials: int
    all_times_ms: list[float]


def clear_l2_cache(device: torch.device = None):
    """
    Clear L2 cache by thrashing with a large tensor.
    256MB write to overwrite even the largest L2 caches (H200=90MB, Blackwell~192MB).
    """
    if device is None:
        device = torch.cuda.current_device()
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)


def time_cuda_function(
    fn: callable,
    args: list,
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> TimingStats:
    """
    Time a CUDA function using CUDA events with L2 cache clearing.

    Args:
        fn: Function to time. Called as fn(*args).
        args: Arguments to pass to fn.
        num_warmup: Number of warmup iterations.
        num_trials: Number of timed trials.
        discard_first: First N trials to discard (cold start).
        device: CUDA device.
        verbose: Print per-trial timing.

    Returns:
        TimingStats with mean, std, min, max, and all times.
    """
    if device is None:
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        # Warmup
        for _ in range(num_warmup):
            fn(*args)
            torch.cuda.synchronize(device=device)

        torch.cuda.empty_cache()

        elapsed_times = []

        for trial in range(num_trials + discard_first):
            torch.cuda.synchronize(device=device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            clear_l2_cache(device=device)

            start_event.record()
            fn(*args)
            end_event.record()

            torch.cuda.synchronize(device=device)

            elapsed_ms = start_event.elapsed_time(end_event)

            if trial >= discard_first:
                elapsed_times.append(elapsed_ms)
                if verbose:
                    print(f"  Trial {trial - discard_first + 1}: {elapsed_ms:.4f} ms")

    if not elapsed_times:
        return TimingStats(0.0, 0.0, 0.0, 0.0, 0, [])

    t = torch.tensor(elapsed_times)
    return TimingStats(
        mean_ms=t.mean().item(),
        std_ms=t.std().item() if len(t) > 1 else 0.0,
        min_ms=t.min().item(),
        max_ms=t.max().item(),
        num_trials=len(elapsed_times),
        all_times_ms=elapsed_times,
    )
