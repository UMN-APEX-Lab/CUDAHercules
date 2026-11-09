"""
CUDA kernel timing infrastructure.

Provides CUDA event-based timing with L2 cache clearing for cold-cache measurements.
Ported from KernelBench's timing.py with adaptations for CUDA-Hercules's function-based
(rather than class-based) kernel interface.

Every timed call is measured two ways:

* **Stream time** -- CUDA events recorded on the current stream around the call
  (the classic measurement).
* **Device-wide completion** -- the host waits for the stop event, records a
  "seen" event, then waits for the whole device (``torch.cuda.synchronize()``)
  and records a "done" event. ``start -> done`` therefore covers work the call
  left running on other streams (private non-blocking streams, graphs, host
  callbacks), plus a small sync/launch overhead that is calibrated on this
  machine with an empty op (and an idle single-stream GPU op) measured the same
  way.

The trial time is ``max(stream_time, device_time - calibrated_overhead)``, so
work that escapes the timed stream is charged at its full cost, while a
correctly joined (or single-stream) call is timed as before (host wake-up jitter
beyond the calibrated p90 is charged too: it cannot be told apart from escaped
work in that window; it is a few us and hits solution and reference alike).
Trials whose charged excess exceeds a calibrated margin are re-measured (up to
``max_retries`` extra calls, keeping the lowest). A trial is counted in
``TimingStats.escape_flagged_trials`` only when the device was still busy after
the host had already observed the stop event (``seen -> done`` beyond its
calibrated overhead plus max(margin, 0.5% of the stream time)).

To keep escaped work from overlapping the L2 flush (which would hide it before
the start event), the flush is completed (synchronized) before the call, and an
SM-filling "gate" kernel is launched just ahead of the start event: it keeps the
stream busy while the host enqueues the call -- so the start event does not pick
up the host launch gap, matching the old flush-overlapped measurement -- and its
blocks occupy every SM's thread slots, so kernels the call launches on other
streams cannot start before the start event.
"""

import ctypes
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class TimingStats:
    """Timing statistics from kernel profiling.

    ``all_times_ms`` (and mean/std/min/max) are the per-trial times charged to
    the function: the stream (CUDA event) time, raised to the device-wide
    completion time when work escaped the timed stream.
    """
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_trials: int
    all_times_ms: list[float]
    # Raw CUDA-event (current-stream) time of each kept trial.
    event_times_ms: list[float] = field(default_factory=list)
    # Kept trials in which the device was still busy after the host had seen
    # the stop event (work escaped the timed stream).
    escape_flagged_trials: int = 0
    # Largest charged device-wide excess over the stream time among kept trials.
    max_escape_ms: float = 0.0
    # Extra calls made to re-measure trials with an excess above the margin.
    retries: int = 0
    # Calls (warmup + timed + retries) whose outputs were verified.
    verified_calls: int = 0


def clear_l2_cache(device: torch.device = None):
    """
    Clear L2 cache by thrashing with a large tensor.
    256MB write to overwrite even the largest L2 caches (H200=90MB, Blackwell~192MB).
    """
    if device is None:
        device = torch.cuda.current_device()
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)


# ── SM-filling gate kernel ─────────────────────────────────────────────────
#
# A spin kernel on %globaltimer, loaded through the driver API from PTX (no
# compiler needed at eval time; the driver JITs it for the live GPU).

GATE_DURATION_US = 200  # ~ the L2 flush the old measurement overlapped the host launch with

_GATE_PTX = rb"""
.version 7.0
.target sm_70
.address_size 64
.visible .entry cuda_hercules_timing_gate(.param .u64 duration_ns)
{
  .reg .pred p;
  .reg .b64 t0, t1, d, dt;
  ld.param.u64 d, [duration_ns];
  mov.u64 t0, %globaltimer;
SPIN:
  mov.u64 t1, %globaltimer;
  sub.u64 dt, t1, t0;
  setp.lt.u64 p, dt, d;
  @p bra SPIN;
  ret;
}
"""


class _Gate:
    """Launches a kernel that fills every SM's thread slots for a fixed time."""

    def __init__(self, device_index: int):
        cu = ctypes.CDLL("libcuda.so.1")
        c_int_p = ctypes.POINTER(ctypes.c_int)
        cu.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                           ctypes.c_char_p]
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            c_int_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        cu.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self._cu = cu

        with torch.cuda.device(device_index):
            torch.cuda.synchronize()  # makes torch's (primary) context current on this thread
            ctx = ctypes.c_void_p()
            self._check(cu.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
            if not ctx.value:
                raise RuntimeError("no current CUDA context")
            self._module = ctypes.c_void_p()
            self._check(cu.cuModuleLoadData(ctypes.byref(self._module), _GATE_PTX),
                        "cuModuleLoadData")
            self._fn = ctypes.c_void_p()
            self._check(cu.cuModuleGetFunction(ctypes.byref(self._fn), self._module,
                                               b"cuda_hercules_timing_gate"),
                        "cuModuleGetFunction")

            props = torch.cuda.get_device_properties(device_index)
            max_threads = props.max_threads_per_multi_processor
            # Pick a block size whose resident blocks exactly fill the SM's
            # thread slots, so no block of any other kernel fits beside them.
            chosen = None
            for block in (256, 128, 192, 384, 512, 64, 96, 32):
                blocks = ctypes.c_int()
                self._check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    ctypes.byref(blocks), self._fn, block, 0), "cuOccupancy")
                if blocks.value * block == max_threads:
                    chosen = (block, blocks.value)
                    break
            if chosen is None:
                raise RuntimeError("no gate block size fills the SM thread slots")
            self.block = chosen[0]
            self.blocks_per_sm = chosen[1]
            self.grid = chosen[1] * props.multi_processor_count

    @staticmethod
    def _check(status: int, what: str):
        if status != 0:
            raise RuntimeError(f"{what} failed with CUresult {status}")

    def launch(self, duration_us: float):
        duration = ctypes.c_uint64(int(duration_us * 1000))
        params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.pointer(duration), ctypes.c_void_p))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._check(self._cu.cuLaunchKernel(self._fn, self.grid, 1, 1, self.block, 1, 1, 0,
                                            stream, params, None), "cuLaunchKernel(gate)")


_GATES: dict = {}


def _device_index(device) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    if isinstance(device, str):
        return _device_index(torch.device(device))
    return int(device)


def _get_gate(device_index: int) -> Optional[_Gate]:
    """The gate for a device, or None when it cannot be loaded (falls back to an idle start)."""
    if device_index not in _GATES:
        try:
            _GATES[device_index] = _Gate(device_index)
        except Exception as e:  # pragma: no cover - depends on the driver
            print(f"[timing] WARNING: SM-filling gate unavailable ({e}); "
                  f"timing calls from an idle GPU instead")
            _GATES[device_index] = None
    return _GATES[device_index]


# ── Measurement ────────────────────────────────────────────────────────────

@dataclass
class SyncCalibration:
    """Device-sync overhead measured with `_measure_call`.

    The "tail" of a call is stop event -> done event; the "post" part is seen
    event -> done event. For a call that leaves nothing running on other
    streams both are pure overhead: host wake-up + event launches (tail), and
    a sync on an already idle device + an event launch (post).
    """
    median_ms: float        # median tail of an empty op
    allowance_ms: float     # tail subtracted before charging device-wide excess (p90)
    p99_ms: float           # 99th percentile tail (diagnostic)
    margin_ms: float        # charged excess above which a trial is re-measured
    post_allowance_ms: float  # seen -> done overhead (p95); beyond it + margin = escape
    gate: bool              # SM-filling gate in use
    gate_us: float
    samples: int

    def as_dict(self) -> dict:
        return {
            "sync_overhead_median_ms": self.median_ms,
            "sync_overhead_allowance_ms": self.allowance_ms,
            "sync_overhead_p99_ms": self.p99_ms,
            "escape_margin_ms": self.margin_ms,
            "post_sync_allowance_ms": self.post_allowance_ms,
            "gate": self.gate,
            "gate_us": self.gate_us,
            "calibration_samples": self.samples,
        }


_CALIBRATIONS: dict = {}

MIN_ESCAPE_MARGIN_MS = 0.010
MAX_ESCAPE_MARGIN_MS = 0.050
ESCAPE_FLAG_REL = 0.005  # flag threshold grows with long calls (host wake-up jitter)


def _measure_call(fn: Callable, args, device_index: int, gate: Optional[_Gate],
                  gate_us: float) -> tuple[float, float, float]:
    """Time one call. Returns (stream_ms, tail_ms, post_ms).

    stream_ms: CUDA events around the call on the current stream.
    tail_ms:   stop event -> "done" event recorded right after a device-wide
               sync: how long the device stayed busy after the stop event, plus
               host wake-up / launch overhead (calibrated away by the caller).
    post_ms:   "seen" event (recorded as soon as the host observed the stop
               event) -> "done" event: device still busy after that point is
               unambiguous escaped work, free of host wake-up jitter.
    """
    torch.cuda.synchronize(device=device_index)
    clear_l2_cache(device=device_index)
    # Finish the flush first: escaped work must not overlap it unseen.
    torch.cuda.synchronize(device=device_index)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    seen_event = torch.cuda.Event(enable_timing=True)
    done_event = torch.cuda.Event(enable_timing=True)

    if gate is not None:
        gate.launch(gate_us)
    start_event.record()
    fn(*args)
    end_event.record()
    end_event.synchronize()                      # the timed stream is done
    seen_event.record()
    torch.cuda.synchronize(device=device_index)  # device-wide: every stream
    done_event.record()
    done_event.synchronize()
    return (start_event.elapsed_time(end_event), end_event.elapsed_time(done_event),
            seen_event.elapsed_time(done_event))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def calibrate_sync_overhead(
    device: Optional[torch.device] = None,
    num_samples: int = 60,
    gate_us: float = GATE_DURATION_US,
    use_cache: bool = False,
) -> SyncCalibration:
    """Measure the device-sync overhead exactly as calls are timed.

    The overhead is the tail (stop event -> device-sync-done event) of calls
    that leave nothing running on other streams. Charged excess is measured
    above `allowance_ms`; an excess above `margin_ms` makes a trial be
    re-measured; a seen -> done time above `post_allowance_ms + margin_ms` is
    flagged as escaped work.
    """
    device_index = _device_index(device)
    if use_cache and device_index in _CALIBRATIONS:
        return _CALIBRATIONS[device_index]

    # Two reference ops: an empty op, and a ~0.5 ms single-thread GPU spin on
    # the current stream (no memory traffic). The sync wake-up after a real
    # kernel runs ~1-3 us longer than after an empty op on this hardware, so
    # the allowance is the larger p90 of the two.
    ops = [lambda: None]
    if hasattr(torch.cuda, "_sleep"):
        ops.append(lambda: torch.cuda._sleep(1_000_000))
    per_op = []
    posts = []
    with torch.cuda.device(device_index):
        gate = _get_gate(device_index)
        for op in ops:
            tails = []
            for i in range(num_samples + 5):
                _, tail, post = _measure_call(op, (), device_index, gate, gate_us)
                if i >= 5:
                    tails.append(tail)
                    posts.append(post)
            per_op.append(tails)

    allowance = max(_quantile(tails, 0.90) for tails in per_op)
    spread = max(_quantile(tails, 0.95) - statistics.median(tails) for tails in per_op)
    calibration = SyncCalibration(
        median_ms=statistics.median(per_op[0]),
        allowance_ms=allowance,
        p99_ms=max(_quantile(tails, 0.99) for tails in per_op),
        margin_ms=min(MAX_ESCAPE_MARGIN_MS, max(MIN_ESCAPE_MARGIN_MS, 2.0 * spread)),
        post_allowance_ms=_quantile(posts, 0.95),
        gate=gate is not None,
        gate_us=gate_us if gate is not None else 0.0,
        samples=num_samples * len(ops),
    )
    _CALIBRATIONS[device_index] = calibration
    return calibration


def time_cuda_function(
    fn: callable,
    args: list,
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    *,
    before_call: Optional[Callable[[int], None]] = None,
    after_call: Optional[Callable[[int], None]] = None,
    calibration: Optional[SyncCalibration] = None,
    max_retries: int = 2,
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
        before_call: Called as before_call(call_index) before every call
            (warmup, timed and retried), outside the timed region -- e.g. to load
            the next input set into the argument tensors and poison the outputs.
        after_call: Called as after_call(call_index) after every call has fully
            completed on the device, outside the timed region -- e.g. to verify
            the outputs. Exceptions propagate.
        calibration: Sync-overhead calibration (see calibrate_sync_overhead);
            measured (or reused from this process) when omitted.
        max_retries: Extra calls allowed per trial to re-measure a trial whose
            device-wide completion exceeds its stream time by more than the
            calibrated margin.

    Returns:
        TimingStats with mean, std, min, max, and all times.
    """
    device_index = _device_index(device)
    if calibration is None:
        calibration = calibrate_sync_overhead(device_index, use_cache=True)

    call_index = 0
    verified = 0
    retries = 0
    elapsed_times = []
    event_times = []
    flagged = 0
    max_escape = 0.0

    with torch.cuda.device(device_index):
        gate = _get_gate(device_index) if calibration.gate else None

        # Warmup
        for _ in range(num_warmup):
            if before_call is not None:
                before_call(call_index)
            fn(*args)
            torch.cuda.synchronize(device=device_index)
            if after_call is not None:
                after_call(call_index)
                verified += 1
            call_index += 1

        torch.cuda.empty_cache()

        for trial in range(num_trials + discard_first):
            best = None
            for attempt in range(1 + max(0, max_retries)):
                if attempt:
                    retries += 1
                if before_call is not None:
                    before_call(call_index)
                stream_ms, tail_ms, post_ms = _measure_call(fn, args, device_index, gate,
                                                            calibration.gate_us)
                if after_call is not None:
                    after_call(call_index)
                    verified += 1
                call_index += 1

                # Escaped work: the device was still busy after the host had
                # seen the stop event (beyond the overhead of that window; the
                # relative term absorbs host jitter on long calls).
                escaped = post_ms > calibration.post_allowance_ms + max(
                    calibration.margin_ms, ESCAPE_FLAG_REL * stream_ms)
                # When work escaped, the device-wide wait ended on that work, so
                # only the single-wait overhead (post allowance) is subtracted.
                allowance = calibration.post_allowance_ms if escaped else calibration.allowance_ms
                excess = max(0.0, tail_ms - allowance)
                charged = stream_ms + excess  # == max(stream, device-wide - overhead)
                if best is None or charged < best[0]:
                    best = (charged, stream_ms, excess, escaped)
                if excess <= calibration.margin_ms:
                    break

            charged, stream_ms, excess, escaped = best
            if trial >= discard_first:
                elapsed_times.append(charged)
                event_times.append(stream_ms)
                max_escape = max(max_escape, excess)
                if escaped:
                    flagged += 1
                if verbose:
                    note = (f"  [work escaped the timed stream: device-wide +{excess:.4f} ms]"
                            if escaped else "")
                    print(f"  Trial {trial - discard_first + 1}: {charged:.4f} ms{note}")

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
        event_times_ms=event_times,
        escape_flagged_trials=flagged,
        max_escape_ms=max_escape,
        retries=retries,
        verified_calls=verified,
    )
