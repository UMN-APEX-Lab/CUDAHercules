"""Class 2 timing hardening: work escaping the timed stream, cross-call caching."""
import pytest
import torch

from cuda_hercules.eval import (TimingVerificationError, _make_timing_input_sets,
                                _poison_outputs, check_correctness)
from cuda_hercules.timing import calibrate_sync_overhead, time_cuda_function

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _work(x, y, reps=8):
    for _ in range(reps):
        torch.mul(x, 1.0001, out=y)


@pytest.fixture(scope="module")
def bufs():
    x = torch.randn(32 << 20, device="cuda")
    return x, torch.empty_like(x)


@pytest.fixture(scope="module")
def calibration():
    return calibrate_sync_overhead()


def test_unjoined_private_stream_is_charged_full_cost(bufs, calibration):
    x, y = bufs
    honest = time_cuda_function(_work, [x, y], calibration=calibration)
    side = torch.cuda.Stream()  # non-blocking w.r.t. the timed stream

    def escape(x, y):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _work(x, y)

    escaped = time_cuda_function(escape, [x, y], calibration=calibration)
    assert escaped.escape_flagged_trials == escaped.num_trials
    assert min(escaped.event_times_ms) < 0.2 * honest.min_ms  # the old measurement was fooled
    assert escaped.min_ms > 0.9 * honest.min_ms


def test_joined_private_stream_is_not_penalized(bufs, calibration):
    x, y = bufs
    honest = time_cuda_function(_work, [x, y], calibration=calibration)
    side = torch.cuda.Stream()

    def fork_join(x, y):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _work(x, y)
        torch.cuda.current_stream().wait_stream(side)

    joined = time_cuda_function(fork_join, [x, y], calibration=calibration)
    assert joined.escape_flagged_trials == 0
    assert joined.mean_ms < 1.05 * honest.mean_ms + 0.01


def test_alternating_sets_catch_a_pointer_keyed_cache():
    n = 1 << 20
    task = {
        "CHECKED_OUTPUTS": None,
        "FUNCTION_SIGNATURE": "void f(const float* x, float* y, int n, cudaStream_t stream);",
        "SCALAR_ARGS": {"n": n},
        "get_inputs": lambda: [("x", torch.randn(n, device="cuda"))],
        "get_outputs": lambda inputs: [("y", torch.full_like(inputs[0][1], float("nan")))],
    }
    ref = lambda x, y, n: torch.mul(x, 2.0, out=y)  # noqa: E731
    live, sets = _make_timing_input_sets(task, ref, 2, first_seed=7, device=None)
    assert not torch.equal(sets[0][0][0][1], sets[1][0][0][1])
    outputs = task["get_outputs"](live)
    stash = {}

    def cached(x, y, n):  # recomputes only when it sees new pointers
        key = (x.data_ptr(), y.data_ptr())
        if key not in stash:
            ref(x, y, n)
            stash[key] = y.clone()
        else:
            y.copy_(stash[key])

    def load(i):
        for (_, dst), (_, src) in zip(live, sets[i % 2][0]):
            dst.copy_(src)
        _poison_outputs(outputs, None)

    def verify(i):
        ok, per_output = check_correctness(sets[i % 2][1], outputs, {"atol": 0, "rtol": 0})
        if not ok:
            raise TimingVerificationError("mismatch", per_output)

    args = [live[0][1], outputs[0][1], n]
    time_cuda_function(ref, args, num_warmup=2, num_trials=2, before_call=load, after_call=verify)
    with pytest.raises(TimingVerificationError):
        time_cuda_function(cached, args, num_warmup=2, num_trials=2,
                           before_call=load, after_call=verify)
