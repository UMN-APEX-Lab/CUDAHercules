"""Exercise actual class2 compilation/evaluation with a no-op CUDA candidate."""
import fcntl
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cuda_hercules.eval import load_task_def
from cuda_hercules.runner import run_task
from cuda_hercules.task_schema import load_task_config


def main():
    task = ROOT / "tasks/class2/hopper/fa3_fwd_hdim128_fp16"
    config = load_task_config(str(task / "task.yaml"))
    signature = load_task_def(str(task))["FUNCTION_SIGNATURE"].strip().rstrip(";")
    source = ('#include <cuda_runtime.h>\n__global__ void noop() {}\n'
              + 'extern "C" ' + signature + ' {noop<<<1,1,0,stream>>>();}\n')
    with tempfile.TemporaryDirectory() as temp:
        candidate = Path(temp) / "noop.cu"
        candidate.write_text(source)
        with (ROOT / ".cache/hopper_gpu.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = run_task(config, str(candidate), measure_perf=False)
    assert result.compiled, result.error_msg
    assert not result.correct and result.status.value == "FAIL", result
    assert result.correctness_detail and not all(
        value["correct"] for value in result.correctness_detail.values())
    report = dict(task=config.task_id, passed=True, compiled=True,
                  noop_rejected=True, status=result.status.value)
    path = ROOT / "results/hopper_eval_poison.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Passed: compiled FA3 no-op rejected by the actual runner/eval pipeline")


if __name__ == "__main__":
    main()
