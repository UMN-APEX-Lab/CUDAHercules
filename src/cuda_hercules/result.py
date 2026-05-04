"""Unified result structure for task evaluation."""

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP_ARCH = "SKIP_ARCH"
    ERROR = "ERROR"


@dataclass
class PerfMetrics:
    """Performance metrics from a single task run."""
    speedup: float = 0.0
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_p90_ms: float = 0.0
    ref_latency_mean_ms: float = 0.0
    ref_latency_min_ms: float = 0.0
    # Per-size breakdown (Class 1): list of {name, kernel_min_ms, ref_min_ms, speedup}
    per_size: list = field(default_factory=list)


@dataclass
class TaskResult:
    """Unified result from evaluating a single task."""
    task_id: str = ""
    compiled: bool = False
    correct: bool = False
    correctness_detail: dict = field(default_factory=dict)
    speedup: float = 0.0
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_p90_ms: float = 0.0
    ref_latency_mean_ms: float = 0.0
    per_size: list = field(default_factory=list)
    domain: str = ""
    level: int = 0
    min_sm: int = 0
    status: TaskStatus = TaskStatus.ERROR
    error_msg: str = ""
