"""Abstract backend interface and resolver."""

from abc import ABC, abstractmethod

from .task_schema import TaskConfig
from .result import PerfMetrics


class Backend(ABC):
    """Base class for task execution backends."""

    @abstractmethod
    def prepare(self, config: TaskConfig, solution_path: str) -> None:
        """Prepare the task (copy solution, set up environment)."""
        ...

    @abstractmethod
    def build(self, config: TaskConfig, solution_path: str) -> tuple[bool, str]:
        """Compile the solution. Returns (compiled, error_msg)."""
        ...

    @abstractmethod
    def run_correctness(self, config: TaskConfig) -> tuple[bool, dict, str]:
        """Check correctness. Returns (correct, detail_dict, error_msg)."""
        ...

    @abstractmethod
    def run_performance(self, config: TaskConfig) -> PerfMetrics:
        """Measure performance. Returns PerfMetrics."""
        ...

    @abstractmethod
    def cleanup(self, config: TaskConfig) -> None:
        """Clean up build artifacts."""
        ...


def resolve_backend(backend_name: str) -> Backend:
    """Resolve a backend name to a Backend instance."""
    if backend_name == "class1_make":
        from .backend_class1 import Class1MakeBackend
        return Class1MakeBackend()
    elif backend_name == "class2_defpy":
        from .backend_class2 import Class2DefpyBackend
        return Class2DefpyBackend()
    elif backend_name == "class3_app":
        from .backend_class3 import Class3AppBackend
        return Class3AppBackend()
    elif backend_name == "class4_challenge":
        from .backend_class4 import Class4ChallengeBackend
        return Class4ChallengeBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
