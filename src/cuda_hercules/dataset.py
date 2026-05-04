"""
Dataset module for loading CUDA-Hercules benchmark tasks.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import toml


@dataclass
class TaskMetadata:
    """Parsed from metadata.toml."""
    source_repo: str = ""
    source_files: list[str] = field(default_factory=list)
    difficulty: str = ""
    expected_optimizations: list[str] = field(default_factory=list)
    bonus_optimizations: list[str] = field(default_factory=list)


@dataclass
class Task:
    """A single CUDA-Hercules benchmark task."""
    name: str
    path: str  # Directory path
    metadata: Optional[TaskMetadata] = None


def load_task_metadata(task_dir: str) -> Optional[TaskMetadata]:
    """Load metadata.toml from a task directory."""
    meta_path = os.path.join(task_dir, "metadata.toml")
    if not os.path.exists(meta_path):
        return None

    data = toml.load(meta_path)
    source = data.get("source", {})
    difficulty = data.get("difficulty", {})
    optimizations = data.get("optimizations", {})

    return TaskMetadata(
        source_repo=source.get("repo", ""),
        source_files=source.get("files", []),
        difficulty=difficulty.get("level", ""),
        expected_optimizations=optimizations.get("expected", []),
        bonus_optimizations=optimizations.get("bonus", []),
    )


def discover_tasks(root_dir: str, task_class: Optional[str] = None) -> list[Task]:
    """
    Discover all benchmark tasks in the tasks/ directory.

    Tasks are organized under tasks/class1/, tasks/class2/, tasks/class3/.

    Args:
        root_dir: CUDA-Hercules project root.
        task_class: If set, only return tasks from that class (e.g., "class2").

    Returns:
        List of Task objects, sorted by name.
    """
    tasks_dir = os.path.join(root_dir, "tasks")
    tasks = []

    search_root = os.path.join(tasks_dir, task_class) if task_class else tasks_dir

    for dirpath, dirs, files in os.walk(search_root):
        if "def.py" not in files:
            continue
        task_dir_name = os.path.basename(dirpath)
        metadata = load_task_metadata(dirpath)
        tasks.append(Task(
            name=task_dir_name,
            path=dirpath,
            metadata=metadata,
        ))

    tasks.sort(key=lambda t: t.name)
    return tasks
