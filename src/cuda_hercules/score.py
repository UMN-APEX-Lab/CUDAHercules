"""
Scoring metrics for CUDA-Hercules benchmark evaluation.

Metrics:
  - compilation_rate: Fraction of solutions that compiled
  - correctness_rate: Fraction that passed correctness checks
  - fastp(p): Fraction correct AND speedup > p
  - geo_mean_speedup: Geometric mean speedup (correct only)
  - geo_mean_speedup_faster: Geometric mean (correct AND faster only)
  - excessive_speedup_count: Number flagged for suspicious >10x speedup

Scoring design inspired by KernelBench (ScalingIntelligence).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .result import TaskResult, TaskStatus


@dataclass
class BenchmarkScore:
    """Aggregate benchmark scoring results."""
    num_tasks: int = 0
    num_compiled: int = 0
    num_correct: int = 0
    num_passed: int = 0
    num_failed: int = 0
    num_skipped_arch: int = 0
    num_errors: int = 0

    compilation_rate: float = 0.0
    correctness_rate: float = 0.0
    fastp_0: float = 0.0       # fast_p(0): fraction correct (any speedup)
    fastp_1_0: float = 0.0     # fast_p(1.0): correct AND speedup >= 1x
    fastp_2_0: float = 0.0     # fast_p(2.0): correct AND speedup >= 2x
    fastp_5_0: float = 0.0     # fast_p(5.0): correct AND speedup >= 5x
    geo_mean_speedup: float = 0.0
    geo_mean_speedup_faster: float = 0.0

    excessive_speedup_count: int = 0
    excessive_speedup_threshold: float = 10.0

    per_class: dict = field(default_factory=dict)
    per_domain: dict = field(default_factory=dict)


def fastp(results: list[TaskResult], p: float) -> float:
    """Fraction of tasks that are correct AND have speedup >= p."""
    total = len(results)
    if total == 0:
        return 0.0
    count = sum(1 for r in results if r.correct and r.speedup >= p)
    return count / total


def geo_mean_speedup(results: list[TaskResult], faster_only: bool = False) -> float:
    """Geometric mean of speedups for correct tasks."""
    speedups = [r.speedup for r in results if r.correct and r.speedup > 0]
    if faster_only:
        speedups = [s for s in speedups if s > 1.0]
    if not speedups:
        return 0.0
    return math.exp(sum(math.log(s) for s in speedups) / len(speedups))


def compute_scores(
    results: list[TaskResult],
    excessive_speedup_threshold: float = 10.0,
) -> BenchmarkScore:
    """Compute comprehensive benchmark scores from task results."""
    score = BenchmarkScore()
    score.num_tasks = len(results)
    score.excessive_speedup_threshold = excessive_speedup_threshold

    if not results:
        return score

    score.num_compiled = sum(1 for r in results if r.compiled)
    score.num_correct = sum(1 for r in results if r.correct)
    score.num_passed = sum(1 for r in results if r.status == TaskStatus.PASS)
    score.num_failed = sum(1 for r in results if r.status == TaskStatus.FAIL)
    score.num_skipped_arch = sum(1 for r in results if r.status == TaskStatus.SKIP_ARCH)
    score.num_errors = sum(1 for r in results if r.status == TaskStatus.ERROR)

    total = score.num_tasks
    score.compilation_rate = score.num_compiled / total
    score.correctness_rate = score.num_correct / total

    score.fastp_0 = fastp(results, 0.0)
    score.fastp_1_0 = fastp(results, 1.0)
    score.fastp_2_0 = fastp(results, 2.0)
    score.fastp_5_0 = fastp(results, 5.0)

    score.geo_mean_speedup = geo_mean_speedup(results, faster_only=False)
    score.geo_mean_speedup_faster = geo_mean_speedup(results, faster_only=True)

    score.excessive_speedup_count = sum(
        1 for r in results
        if r.correct and r.speedup > excessive_speedup_threshold
    )

    # Per-class breakdown
    from collections import defaultdict
    by_class = defaultdict(list)
    by_domain = defaultdict(list)
    for r in results:
        by_class[r.level].append(r)
        if r.domain:
            by_domain[r.domain].append(r)

    for cls, cls_results in by_class.items():
        n = len(cls_results)
        score.per_class[cls] = {
            "num_tasks": n,
            "compilation_rate": sum(1 for r in cls_results if r.compiled) / n,
            "correctness_rate": sum(1 for r in cls_results if r.correct) / n,
            "fastp_1_0": fastp(cls_results, 1.0),
            "geo_mean_speedup": geo_mean_speedup(cls_results),
        }

    for domain, dom_results in by_domain.items():
        n = len(dom_results)
        score.per_domain[domain] = {
            "num_tasks": n,
            "correctness_rate": sum(1 for r in dom_results if r.correct) / n,
            "fastp_1_0": fastp(dom_results, 1.0),
            "geo_mean_speedup": geo_mean_speedup(dom_results),
        }

    return score


def format_score(score: BenchmarkScore) -> str:
    """Format a BenchmarkScore as a human-readable report."""
    lines = [
        f"{'='*70}",
        f"CUDA-Hercules Benchmark Results",
        f"{'='*70}",
        f"",
        f"Tasks:          {score.num_tasks}",
        f"  Compiled:     {score.num_compiled} ({score.compilation_rate:.1%})",
        f"  Correct:      {score.num_correct} ({score.correctness_rate:.1%})",
        f"  Passed:       {score.num_passed}",
        f"  Failed:       {score.num_failed}",
        f"  Skipped:      {score.num_skipped_arch}",
        f"  Errors:       {score.num_errors}",
        f"",
        f"Speedup Metrics:",
        f"  fast_p(0):    {score.fastp_0:.4f}  (correct, any speedup)",
        f"  fast_p(1.0):  {score.fastp_1_0:.4f}  (correct AND >= 1x)",
        f"  fast_p(2.0):  {score.fastp_2_0:.4f}  (correct AND >= 2x)",
        f"  fast_p(5.0):  {score.fastp_5_0:.4f}  (correct AND >= 5x)",
        f"  geo_mean:     {score.geo_mean_speedup:.4f}x  (correct only)",
        f"  geo_mean>1x:  {score.geo_mean_speedup_faster:.4f}x  (correct AND faster)",
        f"",
    ]

    if score.excessive_speedup_count > 0:
        lines.append(f"WARNING: {score.excessive_speedup_count} tasks with excessive speedup "
                     f"(>{score.excessive_speedup_threshold}x) - may indicate reward hacking")
        lines.append("")

    if score.per_class:
        lines.append("Per-Class Breakdown:")
        for cls in sorted(score.per_class):
            s = score.per_class[cls]
            lines.append(f"  Class {cls}: {s['num_tasks']} tasks, "
                         f"correct {s['correctness_rate']:.1%}, "
                         f"fast_p(1) {s['fastp_1_0']:.2f}, "
                         f"geo_mean {s['geo_mean_speedup']:.2f}x")
        lines.append("")

    if score.per_domain:
        lines.append("Per-Domain Breakdown:")
        for dom in sorted(score.per_domain):
            s = score.per_domain[dom]
            lines.append(f"  {dom}: {s['num_tasks']} tasks, "
                         f"correct {s['correctness_rate']:.1%}, "
                         f"geo_mean {s['geo_mean_speedup']:.2f}x")

    lines.append(f"{'='*70}")
    return "\n".join(lines)
