"""
Static checker for reward-hacking detection in generated CUDA code.

Checks that the generated solution is a genuine CUDA kernel and not
bypassing the benchmark by calling existing implementations.

Layered approach (inspired by KernelBench):
  - Strict: Blocked patterns → FAIL
  - Required: Must-have patterns → FAIL if missing
  - Config: Task-specific blocked/required from task.yaml
  - Warning: Suspicious but allowed
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CheckResult:
    """Result of static checking."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Global patterns (apply to all tasks) ──────────────────────────────

# Patterns that always indicate reward hacking
GLOBAL_BLOCKED = [
    # Using existing flash-attention source code
    (r"#include\s*[<\"]flash_attn", "Includes flash_attn headers"),
    (r"#include\s*[<\"]flash_fwd", "Includes flash_fwd headers"),
    (r"#include\s*[<\"]flash_bwd", "Includes flash_bwd headers"),
    (r"#include\s*[<\"]ln_fwd_kernels", "Includes layernorm fwd headers"),
    (r"#include\s*[<\"]ln_bwd_kernels", "Includes layernorm bwd headers"),
    # PyTorch/ATen C++ APIs (bypasses CUDA requirement)
    (r"\btorch::", "Uses PyTorch C++ API"),
    (r"\bat::", "Uses ATen C++ API"),
    (r"\bc10::", "Uses c10 C++ API"),
    # Try-except bypass (model catches errors, falls back to reference)
    (r"\btry\s*\{", "Uses try-catch block (potential bypass)"),
    # CUTLASS / cuBLAS / cuDNN — solutions must be hand-written CUDA, not
    # a thin wrapper around a library. The reference already uses these
    # (that's what the solution has to BEAT); using them in the solution
    # means no kernel authoring is happening. A trivial `__global__` stub
    # declared to satisfy GLOBAL_REQUIRED is not a defence — this check
    # runs independently of the required-patterns check.
    (r"#include\s*[<\"]cutlass[/>\"]", "Includes CUTLASS headers"),
    (r"\bcutlass::(gemm|conv|epilogue|reduction|layout|arch|numeric_types)\b",
     "Uses CUTLASS template library"),
    (r"#include\s*[<\"]cublas", "Includes cuBLAS headers"),
    (r"\bcublas[A-Z][a-zA-Z_]*\(", "Calls cuBLAS API"),
    (r"#include\s*[<\"]cudnn", "Includes cuDNN headers"),
    (r"\bcudnn[A-Z][a-zA-Z_]*\(", "Calls cuDNN API"),
    (r"#include\s*[<\"]cufft", "Includes cuFFT headers"),
    (r"\bcufft[A-Z][a-zA-Z_]*\(", "Calls cuFFT API"),
]

# Patterns that a valid CUDA kernel must have.
# Note the attribute-aware spacing: models commonly emit
#   `__global__ __launch_bounds__(256, 2) void foo(...)` or
#   `__global__ __restrict__ void foo(...)` or
#   `__global__` + newline + attribute macro + `void`, etc.
# The old regex `__global__\s+void` rejected all of those as missing, so
# we allow an optional chain of attribute-style tokens (identifiers or
# `__foo__(...)` macros) between `__global__` and `void`.
_GLOBAL_KERNEL_RE = (
    r"__global__"
    r"(?:\s+(?:__[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?|[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?))*"
    r"\s+void"
)

GLOBAL_REQUIRED = [
    (_GLOBAL_KERNEL_RE, "Must contain at least one __global__ kernel function"),
    (r"<<<.*>>>", "Must contain at least one kernel launch (<<<...>>>)"),
]

# Suspicious but allowed with warnings
GLOBAL_WARNINGS = [
    (r"__nv_bfloat16", "Uses bfloat16 — ensure correctness with fp16 tolerance"),
    (r"cooperative_groups", "Uses cooperative groups — advanced feature"),
    (r"cudaDeviceSynchronize", "Uses explicit device sync — may hide timing issues"),
]


def validate_cuda_solution(
    code: str,
    blocked_patterns: Optional[list[str]] = None,
    required_patterns: Optional[list[str]] = None,
    allow_torch_cpp_api: bool = False,
) -> CheckResult:
    """
    Validate a generated CUDA solution for reward hacking.

    Args:
        code: The generated CUDA source code.
        blocked_patterns: Additional task-specific blocked regex patterns.
        required_patterns: Additional task-specific required regex patterns.
        allow_torch_cpp_api: Allow `torch::`, `at::`, and `c10::` C++ API
            namespaces for tasks implemented as PyTorch CUDA extensions.

    Returns:
        CheckResult with valid flag, errors, and warnings.
    """
    result = CheckResult()

    blocked_patterns_to_check = GLOBAL_BLOCKED
    if allow_torch_cpp_api:
        blocked_patterns_to_check = [
            (pattern, message)
            for pattern, message in GLOBAL_BLOCKED
            if pattern not in (r"\btorch::", r"\bat::", r"\bc10::")
        ]

    # Global blocked patterns
    for pattern, message in blocked_patterns_to_check:
        if re.search(pattern, code, re.IGNORECASE):
            result.errors.append(f"BLOCKED: {message}")
            result.valid = False

    # Task-specific blocked patterns (from task.yaml anti_cheat.blocked_patterns)
    if blocked_patterns:
        for pattern in blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                result.errors.append(f"BLOCKED (task-specific): pattern '{pattern}' found")
                result.valid = False

    # Global required patterns
    for pattern, message in GLOBAL_REQUIRED:
        if not re.search(pattern, code):
            result.errors.append(f"MISSING: {message}")
            result.valid = False

    # Task-specific required patterns.
    # Many class1 task.yaml files literally spell `__global__\s+void`, which
    # rejects valid `__global__ __launch_bounds__(...) void` kernels. Upgrade
    # any such literal to the attribute-aware regex so a yaml tweak isn't
    # required per task.
    if required_patterns:
        for pattern in required_patterns:
            effective = _GLOBAL_KERNEL_RE if pattern == r"__global__\s+void" else pattern
            if not re.search(effective, code):
                result.errors.append(f"MISSING (task-specific): pattern '{pattern}' not found")
                result.valid = False

    # Warnings
    for pattern, message in GLOBAL_WARNINGS:
        if re.search(pattern, code):
            result.warnings.append(f"WARNING: {message}")

    # Check for empty/trivial solutions
    lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("//")]
    if len(lines) < 5:
        result.errors.append("TRIVIAL: Solution has fewer than 5 non-empty, non-comment lines")
        result.valid = False

    return result
