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
    # Prebuilt library kernels — solutions must author their own kernels,
    # not wrap a library kernel that already implements the operation. The
    # reference already uses these (that's what the solution has to BEAT).
    # A trivial `__global__` stub declared to satisfy GLOBAL_REQUIRED is not
    # a defence — this check runs independently of the required-patterns
    # check.
    #
    # For CUTLASS only the kernel-providing layers are blocked:
    # device/kernel/collective/threadblock-level GEMM/conv/epilogue/
    # reduction/transform templates, the reference kernels, the CUTLASS
    # library, and the util/ headers that ship complete kernels (layernorm,
    # rmsnorm, ...) or wrap cuBLAS. CuTe (`cute::`, `#include <cute/...>`)
    # and CUTLASS's low-level building blocks (numeric types, cutlass::arch
    # barriers / register control, pipelines, layouts, warp-level MMA) stay
    # allowed — they are a way to write kernels, not kernels.
    (r"#include\s*[<\"]cutlass/(gemm|conv)/(device|kernel|collective|threadblock)/",
     "Includes a prebuilt CUTLASS GEMM/conv kernel"),
    (r"\bcutlass::(gemm|conv)::(device|kernel|collective|threadblock)\b",
     "Uses a prebuilt CUTLASS GEMM/conv kernel"),
    (r"#include\s*[<\"]cutlass/epilogue/(collective|threadblock)/",
     "Includes a prebuilt CUTLASS epilogue"),
    (r"\bcutlass::epilogue::(collective|threadblock)\b",
     "Uses a prebuilt CUTLASS epilogue"),
    (r"#include\s*[<\"]cutlass/(reduction|transform)/(device|kernel|collective)/",
     "Includes a prebuilt CUTLASS reduction/transform kernel"),
    (r"\bcutlass::(reduction|transform)::(device|kernel|collective)\b",
     "Uses a prebuilt CUTLASS reduction/transform kernel"),
    (r"#include\s*[<\"]cutlass/util/(reference/|cublas_wrappers|device_(layernorm|rmsnorm|groupnorm"
     r"|nchw_to_nhwc|nhwc_to_nchw|nhwc_padding|nhwc_pooling))",
     "Includes a prebuilt CUTLASS util kernel"),
    (r"\bcutlass::(reference|library)::", "Uses CUTLASS reference/library kernels"),
    (r"#include\s*[<\"]cutlass/library/", "Includes the CUTLASS kernel library"),
    (r"#include\s*[<\"]cublas", "Includes cuBLAS headers"),
    (r"\bcublas[A-Z][a-zA-Z0-9_]*\(", "Calls cuBLAS API"),
    (r"#include\s*[<\"]cudnn", "Includes cuDNN headers"),
    (r"\bcudnn[A-Z][a-zA-Z0-9_]*\(", "Calls cuDNN API"),
    (r"#include\s*[<\"]cufft", "Includes cuFFT headers"),
    (r"\bcufft[A-Z][a-zA-Z0-9_]*\(", "Calls cuFFT API"),
]

# Prompt-facing statement of the library rules in GLOBAL_BLOCKED.
# prompt_builder and eval_agent embed this text verbatim so prompts cannot
# drift from what the checker actually enforces.
LIBRARY_RULES_PROMPT = (
    "- Do not call prebuilt library kernels: cuBLAS, cuDNN, cuFFT, or "
    "CUTLASS's device/kernel/collective/threadblock-level templates (e.g. "
    "`cutlass::gemm::device::Gemm`, `GemmUniversalAdapter`, "
    "`CollectiveBuilder`, `cutlass/util/device_layernorm.h`, "
    "`cutlass::reference::*`)\n"
    "- CuTe (`#include <cute/...>`, `cute::`) and CUTLASS's low-level "
    "building blocks (numeric types, `cutlass::arch` barriers and register "
    "control, pipelines, layouts, warp-level MMA) ARE allowed for writing "
    "your own kernels"
)

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
