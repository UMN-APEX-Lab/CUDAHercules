"""
Compiler module: compile .cu files into Python-callable functions.

Given a CUDA source file and a C function signature, this module:
1. Parses the function signature to extract parameter names and types
2. Auto-generates a pybind11/torch wrapper .cpp
3. Compiles both with torch.utils.cpp_extension.load()
4. Returns a callable Python function
"""

import os
import re
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


CUDA_ALLOC_CACHE_INTERPOSE = r"""\
#include <cuda_runtime.h>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cuda_hercules_alloc_cache {

inline std::mutex& cache_mutex() {
    static std::mutex m;
    return m;
}

inline std::unordered_map<size_t, std::vector<void*>>& free_by_size() {
    static std::unordered_map<size_t, std::vector<void*>> cache;
    return cache;
}

inline std::unordered_map<void*, size_t>& live_sizes() {
    static std::unordered_map<void*, size_t> sizes;
    return sizes;
}

inline cudaError_t cached_cuda_malloc_impl(void** ptr, size_t size) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto& free_list = free_by_size()[size];
    if (!free_list.empty()) {
        *ptr = free_list.back();
        free_list.pop_back();
        live_sizes()[*ptr] = size;
        return cudaSuccess;
    }
    cudaError_t err = ::cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        live_sizes()[*ptr] = size;
    }
    return err;
}

// Template wrapper matches CUDA's own `cudaMalloc(T**, size_t)` signature so
// that call sites like `cudaMalloc(&float_ptr, n)` compile after the macro
// substitution below.
template <typename T>
inline cudaError_t cached_cuda_malloc(T** ptr, size_t size) {
    return cached_cuda_malloc_impl(reinterpret_cast<void**>(ptr), size);
}

// Also accept explicit `void**` casts.
inline cudaError_t cached_cuda_malloc(void** ptr, size_t size) {
    return cached_cuda_malloc_impl(ptr, size);
}

inline cudaError_t cached_cuda_free(void* ptr) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto& sizes = live_sizes();
    auto it = sizes.find(ptr);
    if (it == sizes.end()) {
        return ::cudaFree(ptr);
    }
    free_by_size()[it->second].push_back(ptr);
    sizes.erase(it);
    return cudaSuccess;
}

}  // namespace cuda_hercules_alloc_cache

#define cudaMalloc cuda_hercules_alloc_cache::cached_cuda_malloc
#define cudaFree cuda_hercules_alloc_cache::cached_cuda_free
"""


# C type → (torch dtype, C++ data_ptr template type, is_pointer)
C_TYPE_MAP = {
    "const float*":    ("torch::kFloat32", "float",             True),
    "float*":          ("torch::kFloat32", "float",             True),
    "const half*":     ("torch::kFloat16", "at::Half",          True),
    "half*":           ("torch::kFloat16", "at::Half",          True),
    "const __half*":   ("torch::kFloat16", "at::Half",          True),
    "__half*":         ("torch::kFloat16", "at::Half",          True),
    "const double*":   ("torch::kFloat64", "double",            True),
    "double*":         ("torch::kFloat64", "double",            True),
    "const int*":      ("torch::kInt32",   "int",               True),
    "int*":            ("torch::kInt32",   "int",               True),
    "const void*":     (None,              None,                True),   # generic pointer (accepts any dtype)
    "void*":           (None,              None,                True),   # generic pointer (accepts any dtype)
    "int":             (None,              None,                 False),
    "int64_t":         (None,              None,                 False),
    "float":           (None,              None,                 False),
    "double":          (None,              None,                 False),
    "bool":            (None,              None,                 False),
    "cudaStream_t":    (None,              None,                 False),  # special handling
}


def parse_function_signature(signature: str) -> dict:
    """
    Parse a C function signature into structured parameter info.

    Returns:
        {
            "name": "launch_layernorm_forward",
            "return_type": "void",
            "params": [
                {"type": "const float*", "name": "x", "is_pointer": True, ...},
                {"type": "int", "name": "rows", "is_pointer": False},
                {"type": "cudaStream_t", "name": "stream", "is_pointer": False},
            ]
        }
    """
    # Remove comments
    sig = re.sub(r'//[^\n]*', '', signature)
    # Collapse whitespace
    sig = ' '.join(sig.split())
    # Remove trailing semicolon
    sig = sig.strip().rstrip(';').strip()

    # Match: return_type function_name(params)
    m = re.match(r'(\w+)\s+(\w+)\s*\((.*)\)', sig, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse function signature: {signature}")

    return_type = m.group(1)
    func_name = m.group(2)
    params_str = m.group(3).strip()

    params = []
    if params_str:
        for param_str in params_str.split(','):
            param_str = param_str.strip()
            if not param_str:
                continue
            # Parse "const float* x" or "int rows" or "cudaStream_t stream"
            param_info = _parse_single_param(param_str)
            params.append(param_info)

    return {
        "name": func_name,
        "return_type": return_type,
        "params": params,
    }


def _parse_single_param(param_str: str) -> dict:
    """Parse a single parameter like 'const float* x' into structured info."""
    param_str = param_str.strip()

    # Try matching pointer types: "const float* name" or "float* name"
    m = re.match(r'(const\s+\w+\*|\w+\*)\s+(\w+)', param_str)
    if m:
        c_type = re.sub(r'\s+', ' ', m.group(1)).strip()
        # Normalize: "const float *" -> "const float*"
        c_type = re.sub(r'\s*\*', '*', c_type)
        name = m.group(2)
        type_info = C_TYPE_MAP.get(c_type)
        if type_info is None:
            raise ValueError(f"Unknown pointer type: {c_type}")
        torch_dtype, cpp_dtype, is_pointer = type_info
        return {
            "type": c_type,
            "name": name,
            "is_pointer": is_pointer,
            "torch_dtype": torch_dtype,
            "cpp_dtype": cpp_dtype,
        }

    # Try matching scalar types: "int rows", "float eps", "cudaStream_t stream"
    m = re.match(r'(\w+)\s+(\w+)', param_str)
    if m:
        c_type = m.group(1)
        name = m.group(2)
        if c_type not in C_TYPE_MAP:
            raise ValueError(f"Unknown scalar type: {c_type}")
        return {
            "type": c_type,
            "name": name,
            "is_pointer": False,
            "torch_dtype": None,
            "cpp_dtype": None,
            "is_stream": (c_type == "cudaStream_t"),
        }

    raise ValueError(f"Cannot parse parameter: {param_str}")


def generate_pybind_wrapper(parsed_sig: dict) -> str:
    """
    Generate a pybind11 wrapper .cpp that bridges Python (torch::Tensor) to C CUDA functions.

    The wrapper function accepts torch::Tensor for pointer args and Python scalars for others.
    cudaStream_t is obtained from the current CUDA stream.
    """
    func_name = parsed_sig["name"]
    params = parsed_sig["params"]

    # Build the extern "C" declaration
    extern_params = ", ".join(f"{p['type']} {p['name']}" for p in params)
    extern_decl = f'extern "C" void {func_name}({extern_params});'

    # Build wrapper function parameters (Python-facing)
    wrapper_params = []
    for p in params:
        if p.get("is_stream"):
            continue  # stream is obtained internally
        elif p["is_pointer"]:
            wrapper_params.append(f"torch::Tensor {p['name']}")
        elif p["type"] == "int":
            wrapper_params.append(f"int64_t {p['name']}")
        elif p["type"] == "int64_t":
            wrapper_params.append(f"int64_t {p['name']}")
        elif p["type"] == "float":
            wrapper_params.append(f"double {p['name']}")
        elif p["type"] == "double":
            wrapper_params.append(f"double {p['name']}")
        elif p["type"] == "bool":
            wrapper_params.append(f"bool {p['name']}")

    wrapper_params_str = ", ".join(wrapper_params)

    # Build the call arguments
    call_args = []
    for p in params:
        if p.get("is_stream"):
            call_args.append("at::cuda::getCurrentCUDAStream().stream()")
        elif p["is_pointer"]:
            if p["cpp_dtype"] is None:
                # void* — use raw .data_ptr() with cast
                if "const" in p["type"]:
                    call_args.append(f"const_cast<const void*>({p['name']}.data_ptr())")
                else:
                    call_args.append(f"{p['name']}.data_ptr()")
            else:
                call_args.append(f"{p['name']}.data_ptr<{p['cpp_dtype']}>()")
        elif p["type"] in ("int", "int64_t"):
            call_args.append(f"static_cast<{p['type']}>({p['name']})")
        elif p["type"] == "float":
            call_args.append(f"static_cast<float>({p['name']})")
        elif p["type"] == "double":
            call_args.append(f"static_cast<double>({p['name']})")
        elif p["type"] == "bool":
            call_args.append(p["name"])

    call_args_str = ", ".join(call_args)

    wrapper_name = f"wrapper_{func_name}"

    cpp_code = f"""\
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

{extern_decl}

void {wrapper_name}({wrapper_params_str}) {{
    {func_name}({call_args_str});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{func_name}", &{wrapper_name}, "{func_name}");
}}
"""
    return cpp_code


def compile_cuda_module(
    cuda_source: str,
    function_signature: str,
    extra_include_dirs: Optional[list[str]] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
    extra_cflags: Optional[list[str]] = None,
    extra_ldflags: Optional[list[str]] = None,
    interpose_cached_allocators: bool = False,
    build_dir: Optional[str] = None,
    verbose: bool = False,
):
    """
    Compile a .cu file (as string or path) into a Python-callable function.

    Args:
        cuda_source: Either a file path to a .cu file, or the CUDA source code as a string.
        function_signature: The C function signature the CUDA code exports.
        extra_include_dirs: Additional include paths (e.g., CUTLASS headers for reference).
        extra_cuda_cflags: Additional nvcc flags.
        extra_cflags: Additional C++ compiler flags.
        extra_ldflags: Additional linker flags (e.g., -lcuda -lnvrtc).
        build_dir: Directory for build artifacts. Auto-generated if None.
        verbose: Print compilation output.

    Returns:
        A callable Python function that wraps the CUDA function.
    """
    parsed_sig = parse_function_signature(function_signature)
    func_name = parsed_sig["name"]

    # Generate wrapper cpp
    wrapper_cpp = generate_pybind_wrapper(parsed_sig)

    # Determine if cuda_source is a file path or inline source
    if os.path.isfile(cuda_source):
        cuda_source_path = os.path.abspath(cuda_source)
        source_hash = hashlib.md5(
            Path(cuda_source_path).read_bytes()
        ).hexdigest()[:8]
    else:
        # Write inline source to temp file
        source_hash = hashlib.md5(cuda_source.encode()).hexdigest()[:8]
        if build_dir is None:
            build_dir = tempfile.mkdtemp(prefix="cuda_hercules_")
        cuda_source_path = os.path.join(build_dir, f"{func_name}_{source_hash}.cu")
        with open(cuda_source_path, 'w') as f:
            f.write(cuda_source)

    if interpose_cached_allocators:
        if build_dir is None:
            build_dir = tempfile.mkdtemp(prefix="cuda_hercules_")
        wrapped_source_path = os.path.join(build_dir, f"{func_name}_{source_hash}_alloccache.cu")
        with open(wrapped_source_path, "w") as f:
            f.write(CUDA_ALLOC_CACHE_INTERPOSE)
            f.write("\n")
            f.write(f'#include "{cuda_source_path}"\n')
        cuda_source_path = wrapped_source_path

    # Write wrapper to temp file
    if build_dir is None:
        build_dir = tempfile.mkdtemp(prefix="cuda_hercules_")
    wrapper_path = os.path.join(build_dir, f"wrapper_{func_name}_{source_hash}.cpp")
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_cpp)

    # Default compilation flags
    default_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    if extra_cuda_cflags:
        default_cuda_cflags.extend(extra_cuda_cflags)

    default_cflags = ["-O3", "-std=c++17"]
    if extra_cflags:
        default_cflags.extend(extra_cflags)

    include_dirs = []
    if extra_include_dirs:
        include_dirs.extend(extra_include_dirs)

    module_variant = "alloccache" if interpose_cached_allocators else "plain"
    module_name = f"cuda_hercules_{func_name}_{source_hash}_{module_variant}"

    default_ldflags = []
    if extra_ldflags:
        default_ldflags.extend(extra_ldflags)

    if build_dir is not None:
        os.makedirs(build_dir, exist_ok=True)
        _clear_stale_torch_build_lock(build_dir, verbose=verbose)

    # Force verbose=False so torch captures ninja stdout/stderr into the
    # RuntimeError message on failure. With verbose=True torch streams to the
    # terminal and the exception only carries "Error building extension 'xxx'",
    # which destroys nvcc diagnostics needed by self-refine.
    try:
        module = load(
            name=module_name,
            sources=[wrapper_path, cuda_source_path],
            extra_include_paths=include_dirs,
            extra_cuda_cflags=default_cuda_cflags,
            extra_cflags=default_cflags,
            extra_ldflags=default_ldflags if default_ldflags else None,
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:
        if verbose:
            print(str(e))
        raise

    return getattr(module, func_name)


def _clear_stale_torch_build_lock(build_dir: str, verbose: bool = False) -> None:
    """Remove a stale torch cpp_extension baton lock if no build is active.

    torch.utils.cpp_extension uses build_directory/lock as a file baton.
    If a previous Python process is killed while holding the baton, future
    load() calls block forever in FileBaton.wait() even when no ninja/nvcc
    process is still running.
    """
    lock_path = os.path.join(build_dir, "lock")
    if not os.path.exists(lock_path):
        return

    if _has_live_build_process(build_dir):
        if verbose:
            print(f"[compiler] Build lock present and live compiler detected: {lock_path}")
        return

    try:
        os.remove(lock_path)
        if verbose:
            print(f"[compiler] Removed stale torch extension lock: {lock_path}")
    except FileNotFoundError:
        pass


def _has_live_build_process(build_dir: str) -> bool:
    """Best-effort check for active compiler processes using this build dir."""
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False

    compiler_markers = ("ninja", "nvcc", "gcc", "g++", "c++", "cc1plus")
    for line in proc.stdout.splitlines():
        if build_dir not in line:
            continue
        if any(marker in line for marker in compiler_markers):
            return True
    return False


def get_reference_include_dirs(root_dir: str, source_repo: str = "flash-attention") -> list[str]:
    """
    Get include directories for compiling reference code from a source repo.

    Args:
        root_dir: CUDA-Hercules root directory.
        source_repo: Which source repo's headers to include.

    Returns:
        List of include directory paths.
    """
    ref_base = os.path.join(root_dir, "reference_sources", source_repo)
    if not os.path.isdir(ref_base):
        # Try direct path
        ref_base = os.path.join(root_dir, source_repo)

    dirs = []
    if source_repo == "flash-attention":
        dirs = [
            os.path.join(ref_base, "csrc", "flash_attn"),
            os.path.join(ref_base, "csrc", "flash_attn", "src"),
            os.path.join(ref_base, "csrc", "cutlass", "include"),
            os.path.join(ref_base, "csrc", "layer_norm"),
        ]

    return [d for d in dirs if os.path.isdir(d)]
