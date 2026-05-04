"""
Utility functions for CUDA-Hercules.
"""

import json
import os
import re


def extract_cuda_code(response: str) -> str:
    """
    Extract the first CUDA/C++ code block from an LLM response.

    Looks for ```cuda, ```cpp, or ```c++ fenced code blocks.
    Falls back to the first generic code block if no CUDA-specific one is found.
    """
    # Try CUDA/C++ specific code blocks first
    for lang in ["cuda", "cpp", "c++", "c"]:
        pattern = rf"```{re.escape(lang)}\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: any code block
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No code block found, return entire response (might be raw code)
    return response.strip()


def extract_fenced_code(response: str, preferred_langs: list[str] | None = None) -> str:
    """
    Extract code/text from a fenced block, optionally preferring specific languages.

    Falls back to the first fenced block, then to the raw response content.
    """
    langs = preferred_langs or []
    for lang in langs:
        pattern = rf"```{re.escape(lang)}\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


def extract_json_payload(response: str):
    """
    Extract the first complete JSON object or array from a response.

    Returns the decoded JSON value, or None if no valid payload is found.
    """
    candidates = []

    fenced_json = extract_fenced_code(response, ["json", "javascript"])
    if fenced_json:
        candidates.append(fenced_json)
    raw = response.strip()
    if raw:
        candidates.append(raw)

    decoder = json.JSONDecoder()

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass

        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
                return obj
            except Exception:
                continue

    return None


def get_project_root() -> str:
    """Get the CUDA-Hercules project root directory."""
    # Walk up from this file to find the root
    current = os.path.dirname(os.path.abspath(__file__))
    while current != "/":
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)
    raise RuntimeError("Could not find CUDA-Hercules project root")


def get_gpu_name() -> str:
    """Get the name of the current CUDA GPU."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "No GPU"


def get_cuda_arch() -> int:
    """Get the compute capability of the current GPU as an int (e.g., 80 for SM80)."""
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0
