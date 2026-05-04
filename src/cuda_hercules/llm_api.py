"""
Unified LLM API interface for CUDA-Hercules.

Supports two backends:
  - "openai" (default): OpenAI-compatible API for local servers, OpenAI, DeepSeek, etc.
  - "vertex": Google Cloud Vertex AI for Anthropic Claude models.

Features:
  - Local servers (vLLM, SGLang, Ollama) via api_base
  - Cloud providers (OpenAI, DeepSeek, etc.) via api_key
  - Vertex AI (Claude on GCP) via backend="vertex"
  - Reasoning models (o1/o3, Claude thinking, Qwen thinking) via is_reasoning_model
  - Multiple completions (pass@N evaluation) via num_completions
  - Server presets for common configurations

Design inspired by KernelBench (ScalingIntelligence).
"""

import os
import time
from typing import Optional

from openai import OpenAI, APITimeoutError, APIConnectionError

from .utils import extract_cuda_code


# ── Server Presets ────────────────────────────────────────────────────

SERVER_PRESETS = {
    "local": {
        "api_base": "http://localhost:8000/v1",
        "temperature": 0.6,
        "max_tokens": 16384,
    },
    "openai": {
        "model": "gpt-5.4",
        "temperature": 0.0,
        "max_tokens": 16384,
    },
    "deepseek": {
        "model": "deepseek-coder",
        "api_base": "https://api.deepseek.com/v1",
        "temperature": 1.6,
        "max_tokens": 8192,
    },
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 16384,
    },
    "vertex": {
        "backend": "vertex",
        "model": "claude-sonnet-4-6",
        "vertex_region": "global",
        "vertex_project": "neu-research",
        "temperature": 0.7,
        "max_tokens": 16384,
    },
}


# ── Core API ──────────────────────────────────────────────────────────

def _get_client(api_base: str = "", api_key: str = "") -> OpenAI:
    """Create an OpenAI client for the given endpoint."""
    kwargs = {}
    if api_base:
        kwargs["base_url"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    elif api_base and not os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = "dummy"
    return OpenAI(**kwargs)


def _get_vertex_client(region: str = "global", project_id: str = "neu-research"):
    """Create an AnthropicVertex client for Vertex AI."""
    from anthropic import AnthropicVertex
    return AnthropicVertex(region=region, project_id=project_id)


def _query_vertex(
    prompt: str | list[dict],
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    num_completions: int,
    vertex_region: str,
    vertex_project: str,
    top_p: float,
    is_reasoning_model: bool,
) -> str | list[str]:
    """Query Vertex AI (Anthropic Claude) and return the response(s)."""
    client = _get_vertex_client(vertex_region, vertex_project)

    # Build messages — Anthropic API uses system as a top-level param
    if isinstance(prompt, list):
        # Extract system message if present
        sys_msg = ""
        user_messages = []
        for m in prompt:
            if m["role"] == "system":
                sys_msg = m["content"]
            else:
                user_messages.append(m)
        messages = user_messages
        if sys_msg:
            system_prompt = sys_msg
    else:
        messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if not is_reasoning_model:
        # Anthropic API doesn't allow both temperature and top_p simultaneously
        kwargs["temperature"] = temperature

    # Anthropic API doesn't support n>1, so loop for multiple completions
    # Use streaming to avoid 10-minute timeout on long generations
    outputs = []
    for _ in range(num_completions):
        collected = []
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                collected.append(text)
        content = "".join(collected)
        outputs.append(content)

    return outputs[0] if num_completions == 1 else outputs


def query_server(
    prompt: str | list[dict],
    model: str = "gpt-4o",
    system_prompt: str = "You are a CUDA kernel programming expert. Write high-performance, correct CUDA code.",
    temperature: float = 0.6,
    max_tokens: int = 16384,
    num_completions: int = 1,
    api_base: str = "",
    api_key: str = "",
    is_reasoning_model: bool = False,
    reasoning_effort: str = "",
    top_p: float = 1.0,
    backend: str = "openai",
    vertex_region: str = "global",
    vertex_project: str = "neu-research",
) -> str | list[str]:
    """
    Query an LLM server and return the response(s).

    Args:
        prompt: User prompt string, or list of message dicts for chat format.
        model: Model name.
        system_prompt: System message (ignored if prompt is already chat format).
        temperature: Sampling temperature (ignored for reasoning models).
        max_tokens: Maximum output tokens.
        num_completions: Number of completions (for pass@N).
        api_base: API base URL for local/custom servers.
        api_key: API key override.
        is_reasoning_model: If True, skip temperature/top_p (for o1/o3/etc).
        reasoning_effort: For o1/o3 models ("low", "medium", "high").
        top_p: Nucleus sampling parameter.
        backend: "openai" (default) or "vertex" (Vertex AI).
        vertex_region: Vertex AI region (default "global").
        vertex_project: Vertex AI project ID (default "neu-research").

    Returns:
        Single string if num_completions=1, list of strings otherwise.
    """
    if backend == "vertex":
        return _query_vertex(
            prompt=prompt, model=model, system_prompt=system_prompt,
            temperature=temperature, max_tokens=max_tokens,
            num_completions=num_completions,
            vertex_region=vertex_region, vertex_project=vertex_project,
            top_p=top_p, is_reasoning_model=is_reasoning_model,
        )

    # Default: OpenAI-compatible backend
    client = _get_client(api_base, api_key)

    # Build messages
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

    # Build kwargs
    kwargs = {
        "model": model,
        "messages": messages,
        "n": num_completions,
    }

    # Models that require max_completion_tokens instead of max_tokens
    _needs_max_completion_tokens = is_reasoning_model or model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")

    if _needs_max_completion_tokens:
        kwargs["max_completion_tokens"] = max_tokens
        if is_reasoning_model:
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    # 300s per attempt; retry once on timeout/connection hang (e.g. vLLM
    # server killing the TCP connection mid-request — without an explicit
    # timeout the httpx client can block in select() indefinitely).
    request_timeout = 300.0
    try:
        response = client.chat.completions.create(timeout=request_timeout, **kwargs)
    except (APITimeoutError, APIConnectionError) as e:
        print(f"[llm_api] {type(e).__name__} after {request_timeout:.0f}s; retrying once", flush=True)
        response = client.chat.completions.create(timeout=request_timeout, **kwargs)

    # Extract outputs
    outputs = []
    for choice in response.choices:
        content = choice.message.content
        if content is None:
            # Thinking models may have content in reasoning field
            content = getattr(choice.message, "reasoning", None) or ""
        outputs.append(content)

    return outputs[0] if num_completions == 1 else outputs


def generate_kernel(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.6,
    max_tokens: int = 16384,
    api_base: str = "",
    api_key: str = "",
    backend: str = "openai",
    vertex_region: str = "global",
    vertex_project: str = "neu-research",
    **kwargs,
) -> str:
    """Query an LLM and extract CUDA code from the response."""
    response = query_server(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key,
        backend=backend,
        vertex_region=vertex_region,
        vertex_project=vertex_project,
        **kwargs,
    )
    return extract_cuda_code(response)


# ── Factory Function ──────────────────────────────────────────────────

def create_inference_server(
    preset: str = "",
    model: str = "",
    api_base: str = "",
    api_key: str = "",
    backend: str = "",
    vertex_region: str = "",
    vertex_project: str = "",
    greedy: bool = False,
    verbose: bool = False,
    **overrides,
) -> callable:
    """
    Create a reusable LLM query function from a preset or custom config.

    Args:
        preset: Preset name ("local", "openai", "deepseek", "anthropic", "vertex").
        model: Model name override.
        api_base: API base URL override.
        api_key: API key override.
        backend: "openai" or "vertex". Overrides preset default.
        vertex_region: Vertex AI region override.
        vertex_project: Vertex AI project ID override.
        greedy: If True, set temperature=0.
        verbose: Print query info.
        **overrides: Additional keyword args to pass to query_server.

    Returns:
        Callable that takes (prompt) and returns LLM response string.

    Usage:
        llm = create_inference_server(preset="local", model="Qwen/Qwen3.5-35B-A3B")
        llm = create_inference_server(preset="vertex")  # Vertex AI with defaults
        code = llm("Write a CUDA kernel for matrix multiply")
    """
    config = {}
    if preset and preset in SERVER_PRESETS:
        config.update(SERVER_PRESETS[preset])

    if model:
        config["model"] = model
    if api_base:
        config["api_base"] = api_base
    if api_key:
        config["api_key"] = api_key
    if backend:
        config["backend"] = backend
    if vertex_region:
        config["vertex_region"] = vertex_region
    if vertex_project:
        config["vertex_project"] = vertex_project
    if greedy:
        config["temperature"] = 0.0
        config["top_p"] = 1.0

    config.update(overrides)

    def _query(prompt: str | list[dict]) -> str:
        if verbose:
            print(f"[LLM] model={config.get('model','?')}, "
                  f"temp={config.get('temperature','?')}, "
                  f"max_tokens={config.get('max_tokens','?')}")
            t0 = time.time()

        result = query_server(prompt=prompt, **config)

        if verbose:
            elapsed = time.time() - t0
            print(f"[LLM] Response in {elapsed:.1f}s")

        return result

    return _query
