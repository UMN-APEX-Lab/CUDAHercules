"""OpenAI-style chat-completions facade over the official Anthropic Messages API.

agent_loop.py speaks OpenAI chat-completions (messages with tool_calls / role=tool). This module
exposes the same `client.chat.completions.create(**kwargs)` surface, but sends the request to the
Anthropic API (adaptive thinking with a configurable effort, streaming, automatic prompt caching)
and converts the answer back to an OpenAI `ChatCompletion`.

Usage / spend is appended to <run_dir>/api_usage.jsonl after every call so a limited key can be
watched while a run is in progress. When the key is exhausted the API error propagates and ends
the task (it is not retried).

Requires the `anthropic` package (pip install anthropic).
"""

import json
import os
import re
import threading
import time

import anthropic
from openai.types.chat import ChatCompletion

# $/1M tokens, claude-opus-5 (first-party): input, cache write (5 min), cache read, output.
PRICE = {"in": 5.0, "cache_write": 6.25, "cache_read": 0.5, "out": 25.0}

_SHARED: dict = {}
_SHARED_LOCK = threading.Lock()


def shared_client(run_dir: str, **kw) -> "ClaudeClient":
    """One client per run, so usage totals accumulate across tasks."""
    with _SHARED_LOCK:
        if run_dir not in _SHARED:
            _SHARED[run_dir] = ClaudeClient(run_dir=run_dir, **kw)
        return _SHARED[run_dir]


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kw):
        return self._owner._create(**kw)


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class ClaudeClient:
    def __init__(self, *, run_dir: str, claude_key: str, effort: str = "low", read_timeout: float = 600.0):
        self.run_dir = run_dir
        self.effort = effort
        self._claude = anthropic.Anthropic(
            api_key=claude_key, timeout=anthropic.Timeout(read_timeout, connect=30.0), max_retries=2)
        self.chat = _Chat(self)
        self._lock = threading.Lock()
        self._thinking: dict = {}   # first tool_call id -> raw thinking blocks of that assistant turn
        self.totals = {"calls": 0, "in": 0, "cache_write": 0, "cache_read": 0, "out": 0, "usd": 0.0}
        os.makedirs(run_dir, exist_ok=True)

    # -- bookkeeping ----------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.totals)

    def _record(self, usage: dict, dt: float, extra: dict) -> None:
        usd = (usage["in"] * PRICE["in"] + usage["cache_write"] * PRICE["cache_write"]
               + usage["cache_read"] * PRICE["cache_read"] + usage["out"] * PRICE["out"]) / 1e6
        with self._lock:
            t = self.totals
            t["calls"] += 1
            for k in ("in", "cache_write", "cache_read", "out"):
                t[k] += usage[k]
            t["usd"] += usd
            rec = dict(ts=time.time(), dt=round(dt, 1), usd=round(usd, 4), usd_total=round(t["usd"], 3),
                       **usage, **extra)
            with open(os.path.join(self.run_dir, "api_usage.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")

    # -- request conversion ---------------------------------------------------
    def _to_anthropic(self, messages: list, use_thinking_blocks: bool):
        system = ""
        out: list = []

        def push(role, blocks):
            if out and out[-1]["role"] == role:
                if role == "user":
                    # tool_result blocks must stay in front of plain text
                    prev = out[-1]["content"]
                    tr = [b for b in prev + blocks if b["type"] == "tool_result"]
                    rest = [b for b in prev + blocks if b["type"] != "tool_result"]
                    out[-1]["content"] = tr + rest
                else:
                    out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            role = m["role"]
            if role == "system":
                system += (m.get("content") or "") + "\n"
            elif role == "user":
                push("user", [{"type": "text", "text": m["content"] or "(empty)"}])
            elif role == "assistant":
                blocks = []
                tcs = m.get("tool_calls") or []
                if use_thinking_blocks and tcs:
                    blocks.extend(self._thinking.get(tcs[0]["id"], []))
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in tcs:
                    try:
                        inp = json.loads(tc["function"]["arguments"] or "{}")
                        if not isinstance(inp, dict):
                            inp = {"value": inp}
                    except json.JSONDecodeError:
                        inp = {"_invalid_arguments": (tc["function"]["arguments"] or "")[:200]}
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": inp})
                if not blocks:
                    blocks = [{"type": "text", "text": "(no response)"}]
                push("assistant", blocks)
            elif role == "tool":
                push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                               "content": m["content"] or "(no output)"}])
        return system.strip(), out

    @staticmethod
    def _tools(tools: list) -> list:
        return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                 "input_schema": t["function"]["parameters"]} for t in tools]

    # -- API call ---------------------------------------------------------------
    def _call(self, kw: dict):
        max_tokens = kw.get("max_completion_tokens") or kw.get("max_tokens") or 65536
        for use_thinking_blocks in (True, False):
            system, msgs = self._to_anthropic(kw["messages"], use_thinking_blocks)
            params = dict(model=kw["model"], max_tokens=max_tokens, system=system, messages=msgs,
                          tools=self._tools(kw.get("tools") or []),
                          thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                          cache_control={"type": "ephemeral"})
            t0 = time.time()
            try:
                with self._claude.messages.stream(**params) as stream:
                    msg = stream.get_final_message()
            except anthropic.BadRequestError as e:
                # edited history (old tool outputs elided) can invalidate stored thinking blocks
                if use_thinking_blocks and re.search(r"thinking", str(e), re.I):
                    continue
                raise
            return msg, time.time() - t0
        raise RuntimeError("unreachable")

    def _to_completion(self, msg, model: str) -> ChatCompletion:
        text = "".join(b.text for b in msg.content if b.type == "text")
        think = [b.model_dump(exclude_none=True) for b in msg.content
                 if b.type in ("thinking", "redacted_thinking")]
        uses = [b for b in msg.content if b.type == "tool_use"]
        truncated = msg.stop_reason == "max_tokens"
        tool_calls = []
        for b in uses:
            # a tool_use cut off by max_tokens carries a silently truncated input: make it invalid
            # JSON so the agent loop reports the token limit instead of executing half a file
            args = "{TRUNCATED" if truncated else json.dumps(b.input)
            tool_calls.append({"id": b.id, "type": "function", "function": {"name": b.name, "arguments": args}})
        if tool_calls and think:
            self._thinking[tool_calls[0]["id"]] = think
        finish = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls",
                  "max_tokens": "length", "refusal": "content_filter"}.get(msg.stop_reason, "stop")
        u = msg.usage
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        prompt = u.input_tokens + cw + cr
        return ChatCompletion.model_validate({
            "id": msg.id, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text or None,
                                     "tool_calls": tool_calls or None}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": u.output_tokens,
                      "total_tokens": prompt + u.output_tokens}})

    def _create(self, **kw):
        msg, dt = self._call(kw)
        u = msg.usage
        self._record({"in": u.input_tokens,
                      "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
                      "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                      "out": u.output_tokens}, dt,
                     {"stop": msg.stop_reason, "request_id": getattr(msg, "_request_id", None)})
        return self._to_completion(msg, kw["model"])
