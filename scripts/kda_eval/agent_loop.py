#!/usr/bin/env python3
"""KDA autonomous agent loop over an OpenAI-compatible chat-completions API.

KDA (github.com/NVlabs/kda) is a workflow + prompts; this is the minimal
agent runtime that drives it without a human: the model gets the KDA
autonomous-flow prompt and four tools (bash, read_file, write_file,
edit_file). The loop runs OUTSIDE the sandbox and talks to the LLM API; every
tool call is executed in a fresh bwrap sandbox (sandbox.py) that has no
network, so the API key and the host filesystem never reach the agent.

KDA's budget hook (kda_lite/budget_guard.py) is called exactly as Claude
Code would call it: after each batch of tool calls ("post-tool-use", its note
is appended to the last tool result) and whenever the model ends its turn
("stop", a block decision is sent back as a user message).
"""

import json
import os
import subprocess
import sys
import time

from openai import (APIConnectionError, APITimeoutError, BadRequestError,
                    InternalServerError, OpenAI, RateLimitError)

from . import sandbox as sbx

# transient provider errors (OpenAI-compatible gateway, or the native Anthropic API when the optional
# `anthropic` package is installed and --claude-key-env is used): back off and retry
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
BAD_REQUEST = (BadRequestError,)
try:
    import anthropic
    RETRYABLE += (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
    BAD_REQUEST += (anthropic.BadRequestError,)
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
KDA_HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kda_lite", "budget_guard.py")

MAX_TOOL_OUTPUT = 30000
KEEP_RECENT_MESSAGES = 16

SYSTEM_PROMPT = """You are an autonomous GPU kernel engineer running the Kernel Design \
Agents (KDA) workflow. You act only through the provided tools. Every tool call runs in an \
isolated sandbox whose working directory is /workspace; there is no network access.

No human will read or answer your messages. Never ask questions or wait for confirmation: \
decide, record your reasoning in the workspace, and proceed. Keep making tool calls until the \
promotion criteria are met or the budget is exhausted; a reply without tool calls ends your \
turn. Write large files in pieces if needed (write_file, then edit_file or appending via bash) \
so that a single tool call stays well below your output limit."""

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": ("Run a bash command in the sandbox (cwd /workspace). Each call is a fresh "
                        "shell: only files under /workspace persist; /tmp, environment changes and "
                        "background processes do not survive between calls. Returns exit code, "
                        "stdout and stderr (long output is truncated in the middle)."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "description": "default 600, max 1800"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file with line numbers (max 2000 lines per call).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "absolute, or relative to /workspace"},
            "offset": {"type": "integer", "description": "1-based first line"},
            "limit": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file under /workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": ("Replace old_string with new_string in a file under /workspace. old_string "
                        "must occur exactly once unless replace_all is true."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string"},
            "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
            "required": ["path", "old_string", "new_string"]}}},
]


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-half:]


class AgentLoop:
    def __init__(self, client: OpenAI, model: str, spec: sbx.SandboxSpec, task_out: str,
                 judge_socket: str, deadline: float, *, max_completion_tokens: int,
                 context_limit: int, max_llm_calls: int, max_total_tokens: int,
                 reasoning_effort: str = ""):
        self.client = client
        self.model = model
        self.spec = spec
        self.task_out = task_out
        self.judge_socket = judge_socket
        self.deadline = deadline
        self.max_completion_tokens = max_completion_tokens
        self.context_limit = context_limit
        self.max_llm_calls = max_llm_calls
        self.max_total_tokens = max_total_tokens
        self.reasoning_effort = reasoning_effort
        self.transcript = open(os.path.join(task_out, "transcript.jsonl"), "a")
        self.t0 = time.time()
        self.stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "tool_calls": {}, "stop_blocks": 0, "api_retries": 0, "elisions": 0}

    # -- logging ------------------------------------------------------------
    def _log(self, **rec) -> None:
        rec["t"] = round(time.time() - self.t0, 2)
        self.transcript.write(json.dumps(rec, default=str) + "\n")
        self.transcript.flush()

    # -- KDA budget hook ----------------------------------------------------
    def _hook(self, event: str) -> dict | None:
        env = dict(os.environ,
                   KDA_BUDGET_STATUS_CMD=f"{sys.executable} {os.path.join(HERE, 'submit.py')} --budget-json",
                   KH_JUDGE_SOCKET=self.judge_socket,
                   KDA_BUDGET_STATE=os.path.join(self.task_out, "kda_budget_state.json"))
        try:
            p = subprocess.run([sys.executable, KDA_HOOK, event], input="{}", capture_output=True,
                               text=True, env=env, timeout=90)
            return json.loads(p.stdout) if p.stdout.strip() else None
        except Exception:
            return None

    # -- tools --------------------------------------------------------------
    def _remaining(self) -> float:
        return self.deadline - time.time()

    def _run_tool(self, name: str, args: dict) -> str:
        if name == "bash":
            timeout = min(int(args.get("timeout_s") or 600), 1800, max(int(self._remaining()) + 60, 60))
            rc, out, err, timed_out = sbx.run_in_sandbox(
                self.spec, ["/bin/bash", "-c", args["command"]], timeout_s=timeout)
            head = f"exit_code: {rc}" + (f" (killed after {timeout}s timeout)" if timed_out else "")
            return _truncate(f"{head}\n[stdout]\n{out}\n[stderr]\n{err}")
        if name in ("read_file", "write_file", "edit_file"):
            req = dict(args, op=name.split("_")[0])
            rc, out, err, _ = sbx.run_in_sandbox(
                self.spec, ["/usr/bin/python3", "/opt/kh/fstool.py"], stdin=json.dumps(req), timeout_s=120)
            try:
                resp = json.loads(out)
                return _truncate(resp["output"] if resp["ok"] else "ERROR: " + resp["output"])
            except Exception:
                return f"ERROR: file tool failed (rc={rc}): {err[-2000:]}"
        return f"ERROR: unknown tool {name!r}"

    # -- context management -------------------------------------------------
    def _elide_old(self, messages: list) -> None:
        """Shrink old tool outputs and old write_file payloads. The KDA
        evidence records in /workspace keep what matters re-readable."""
        cutoff = len(messages) - KEEP_RECENT_MESSAGES
        for m in messages[2:max(cutoff, 2)]:
            if m["role"] == "tool" and len(m["content"]) > 600:
                m["content"] = m["content"][:400] + "\n... [old tool output elided to save context] ..."
            elif m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc["function"]
                    if fn["name"] in ("write_file", "edit_file") and len(fn["arguments"]) > 600:
                        try:
                            a = json.loads(fn["arguments"])
                            for k in ("content", "old_string", "new_string"):
                                if k in a and len(a[k]) > 200:
                                    a[k] = a[k][:120] + f"... [{len(a[k])} chars elided]"
                            fn["arguments"] = json.dumps(a)
                        except Exception:
                            pass
        self.stats["elisions"] += 1

    # -- API ----------------------------------------------------------------
    def _complete(self, messages: list):
        kwargs = dict(model=self.model, messages=messages, tools=TOOLS, tool_choice="auto",
                      max_completion_tokens=self.max_completion_tokens)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        for attempt in range(8):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RETRYABLE as e:
                self.stats["api_retries"] += 1
                self._log(type="api_error", error=f"{type(e).__name__}: {e}"[:1000], attempt=attempt)
                if self._remaining() < 60:
                    raise
                time.sleep(min(120, 5 * 2 ** attempt))
            except BAD_REQUEST as e:
                msg = str(e).lower()
                self._log(type="api_error", error=f"BadRequest: {e}"[:2000], attempt=attempt)
                if any(s in msg for s in ("context", "too long", "too many tokens", "prompt is too long")):
                    self._elide_old(messages)
                    continue
                raise
        raise RuntimeError("API retries exhausted")

    # -- main loop ----------------------------------------------------------
    def run(self, user_prompt: str) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}]
        end_reason = "model_stopped"
        try:
            while True:
                if self._remaining() <= 0:
                    end_reason = "time_limit"
                    break
                if self.stats["llm_calls"] >= self.max_llm_calls:
                    end_reason = "max_llm_calls"
                    break
                if self.max_total_tokens and (self.stats["prompt_tokens"] + self.stats["completion_tokens"]
                                              >= self.max_total_tokens):
                    end_reason = "max_total_tokens"
                    break

                resp = self._complete(messages)
                self.stats["llm_calls"] += 1
                choice = resp.choices[0]
                msg = choice.message
                usage = resp.usage.model_dump() if resp.usage else {}
                self.stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
                self.stats["completion_tokens"] += usage.get("completion_tokens") or 0
                self._log(type="response", finish_reason=choice.finish_reason, usage=usage,
                          message=msg.model_dump(exclude_none=True))

                assistant = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    assistant["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls]
                messages.append(assistant)

                if not msg.tool_calls:
                    hook = self._hook("stop")
                    if hook and hook.get("decision") == "block":
                        self.stats["stop_blocks"] += 1
                        messages.append({"role": "user", "content": hook["reason"]})
                        self._log(type="stop_blocked", reason=hook["reason"])
                        continue
                    break

                for tc in assistant["tool_calls"]:
                    name = tc["function"]["name"]
                    self.stats["tool_calls"][name] = self.stats["tool_calls"].get(name, 0) + 1
                    t_tool = time.time()
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                        output = self._run_tool(name, args)
                    except json.JSONDecodeError:
                        output = ("ERROR: tool arguments are not valid JSON"
                                  + (" (your output hit the token limit; write large files in smaller "
                                     "pieces)" if choice.finish_reason == "length" else ""))
                    except Exception as e:
                        output = f"ERROR: {type(e).__name__}: {e}"
                    self._log(type="tool", name=name, args=tc["function"]["arguments"][:4000],
                              output=output[:8000], dt=round(time.time() - t_tool, 1))
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})

                hook = self._hook("post-tool-use")
                note = ((hook or {}).get("hookSpecificOutput") or {}).get("additionalContext")
                if note:
                    messages[-1]["content"] += "\n\n" + note
                    self._log(type="budget_note", note=note)

                if (usage.get("prompt_tokens") or 0) > self.context_limit:
                    self._elide_old(messages)
        except Exception as e:
            end_reason = f"error: {type(e).__name__}: {e}"[:500]
            self._log(type="fatal", error=end_reason)
        finally:
            with open(os.path.join(self.task_out, "messages.json"), "w") as f:
                json.dump(messages, f, indent=1, default=str)
            self.transcript.close()
        return dict(self.stats, end_reason=end_reason, wall_s=round(time.time() - self.t0, 1))
