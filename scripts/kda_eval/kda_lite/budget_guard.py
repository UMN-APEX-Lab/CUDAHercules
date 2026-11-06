#!/usr/bin/env python3
"""KDA budget guard: Claude Code hook for unattended, budgeted sessions.

The harness that hosts the session owns and enforces the budget; this hook
only keeps the agent aware of it and keeps the KDA loop going.

  post-tool-use  Inject a short budget note into the agent's context each
                 time a time or evaluation milestone (25/50/75/90%) is
                 crossed.
  stop           Block the agent from ending the session while the
                 promotion criteria are unmet and meaningful budget remains
                 (at most KDA_MAX_STOP_BLOCKS times).

Configuration (environment):
  KDA_BUDGET_STATUS_CMD   command printing a JSON object with
                          elapsed_s, limit_s, evals_used, evals_limit,
                          promoted (bool), summary (str)       [required]
  KDA_BUDGET_STATE        state file   (default: $HOME/.kda_budget_state.json)
  KDA_STOP_MIN_REMAINING  remaining time fraction below which stopping is
                          always allowed                      (default: 0.15)
  KDA_MAX_STOP_BLOCKS     max number of blocked stop attempts (default: 8)
  KDA_STATUS_INTERVAL_S   min seconds between status polls on
                          post-tool-use                       (default: 20)

Fails open: if the status command is missing or errors, the hook does
nothing. stdlib only.
"""

import json
import os
import shlex
import subprocess
import sys
import time

MILESTONES = (0.25, 0.5, 0.75, 0.9)


def _state_path() -> str:
    return os.environ.get("KDA_BUDGET_STATE",
                          os.path.join(os.path.expanduser("~"), ".kda_budget_state.json"))


def _load_state() -> dict:
    try:
        with open(_state_path()) as f:
            return json.load(f)
    except Exception:
        return {"milestone": 0.0, "stop_blocks": 0, "last_poll": 0.0}


def _save_state(state: dict) -> None:
    with open(_state_path(), "w") as f:
        json.dump(state, f)


def _status() -> dict | None:
    cmd = os.environ.get("KDA_BUDGET_STATUS_CMD", "")
    if not cmd:
        return None
    try:
        out = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout)
    except Exception:
        return None


def _fractions(st: dict) -> tuple[float, float]:
    t = st["elapsed_s"] / st["limit_s"] if st.get("limit_s") else 0.0
    e = st["evals_used"] / st["evals_limit"] if st.get("evals_limit") else 0.0
    return t, e


def post_tool_use(state: dict) -> None:
    interval = float(os.environ.get("KDA_STATUS_INTERVAL_S", "20"))
    if time.time() - state.get("last_poll", 0.0) < interval:
        return
    state["last_poll"] = time.time()
    st = _status()
    if not st:
        return
    used = max(_fractions(st))
    crossed = [m for m in MILESTONES if used >= m and m > state.get("milestone", 0.0)]
    if not crossed:
        return
    state["milestone"] = crossed[-1]
    note = f"[KDA budget] {int(crossed[-1] * 100)}% of the budget used. {st.get('summary', '')}"
    if crossed[-1] >= 0.9:
        note += (" Stop exploring: make sure the deliverable holds your best validated "
                 "candidate, update the evidence records, and finish.")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                             "additionalContext": note}}))


def stop(state: dict) -> None:
    st = _status()
    if not st or st.get("promoted"):
        return
    t_frac, _ = _fractions(st)
    min_remaining = float(os.environ.get("KDA_STOP_MIN_REMAINING", "0.15"))
    max_blocks = int(os.environ.get("KDA_MAX_STOP_BLOCKS", "8"))
    evals_left = st.get("evals_limit", 0) - st.get("evals_used", 0)
    if 1.0 - t_frac <= min_remaining or evals_left <= 0 or state.get("stop_blocks", 0) >= max_blocks:
        return
    state["stop_blocks"] = state.get("stop_blocks", 0) + 1
    reason = (
        "The promotion criteria are not met yet and budget remains "
        f"({st.get('summary', '')}). No human will answer: continue the KDA loop on your "
        "own. Review docs/plan.md and candidates.jsonl, pick the most promising next "
        "candidate, implement it, and validate it. Stop only when the promotion "
        "criteria are met or the budget is nearly exhausted."
    )
    print(json.dumps({"decision": "block", "reason": reason}))


def main() -> int:
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        sys.stdin.read()  # hook payload; not needed
    except Exception:
        pass
    state = _load_state()
    if event == "post-tool-use":
        post_tool_use(state)
    elif event == "stop":
        stop(state)
    _save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
