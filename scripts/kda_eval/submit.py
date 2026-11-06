#!/usr/bin/python3
"""submit — send a candidate to the task judge (runs inside the sandbox).

Usage:
  submit [FILE]           compile + correctness + timing vs. the reference
  submit --check [FILE]   compile + correctness only (no timing)
  submit --status         evaluations used / remaining, time left, best result
  submit --budget-json    machine-readable budget status (for KDA hooks)

FILE defaults to the task's solution file in /workspace. Any path is
accepted; its contents are evaluated as the task's solution file. Every call
except --status / --budget-json consumes one evaluation from the budget.

stdlib only (runs under the sandbox's system python3).
"""

import argparse
import json
import os
import socket
import sys


def _request(payload: dict, timeout: float = 7200.0) -> dict:
    path = os.environ.get("KH_JUDGE_SOCKET", "/run/kh/judge.sock")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    s.sendall((json.dumps(payload) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode())


def main() -> int:
    ap = argparse.ArgumentParser(prog="submit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--check", action="store_true", help="correctness only, skip timing")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--budget-json", action="store_true")
    args = ap.parse_args()

    if args.status or args.budget_json:
        resp = _request({"op": "status"}, timeout=60)
        if args.budget_json:
            print(json.dumps(resp.get("budget", {})))
        else:
            print(resp.get("text", json.dumps(resp, indent=2)))
        return 0

    sol_name = os.environ.get("KH_SOLUTION_FILE", "")
    path = args.file or os.path.join("/workspace", sol_name)
    if not os.path.isfile(path):
        print(f"submit: no such file: {path}", file=sys.stderr)
        return 2
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    resp = _request({"op": "submit", "mode": "check" if args.check else "full",
                     "source_path": os.path.abspath(path), "content": content})
    print(resp.get("text", json.dumps(resp, indent=2)))
    if resp.get("status") == "PASS":
        return 0
    return 2 if resp.get("rejected") else 1


if __name__ == "__main__":
    sys.exit(main())
