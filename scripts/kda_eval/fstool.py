#!/usr/bin/python3
"""File tools executed INSIDE the sandbox (read / write / edit).

The agent loop runs outside the sandbox; doing file I/O in here means paths
(including symlinks the agent created) resolve against the sandbox view, so
nothing outside the sandbox can be read or written through these tools.

Request: JSON on stdin {"op": "read"|"write"|"edit", ...}; reply: JSON on stdout.
stdlib only.
"""

import json
import os
import sys

WORKSPACE = "/workspace"
MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000


def _path(p: str) -> str:
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(WORKSPACE, p))


def op_read(req: dict) -> dict:
    path = _path(req["path"])
    offset = max(int(req.get("offset") or 1), 1)
    limit = min(int(req.get("limit") or MAX_READ_LINES), MAX_READ_LINES)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    chunk = lines[offset - 1:offset - 1 + limit]
    out = "".join(f"{offset + i:6d}\t{l[:MAX_LINE_CHARS].rstrip(chr(10))}\n" for i, l in enumerate(chunk))
    more = len(lines) - (offset - 1 + len(chunk))
    if more > 0:
        out += f"... ({more} more lines; use offset={offset + len(chunk)})\n"
    return {"ok": True, "output": out or "(empty file)\n"}


def op_write(req: dict) -> dict:
    path = _path(req["path"])
    if not path.startswith(WORKSPACE + "/"):
        return {"ok": False, "output": f"refusing to write outside {WORKSPACE}: {path}"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(req["content"])
    return {"ok": True, "output": f"wrote {len(req['content'])} chars to {path}"}


def op_edit(req: dict) -> dict:
    path = _path(req["path"])
    if not path.startswith(WORKSPACE + "/"):
        return {"ok": False, "output": f"refusing to edit outside {WORKSPACE}: {path}"}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    old, new = req["old_string"], req["new_string"]
    n = text.count(old)
    if n == 0:
        return {"ok": False, "output": "old_string not found"}
    if n > 1 and not req.get("replace_all"):
        return {"ok": False, "output": f"old_string occurs {n} times; make it unique or set replace_all"}
    text = text.replace(old, new) if req.get("replace_all") else text.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"ok": True, "output": f"replaced {n if req.get('replace_all') else 1} occurrence(s) in {path}"}


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        resp = {"read": op_read, "write": op_write, "edit": op_edit}[req["op"]](req)
    except Exception as e:
        resp = {"ok": False, "output": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(resp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
