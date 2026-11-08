#!/bin/bash
# Copy tasks/class1/include/kh_verify.h into every task directory that already has a copy
# (or into the directories given as arguments). The runner copies only the task directory,
# so each task carries its own copy; tasks/class1/include/kh_verify.h is the source of truth.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/tasks/class1/include/kh_verify.h"
if [ "$#" -gt 0 ]; then dirs=("$@"); else
  mapfile -t dirs < <(find "$ROOT/tasks" -name kh_verify.h -not -path "$ROOT/tasks/class1/include/*" -exec dirname {} \;)
fi
for d in "${dirs[@]}"; do cp "$SRC" "$d/kh_verify.h"; echo "synced $d"; done
