#!/usr/bin/env python3
"""Compatibility shim — class3-only wrapper around scripts/hacking_audit.py.

The unified auditor `scripts/hacking_audit.py` handles Class 1 / Class 2
/ Class 3 with class-specific system prompts. This file exists to keep
existing scripts that reference `class3_hacking_audit.py` working.

New code should call `scripts/hacking_audit.py` directly.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

# Delegate to the unified script. Module is named via importlib because the
# unified file has a dash-free name.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "hacking_audit", os.path.join(here, "hacking_audit.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    sys.exit(_mod.main())
