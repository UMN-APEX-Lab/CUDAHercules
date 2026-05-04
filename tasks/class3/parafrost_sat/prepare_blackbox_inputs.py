#!/usr/bin/env python3
"""Prepare benchmark-owned standardized inputs for the parafrost_sat black-box task.

Staging pipeline:
  1. Read the curated subset manifest embedded in `prepare_data.py`
     (10 instances pinned from Zenodo 5138008, SAT Competition 2013-2021).
  2. Ensure each `data/<name>.cnf` exists; auto-runs `prepare_data.py` to
     download + extract from Zenodo zip if missing.
  3. Symlink (or copy) `data/<name>.cnf` -> `.blackbox_inputs_v1/instances/<name>.cnf`.
  4. Emit `.blackbox_inputs_v1/instances.json` in the format the eval harness
     expects (version, format, instances[]).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from prepare_data import SUBSET_MANIFEST

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
PREPARE_DATA = os.path.join(TASK_DIR, "prepare_data.py")
OUT_DIR = os.path.join(TASK_DIR, ".blackbox_inputs_v1")
INSTANCE_DIR = os.path.join(OUT_DIR, "instances")


def _ensure_data() -> None:
    """If any subset CNF is missing under data/, invoke prepare_data.py."""
    subset = SUBSET_MANIFEST
    missing = [
        inst["name"] + ".cnf"
        for inst in subset.get("instances", [])
        if not os.path.isfile(os.path.join(DATA_DIR, inst["name"] + ".cnf"))
    ]
    if not missing:
        return
    print(
        f"[prepare_blackbox_inputs] {len(missing)} subset CNFs missing from "
        f"{DATA_DIR}; invoking {PREPARE_DATA}",
        file=sys.stderr,
    )
    subprocess.check_call([sys.executable, PREPARE_DATA])


def _stage_cnf(src_cnf: str, dst_cnf: str) -> None:
    """Symlink src_cnf -> dst_cnf if possible, else copy. Idempotent."""
    if os.path.islink(dst_cnf) or os.path.isfile(dst_cnf):
        return
    try:
        os.symlink(os.path.abspath(src_cnf), dst_cnf)
    except OSError:
        shutil.copy2(src_cnf, dst_cnf)


def main() -> int:
    _ensure_data()

    subset = SUBSET_MANIFEST

    os.makedirs(INSTANCE_DIR, exist_ok=True)

    manifest = {
        "version": 1,
        "format": "dimacs-cnf",
        "source": subset.get("source", "Zenodo 5138008 subset"),
        "timeout_sec_per_instance": subset.get("timeout_sec_per_instance", 1800),
        "instances": [],
    }

    for inst in subset["instances"]:
        name = inst["name"]
        src_cnf = os.path.join(DATA_DIR, name + ".cnf")
        if not os.path.isfile(src_cnf):
            raise FileNotFoundError(
                f"missing CNF for {name}: {src_cnf} "
                f"(prepare_data.py should have staged it)"
            )
        dst_cnf = os.path.join(INSTANCE_DIR, name + ".cnf")
        _stage_cnf(src_cnf, dst_cnf)
        manifest["instances"].append(
            {
                "name": name,
                "file": os.path.join("instances", name + ".cnf"),
                "description": f"{inst.get('tier','?')} tier ({inst.get('reference_verdict','?')}, "
                               f"{inst.get('reference_time_sec',0):.1f}s ParaFROST reference)",
                "reference_verdict": inst.get("reference_verdict", ""),
                "reference_time_sec": inst.get("reference_time_sec", -1),
                "tier": inst.get("tier", ""),
            }
        )

    manifest_path = os.path.join(OUT_DIR, "instances.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
