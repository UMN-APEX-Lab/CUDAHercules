#!/usr/bin/env python3
"""Download the test systems of class2/general/sparse_complex_symmetric_solve.

Two frequency-domain FEM discretizations of 3D electromagnetic problems (7.0M and 18.7M unknowns), two
frequencies each, stored as zstd-compressed COO files in a Hugging Face dataset repository:

    sparse_solver/<system>/<frequency>/{A.bin.zst,B.bin.zst}

    python scripts/prepare_solver_data.py --repo <hf-dataset-repo> [--out DIR] [--dev]

They are unpacked to <out>/<system>/<frequency>/{A.bin,B.bin}; the default --out is
$HERCULES_DATA_DIR/sparse_solver (else <repo>/data/sparse_solver, which the evaluator uses when
HERCULES_DATA_DIR is unset). --dev fetches only the 7.0M-unknown system. Needs huggingface_hub and the
zstd command-line tool.
"""
import argparse
import os
import subprocess
import sys

CASES = [("em_7M", "1000000.000000"), ("em_7M", "1000000000.000000"),
         ("em_18M", "5000000.000000"), ("em_18M", "142000000.000000")]


def main():
    default_root = os.environ.get("HERCULES_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("HERCULES_HF_DATASET"), help="Hugging Face dataset repository")
    ap.add_argument("--out", default=os.path.join(default_root, "sparse_solver"))
    ap.add_argument("--dev", action="store_true", help="only the 7.0M-unknown system")
    args = ap.parse_args()
    if not args.repo:
        sys.exit("pass --repo (or set HERCULES_HF_DATASET)")
    from huggingface_hub import snapshot_download

    cases = [c for c in CASES if not args.dev or c[0] == "em_7M"]
    patterns = [f"sparse_solver/{s}/{f}/*.zst" for s, f in cases]
    cache = snapshot_download(repo_id=args.repo, repo_type="dataset", allow_patterns=patterns)
    for s, f in cases:
        dst = os.path.join(args.out, s, f)
        os.makedirs(dst, exist_ok=True)
        for name in ("A.bin", "B.bin"):
            src = os.path.join(cache, "sparse_solver", s, f, name + ".zst")
            out = os.path.join(dst, name)
            if os.path.exists(out):
                continue
            subprocess.run(["zstd", "-d", "-q", "-f", src, "-o", out + ".part"], check=True)
            os.replace(out + ".part", out)
        print(f"{s}/{f}: ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
