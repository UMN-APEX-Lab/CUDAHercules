#!/usr/bin/env python3
"""Download the test systems of class2/general/sparse_complex_symmetric_solve.

Two frequency-domain FEM discretizations of 3D electromagnetic problems (7.0M and 18.7M unknowns), two
frequencies each, stored as zstd-compressed COO files in the task-data repository:

    sparse_solver/<system>/<frequency>/{A.bin.zst,B.bin.zst}

    python scripts/prepare_solver_data.py [--base-url URL | --repo HF_DATASET] [--out DIR] [--dev]

They are unpacked to <out>/<system>/<frequency>/{A.bin,B.bin}; the default --out is
$HERCULES_DATA_DIR/sparse_solver (else <repo>/data/sparse_solver, which the evaluator uses when
HERCULES_DATA_DIR is unset). --dev fetches only the 7.0M-unknown system. Files are fetched over HTTP
from --base-url (default: the data repository's anonymous mirror), or with huggingface_hub from --repo.
Needs the zstd command-line tool.
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

DEFAULT_BASE_URL = "https://anonymous-hf.com/api/a/cs6btyea9aef"   # <base>/resolve/<path>

CASES = [("em_7M", "1000000.000000"), ("em_7M", "1000000000.000000"),
         ("em_18M", "5000000.000000"), ("em_18M", "142000000.000000")]


def main():
    default_root = os.environ.get("HERCULES_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("HERCULES_DATA_URL", DEFAULT_BASE_URL),
                    help="files are fetched from <base-url>/resolve/<path>")
    ap.add_argument("--repo", default=None, help="fetch from this Hugging Face dataset repository instead")
    ap.add_argument("--out", default=os.path.join(default_root, "sparse_solver"))
    ap.add_argument("--dev", action="store_true", help="only the 7.0M-unknown system")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.dev or c[0] == "em_7M"]
    cache = None
    if args.repo:
        from huggingface_hub import snapshot_download
        patterns = [f"sparse_solver/{s}/{f}/*.zst" for s, f in cases]
        cache = snapshot_download(repo_id=args.repo, repo_type="dataset", allow_patterns=patterns)
    for s, f in cases:
        dst = os.path.join(args.out, s, f)
        os.makedirs(dst, exist_ok=True)
        for name in ("A.bin", "B.bin"):
            out = os.path.join(dst, name)
            if os.path.exists(out):
                continue
            rel = f"sparse_solver/{s}/{f}/{name}.zst"
            if cache:
                src = os.path.join(cache, rel)
            else:
                src = out + ".zst.part"
                req = urllib.request.Request(f"{args.base_url}/resolve/{rel}", headers={"User-Agent": "curl/8.5.0"})
                with urllib.request.urlopen(req) as r, open(src, "wb") as w:
                    shutil.copyfileobj(r, w, 1 << 22)
            subprocess.run(["zstd", "-d", "-q", "-f", src, "-o", out + ".part"], check=True)
            os.replace(out + ".part", out)
            if not cache:
                os.remove(src)
        print(f"{s}/{f}: ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
