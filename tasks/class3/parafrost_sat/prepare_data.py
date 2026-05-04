#!/usr/bin/env python3
"""Download + stage the parafrost_sat Zenodo j2021 subset (10 instances).

The benchmark uses 10 real-world SAT formulas curated from the Zenodo record
5138008 (Osama/Wijs/Biere "Certified SAT solving with GPU accelerated
inprocessing", FMSD'24). The exact 10 instances are pinned in this script so
the release does not need to commit dataset files.

Behaviour:
  - Uses the embedded subset manifest to learn which 10 .cnf.xz files are needed.
  - If `<task_dir>/data/<name>.cnf` already exists for all 10, no-op.
  - Otherwise, downloads `benchmarks-j2021.zip` from Zenodo (verified by MD5)
    into a cache dir (default `<task_dir>/_zenodo_cache/`, kept inside the
    task directory so no external mounts are required), then extracts +
    LZMA-decompresses just the 10 files into `<task_dir>/data/<name>.cnf`.
  - The cached zip is ~7 GB. Override the cache location with `--data-root
    <DIR>` or `PARAFROST_DATA_ROOT=<DIR>` to point at an existing zip on
    a larger disk; or pass `--no-cache` to delete the zip after extraction.

Idempotent. Safe to invoke as `python prepare_data.py` directly or via
`bash prepare_data.sh`.
"""

from __future__ import annotations

import argparse
import hashlib
import lzma
import os
import urllib.request
import zipfile

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")

ZENODO_URL = "https://zenodo.org/records/5138008/files/benchmarks-j2021.zip"
ZIP_MD5 = "4d0aed943bede82a4b44aed94eb2a8e0"
ZIP_SIZE_GB = 7  # approximate; for the user-facing message only

SUBSET_MANIFEST = {
    "version": 1,
    "format": "dimacs-cnf",
    "source": "Zenodo 5138008 - Large SAT Benchmark Suite (Osama/Wijs/Biere FMSD'24)",
    "timeout_sec_per_instance": 1800,
    "instances": [
        {
            "name": "ps_300_311_20",
            "compressed_size": 6660380,
            "reference_verdict": "SAT",
            "reference_time_sec": 10.784,
            "tier": "small",
        },
        {
            "name": "sokoban-p01.sas.ex.17",
            "compressed_size": 1566924,
            "reference_verdict": "UNSAT",
            "reference_time_sec": 5.9190000000000005,
            "tier": "small",
        },
        {
            "name": "manol-pipe-c7nidw",
            "compressed_size": 1286580,
            "reference_verdict": "UNSAT",
            "reference_time_sec": 25.411,
            "tier": "small",
        },
        {
            "name": "ps_200_301_70",
            "compressed_size": 4292048,
            "reference_verdict": "UNSAT",
            "reference_time_sec": 25.76,
            "tier": "small",
        },
        {
            "name": "abw-N-bcsstk07.mtx-w44",
            "compressed_size": 6613536,
            "reference_verdict": "UNSAT",
            "reference_time_sec": 46.160000000000004,
            "tier": "mid_low",
        },
        {
            "name": "Mickey_out250_known_last146_0",
            "compressed_size": 3229096,
            "reference_verdict": "SAT",
            "reference_time_sec": 49.393,
            "tier": "mid_low",
        },
        {
            "name": "gaussian.c.75.smt2-cvc4",
            "compressed_size": 22294104,
            "reference_verdict": "SAT",
            "reference_time_sec": 124.56299999999999,
            "tier": "medium",
        },
        {
            "name": "Mickey_out250_known_last147_0_u",
            "compressed_size": 3229872,
            "reference_verdict": "UNSAT",
            "reference_time_sec": 101.08,
            "tier": "medium",
        },
        {
            "name": "sv-comp19_prop-reachsafety.barrier_3t_true-unreach-call.i-witness",
            "compressed_size": 350799180,
            "reference_verdict": "SAT",
            "reference_time_sec": 1085.2330000000002,
            "tier": "large",
        },
        {
            "name": "T124.2.1",
            "compressed_size": 27979560,
            "reference_verdict": "UNSAT",
            "reference_time_sec": 573.736,
            "tier": "large",
        },
    ],
    "notes": "Reproducible subset of 10 instances from Zenodo record 5138008. Run prepare_data.py to download and extract.",
}


def _md5_of(path: str, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _md5_ok(path: str, expected: str) -> bool:
    if not os.path.isfile(path):
        return False
    return _md5_of(path) == expected


def _download(url: str, dst: str) -> None:
    """Stream-download with simple progress reporting."""
    tmp = dst + ".part"
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", "0"))
        done = 0
        last_pct = -1
        with open(tmp, "wb") as f:
            while True:
                buf = r.read(4 * 1024 * 1024)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total > 0:
                    pct = int(100 * done / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(
                            f"  downloading: {pct:>3d}%  "
                            f"({done / (1 << 30):.2f} / {total / (1 << 30):.2f} GB)",
                            flush=True,
                        )
                        last_pct = pct
    os.replace(tmp, dst)


def _decompress_xz_member(zf: zipfile.ZipFile, member: str, out_path: str) -> None:
    """Read .cnf.xz member from zip, LZMA-decompress to out_path (streamed)."""
    dec = lzma.LZMADecompressor()
    tmp = out_path + ".part"
    with zf.open(member) as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(4 * 1024 * 1024)
            if not chunk:
                # flush any tail bytes
                tail = dec.flush() if hasattr(dec, "flush") else b""
                if tail:
                    dst.write(tail)
                break
            out = dec.decompress(chunk)
            if out:
                dst.write(out)
    os.replace(tmp, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download parafrost_sat Zenodo subset.")
    ap.add_argument(
        "--data-root",
        default=os.environ.get("PARAFROST_DATA_ROOT", os.path.join(TASK_DIR, "_zenodo_cache")),
        help="Where to cache the downloaded Zenodo zip. "
             "Default: <task_dir>/_zenodo_cache (no external mount required). "
             "Override via $PARAFROST_DATA_ROOT or this flag if you want to "
             "share an existing cache on a larger disk.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-extract even if data/<name>.cnf already exists",
    )
    ap.add_argument(
        "--no-cache", action="store_true",
        help="Delete the Zenodo zip after extraction (saves ~7 GB; subsequent "
             "runs would need to redownload).",
    )
    args = ap.parse_args()

    manifest = SUBSET_MANIFEST
    instances = manifest.get("instances", [])
    if len(instances) != 10:
        print(
            f"[prepare_data] manifest has {len(instances)} instances, expected 10",
            file=sys.stderr,
        )
        return 1

    os.makedirs(DATA_DIR, exist_ok=True)
    needed: list[tuple[str, str]] = []  # (xz_member_basename, out_cnf_path)
    for inst in instances:
        name = inst["name"]
        cnf_out = os.path.join(DATA_DIR, name + ".cnf")
        if not args.force and os.path.isfile(cnf_out) and os.path.getsize(cnf_out) > 0:
            continue
        needed.append((name + ".cnf.xz", cnf_out))

    if not needed:
        print(f"[prepare_data] all 10 instances already staged in {DATA_DIR}")
        return 0

    print(f"[prepare_data] need to stage {len(needed)} instance(s)")

    # Download zip if not cached / md5 mismatch
    os.makedirs(args.data_root, exist_ok=True)
    zip_path = os.path.join(args.data_root, "benchmarks-j2021.zip")
    if not _md5_ok(zip_path, ZIP_MD5):
        if os.path.isfile(zip_path):
            print(f"[prepare_data] cached zip md5 mismatch, redownloading -> {zip_path}")
            os.remove(zip_path)
        else:
            print(
                f"[prepare_data] downloading Zenodo 5138008 (~{ZIP_SIZE_GB} GB) "
                f"-> {zip_path}"
            )
        _download(ZENODO_URL, zip_path)
        if not _md5_ok(zip_path, ZIP_MD5):
            print(
                f"[prepare_data] FATAL: md5 mismatch after download "
                f"(expected {ZIP_MD5}, got {_md5_of(zip_path)})",
                file=sys.stderr,
            )
            return 2
        print("[prepare_data] zip md5 OK")
    else:
        print(f"[prepare_data] using cached {zip_path}")

    # Build basename → full member name map (Zenodo zip flattens .cnf.xz at root)
    print(f"[prepare_data] indexing zip…")
    with zipfile.ZipFile(zip_path) as zf:
        members_by_basename = {os.path.basename(n): n for n in zf.namelist()}
        for xz_name, cnf_out in needed:
            member = members_by_basename.get(xz_name)
            if member is None:
                print(
                    f"[prepare_data] FATAL: {xz_name} not present in zip "
                    f"(expected member exists in benchmarks-j2021.zip; "
                    f"check subset manifest or zip integrity)",
                    file=sys.stderr,
                )
                return 3
            print(f"  staging {os.path.basename(cnf_out)}")
            _decompress_xz_member(zf, member, cnf_out)

    if args.no_cache:
        try:
            os.remove(zip_path)
            print(f"[prepare_data] removed cached zip ({zip_path})")
        except OSError as e:
            print(f"[prepare_data] could not remove cached zip: {e}", file=sys.stderr)

    print(f"[prepare_data] done — 10 instances ready under {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
