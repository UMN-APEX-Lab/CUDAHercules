#!/usr/bin/env python3
"""Download and convert the test matrices of the CSR sparse tasks (class1/general/90_csr_spmv,
91_csr_spmm, 92_csr_spgemm).

Twelve public matrices: ten from the SuiteSparse Matrix Collection, the Reddit graph (DGL) and the
ogbn-proteins graph (OGB). Each is written as <out>/<name>.bin:

    int32 rows, int32 cols, int32 nnz, int32 row_ptr[rows + 1], int32 col_ind[nnz], float32 values[nnz]

with sorted column indices. Symmetric Matrix Market storage is expanded to both triangles;
pattern-only matrices and the two graphs get the value 1.0.

    python scripts/prepare_sparse_data.py [--out DIR] [--download-dir DIR] [--only NAME ...]

The harnesses read $HERCULES_DATA_DIR/sparse_matrices, the default --out ($HERCULES_DATA_DIR, else
<repo>/data, which the evaluator uses when HERCULES_DATA_DIR is unset). Needs numpy, scipy, pandas.
"""
import argparse
import io
import os
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np
import scipy.io
import scipy.sparse

SUITESPARSE_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
REDDIT_URL = "https://data.dgl.ai/dataset/reddit.zip"
PROTEINS_URL = "https://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip"

# (source, group, name, rows, nnz)
MATRICES = [
    ("suitesparse", "Schenk_AFE", "af_shell10", 1_508_065, 52_672_325),
    ("suitesparse", "Janna", "CoupCons3D", 416_800, 22_322_336),
    ("suitesparse", "DIMACS10", "delaunay_n24", 16_777_216, 100_663_202),
    ("suitesparse", "DIMACS10", "asia_osm", 11_950_757, 25_423_206),
    ("suitesparse", "Belcastro", "human_gene1", 22_283, 24_669_643),
    ("suitesparse", "Belcastro", "mouse_gene", 45_101, 28_967_291),
    ("suitesparse", "Gupta", "gupta3", 16_783, 9_323_427),
    ("suitesparse", "LAW", "in-2004", 1_382_908, 16_917_053),
    ("ogb", "OGB", "ogbn-proteins", 132_534, 79_122_504),
    ("dgl", "DGL", "reddit", 232_965, 114_615_892),
    ("suitesparse", "QY", "case39", 40_216, 1_042_160),
    ("suitesparse", "TSOPF", "TSOPF_FS_b300_c2", 56_814, 8_767_466),
]


def download(url, dest):
    if os.path.exists(dest):
        return
    print(f"  downloading {url}", flush=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def load_suitesparse(ddir, group, name):
    archive = os.path.join(ddir, f"{name}.tar.gz")
    download(SUITESPARSE_URL.format(group=group, name=name), archive)
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(f"/{name}.mtx"))
        coo = scipy.io.mmread(tar.extractfile(member))  # symmetric -> both triangles, pattern -> 1.0
    return coo.tocsr()


def load_reddit(ddir):
    archive = os.path.join(ddir, "reddit.zip")
    download(REDDIT_URL, archive)
    with zipfile.ZipFile(archive) as z, z.open("reddit_graph.npz") as f:
        graph = scipy.sparse.load_npz(io.BytesIO(f.read())).tocsr()
    graph.data = np.ones_like(graph.data, dtype=np.float32)
    return graph


def load_ogbn_proteins(ddir):
    import pandas as pd
    archive = os.path.join(ddir, "proteins.zip")
    download(PROTEINS_URL, archive)
    with zipfile.ZipFile(archive) as z:
        n = int(pd.read_csv(z.open("proteins/raw/num-node-list.csv.gz"), compression="gzip",
                            header=None).iloc[0, 0])
        edges = pd.read_csv(z.open("proteins/raw/edge.csv.gz"), compression="gzip", header=None).to_numpy()
    adj = scipy.sparse.coo_matrix((np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
                                  shape=(n, n)).tocsr()
    adj.data[:] = 1.0                   # duplicate edges -> unit adjacency
    return adj.maximum(adj.T).tocsr()   # undirected


def write_bin(csr, path):
    csr.sum_duplicates()
    csr.sort_indices()
    rows, cols = csr.shape
    with open(path + ".part", "wb") as f:
        np.array([rows, cols, csr.nnz], dtype=np.int32).tofile(f)
        csr.indptr.astype(np.int32).tofile(f)
        csr.indices.astype(np.int32).tofile(f)
        csr.data.astype(np.float32).tofile(f)
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.environ.get("HERCULES_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ap.add_argument("--out", default=os.path.join(default_root, "sparse_matrices"))
    ap.add_argument("--download-dir", default=None, help="archive cache (default: <out>/downloads)")
    ap.add_argument("--only", nargs="*", help="subset of matrix names")
    args = ap.parse_args()
    ddir = args.download_dir or os.path.join(args.out, "downloads")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(ddir, exist_ok=True)
    failed = []
    for source, group, name, exp_rows, exp_nnz in MATRICES:
        if args.only and name not in args.only:
            continue
        path = os.path.join(args.out, f"{name}.bin")
        if os.path.exists(path):
            print(f"[skip] {name}")
            continue
        print(f"[build] {name}", flush=True)
        try:
            if source == "suitesparse":
                csr = load_suitesparse(ddir, group, name)
            elif source == "dgl":
                csr = load_reddit(ddir)
            else:
                csr = load_ogbn_proteins(ddir)
            if csr.shape[0] != exp_rows or csr.nnz != exp_nnz:
                raise ValueError(f"got {csr.shape[0]} rows / {csr.nnz} nnz, expected {exp_rows} / {exp_nnz}")
            write_bin(csr, path)
            print(f"  {csr.shape[0]:,} x {csr.shape[1]:,}, nnz {csr.nnz:,}")
        except Exception as exc:  # report and continue with the other matrices
            print(f"  FAILED: {exc!r}")
            failed.append(name)
    if failed:
        print(f"failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
