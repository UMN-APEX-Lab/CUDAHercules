#!/usr/bin/env python3
"""Download ogbn-papers100M and convert to MGG binary CSR format."""
import os
import sys
import builtins
import numpy as np

# Auto-confirm OGB download prompt
builtins.input = lambda *a, **k: 'y'

data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ogb_dir = os.path.join(data_dir, "ogb_download")
bin_dir = os.path.join(data_dir, "bin")

beg_file = os.path.join(bin_dir, "paper100M_beg_pos.bin")
csr_file = os.path.join(bin_dir, "paper100M_csr.bin")
weight_file = os.path.join(bin_dir, "paper100M_weight.bin")

if os.path.isfile(beg_file) and os.path.isfile(csr_file):
    print(f"Already converted: {bin_dir}/paper100M_*.bin")
    sys.exit(0)

os.makedirs(bin_dir, exist_ok=True)

# Download
from ogb.nodeproppred import NodePropPredDataset
print("Downloading ogbn-papers100M (56 GB, may take 30+ min)...", flush=True)
dataset = NodePropPredDataset(name="ogbn-papers100M", root=ogb_dir)
graph = dataset[0][0]

edge_index = graph["edge_index"]  # [2, num_edges]
num_nodes = int(graph["num_nodes"])
num_edges = edge_index.shape[1]
print(f"Loaded: {num_nodes} nodes, {num_edges} edges")

# Build CSR
print("Building CSR (sorting edges)...", flush=True)
src = edge_index[0].astype(np.int64)
dst = edge_index[1].astype(np.int64)

order = np.argsort(src, kind='mergesort')
src_sorted = src[order]
dst_sorted = dst[order]

row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
np.add.at(row_ptr, src_sorted + 1, 1)
row_ptr = np.cumsum(row_ptr)
assert row_ptr[num_nodes] == num_edges, f"CSR mismatch: {row_ptr[num_nodes]} != {num_edges}"

# Write binary (int64 = long, matches MGG graph.h template<long, long, ...>)
print(f"Writing {beg_file} ({(num_nodes+1)*8/1e9:.2f} GB)...", flush=True)
row_ptr.tofile(beg_file)

print(f"Writing {csr_file} ({num_edges*8/1e9:.2f} GB)...", flush=True)
dst_sorted.tofile(csr_file)

print(f"Writing {weight_file}...", flush=True)
np.ones(num_edges, dtype=np.int64).tofile(weight_file)

print("Done!")
for f in [beg_file, csr_file, weight_file]:
    print(f"  {os.path.basename(f)}: {os.path.getsize(f)/1e9:.2f} GB")
