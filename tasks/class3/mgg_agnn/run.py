#!/usr/bin/env python3
"""
MGG AGNN Training -- CUDA-Hercules Class 3 harness

Builds two PyTorch CUDA extensions:
  1. Reference: MGG's warp-per-node AGNN kernel (OSDI'23)
  2. Solution: LLM's optimized AGNN kernel

Trains a 4-layer AGNN on the amazon0505 graph (TC-GNN paper Type III).
Correctness: solution training converges and achieves similar loss to reference.
Performance: average time per training epoch.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from scipy.sparse import coo_matrix

# -- Configuration -----------------------------------------------------------

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(TASK_DIR, '..', 'tcgnn_gcn', 'data', 'amazon0505.npz')

# amazon0505: 410,236 nodes, 4,878,875 edges (TC-GNN paper Table 4, Type III)
INPUT_DIM = 96     # per paper Table 4
HIDDEN_DIM = 32    # AGNN uses smaller hidden (from MGG paper)
NUM_CLASSES = 22   # per paper Table 4
NUM_LAYERS = 4

WARMUP_EPOCHS = 10
BENCHMARK_EPOCHS = int(os.environ.get('KH_BENCHMARK', '0')) or 200
CORRECTNESS_EPOCHS = 50   # epochs for convergence check

# -- Graph Loading -----------------------------------------------------------

def load_graph_csr(path):
    """Load graph from .npz file (TC-GNN format) and build CSR."""
    graph_obj = np.load(path)
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_nodes = int(graph_obj['num_nodes'])
    num_edges = len(src_li)

    # Build CSR via scipy
    val = np.ones(num_edges, dtype=np.int32)
    scipy_coo = coo_matrix((val, (src_li, dst_li)), shape=(num_nodes, num_nodes))
    scipy_csr = scipy_coo.tocsr()

    row_pointers = np.array(scipy_csr.indptr, dtype=np.int32)
    column_index = np.array(scipy_csr.indices, dtype=np.int32)

    return num_nodes, row_pointers, column_index

# -- Autograd Function for full AGNN layer -----------------------------------

def make_agnn_function(module):
    """Create autograd Function that captures the full AGNN layer module."""

    class AGNNFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, weights, row_pointers, column_index, num_nodes, num_edges, apply_relu):
            result = module.forward(
                X,
                weights,
                row_pointers,
                column_index,
                num_nodes,
                num_edges,
                apply_relu,
            )
            output = result[0]
            edge_attention = result[1]
            ctx.save_for_backward(X, weights, edge_attention, output, row_pointers, column_index)
            ctx.num_nodes = num_nodes
            ctx.num_edges = num_edges
            ctx.apply_relu = apply_relu
            return output

        @staticmethod
        def backward(ctx, d_output):
            X, weights, edge_attention, output, row_pointers, column_index = ctx.saved_tensors
            d_input, d_weights = module.backward(
                d_output,
                X,
                weights,
                edge_attention,
                output,
                row_pointers,
                column_index,
                ctx.num_nodes,
                ctx.num_edges,
                ctx.apply_relu,
            )
            return d_input, d_weights, None, None, None, None, None

    return AGNNFunction

# -- AGNN Model --------------------------------------------------------------

class AGNN(torch.nn.Module):
    """Multi-layer AGNN: each layer is implemented by the extension module."""

    def __init__(self, agnn_fn, graph_info, input_dim, hidden_dim, num_classes, num_layers=4):
        super().__init__()
        self.agnn_fn = agnn_fn
        self.graph_info = graph_info   # (row_ptr, col_idx, num_nodes, num_edges)

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(dims[i], dims[i + 1]) *
                               (2.0 / (dims[i] + dims[i + 1])) ** 0.5)
            for i in range(num_layers)
        ])

    def forward(self, x):
        row_ptr, col_idx, num_nodes, num_edges = self.graph_info
        for i, w in enumerate(self.weights):
            apply_relu = i < len(self.weights) - 1
            x = self.agnn_fn.apply(
                x,
                w,
                row_ptr,
                col_idx,
                num_nodes,
                num_edges,
                apply_relu,
            )
        return x

# -- Training loop -----------------------------------------------------------

def train_epochs(model, x, labels, num_epochs, optimizer):
    """Train for num_epochs, return list of loss values."""
    losses = []
    for _ in range(num_epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(F.log_softmax(out, dim=1), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    return losses

def benchmark_training(model, x, labels, warmup_epochs, timed_epochs):
    """Warmup then time timed_epochs, return per-epoch times in ms."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(warmup_epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(F.log_softmax(out, dim=1), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed epochs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed_epochs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed_epochs)]

    for i in range(timed_epochs):
        start_events[i].record()
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(F.log_softmax(out, dim=1), labels)
        loss.backward()
        optimizer.step()
        end_events[i].record()

    torch.cuda.synchronize()
    return [start_events[i].elapsed_time(end_events[i]) for i in range(timed_epochs)]


def all_finite(values):
    """Return True iff every numeric value is finite."""
    return all(math.isfinite(v) for v in values)

# -- Main --------------------------------------------------------------------

def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_cpp = os.path.join(task_dir, 'wrapper.cpp')
    ref_cu = os.path.join(task_dir, 'ref_kernel.cu')
    sol_cu = os.path.join(task_dir, 'solution.cu')

    cuda_flags = ['-O3', '--use_fast_math', '--expt-relaxed-constexpr']

    print("Building reference extension...", flush=True)
    ref_mod = load(name='agnn_ref', sources=[wrapper_cpp, ref_cu],
                   extra_cuda_cflags=cuda_flags, verbose=False)

    print("Building solution extension...", flush=True)
    sol_mod = load(name='agnn_sol', sources=[wrapper_cpp, sol_cu],
                   extra_cuda_cflags=cuda_flags, verbose=False)

    # Load real graph dataset (amazon0505, TC-GNN paper Type III)
    print(f"Loading graph: {GRAPH_FILE}...", flush=True)
    num_nodes, row_pointers, column_index = load_graph_csr(GRAPH_FILE)
    num_edges = len(column_index)
    print(f"Nodes: {num_nodes}, Edges: {num_edges}")

    # Move graph data to GPU
    row_ptr_gpu = torch.from_numpy(row_pointers).int().cuda()
    col_idx_gpu = torch.from_numpy(column_index).int().cuda()
    graph_info = (row_ptr_gpu, col_idx_gpu, num_nodes, num_edges)

    # Create autograd functions
    ref_agnn_fn = make_agnn_function(ref_mod)
    sol_agnn_fn = make_agnn_function(sol_mod)

    # Features and labels (fixed seed)
    torch.manual_seed(42)
    x = torch.randn(num_nodes, INPUT_DIM, device='cuda')
    labels = torch.randint(0, NUM_CLASSES, (num_nodes,), device='cuda')

    all_passed = True

    print(f"\n=== AGNN Training: N={num_nodes}, E={num_edges}, D={INPUT_DIM} ===")

    # -- Correctness: both ref and sol should converge --
    torch.manual_seed(123)
    ref_model_c = AGNN(ref_agnn_fn, graph_info, INPUT_DIM, HIDDEN_DIM,
                       NUM_CLASSES, NUM_LAYERS).cuda()
    sol_model_c = AGNN(sol_agnn_fn, graph_info, INPUT_DIM, HIDDEN_DIM,
                       NUM_CLASSES, NUM_LAYERS).cuda()
    sol_model_c.load_state_dict(ref_model_c.state_dict())

    ref_opt_c = torch.optim.Adam(ref_model_c.parameters(), lr=0.01)
    sol_opt_c = torch.optim.Adam(sol_model_c.parameters(), lr=0.01)

    ref_losses = train_epochs(ref_model_c, x, labels, CORRECTNESS_EPOCHS, ref_opt_c)
    sol_losses = train_epochs(sol_model_c, x, labels, CORRECTNESS_EPOCHS, sol_opt_c)

    ref_first, ref_last = ref_losses[0], ref_losses[-1]
    sol_first, sol_last = sol_losses[0], sol_losses[-1]

    # Loss at convergence checkpoints (25%, 50%, 75%, 100%)
    def loss_at_checkpoints(losses):
        n = len(losses)
        pcts = [25, 50, 75, 100]
        return {p: losses[max(0, n * p // 100 - 1)] for p in pcts}

    ref_ckpts = loss_at_checkpoints(ref_losses)
    sol_ckpts = loss_at_checkpoints(sol_losses)

    print(f"Ref loss: {ref_first:.4f} -> {ref_last:.4f}")
    print(f"Solution loss: {sol_first:.4f} -> {sol_last:.4f}")
    if ref_last > 0:
        print(f"Loss ratio: {sol_last / ref_last:.4f} (solution_last / ref_last)")

    print(f"Ref loss checkpoints: " + " | ".join(
        f"{p}%={ref_ckpts[p]:.4f}" for p in [25, 50, 75, 100]))
    print(f"Solution loss checkpoints: " + " | ".join(
        f"{p}%={sol_ckpts[p]:.4f}" for p in [25, 50, 75, 100]))

    if not all_finite(ref_losses):
        print("FAILED (reference produced non-finite loss)")
        all_passed = False

    if not all_finite(sol_losses):
        print("FAILED (solution produced non-finite loss)")
        all_passed = False

    # Check 1: solution loss decreased (model is learning)
    if all_finite([sol_first, sol_last]) and sol_last >= sol_first:
        print("FAILED (solution did not converge)")
        all_passed = False

    # Check 2: solution final loss is within 3x of reference final loss
    if all_finite([ref_last, sol_last]) and ref_last > 0 and sol_last / ref_last > 3.0:
        print(f"FAILED (solution loss {sol_last:.4f} >> ref loss {ref_last:.4f})")
        all_passed = False

    if all_passed:
        print("Passed")

    del ref_model_c, sol_model_c, ref_opt_c, sol_opt_c
    torch.cuda.empty_cache()

    # -- Benchmark training --
    if BENCHMARK_EPOCHS > 0 and all_passed:
        torch.manual_seed(456)
        ref_model_b = AGNN(ref_agnn_fn, graph_info, INPUT_DIM, HIDDEN_DIM,
                           NUM_CLASSES, NUM_LAYERS).cuda()
        sol_model_b = AGNN(sol_agnn_fn, graph_info, INPUT_DIM, HIDDEN_DIM,
                           NUM_CLASSES, NUM_LAYERS).cuda()
        sol_model_b.load_state_dict(ref_model_b.state_dict())

        ref_times = benchmark_training(ref_model_b, x, labels, WARMUP_EPOCHS, BENCHMARK_EPOCHS)
        sol_times = benchmark_training(sol_model_b, x, labels, WARMUP_EPOCHS, BENCHMARK_EPOCHS)

        ref_avg = sum(ref_times) / len(ref_times)
        ref_min = min(ref_times)
        sol_avg = sum(sol_times) / len(sol_times)
        sol_min = min(sol_times)

        print(f"Ref time: {ref_avg:.4f} ms (avg over {BENCHMARK_EPOCHS} epochs, min: {ref_min:.4f} ms)")
        print(f"Kernel time: {sol_avg:.4f} ms (avg over {BENCHMARK_EPOCHS} epochs, min: {sol_min:.4f} ms)")

        if sol_min > 0:
            speedup = ref_min / sol_min
            print(f"Speedup: {speedup:.4f}x (ref_min / kernel_min)")

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
