#!/usr/bin/env python3
"""
TC-GNN GCN Training -- CUDA-Hercules Class 3 harness

Builds two PyTorch CUDA extensions:
  1. Reference: TC-GNN SpMM with WMMA 16x16x8 TF32
  2. Solution: LLM's optimized SpMM

Runs 2-layer GCN training on one or more TC-GNN graphs.
Correctness: solution training converges and tracks reference loss closely.
Performance: average time per training epoch, aggregated across graphs.
"""

import os
import sys
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from scipy.sparse import coo_matrix

# ── Configuration ──────────────────────────────────────────────────────

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_SPECS = {
    "amazon0505": {"input_dim": 96, "num_classes": 22},
    "artist": {"input_dim": 100, "num_classes": 12},
    "soc-BlogCatalog": {"input_dim": 128, "num_classes": 39},
    "amazon0601": {"input_dim": 96, "num_classes": 22},
}
DEFAULT_GRAPH_ORDER = ["amazon0505", "artist", "soc-BlogCatalog", "amazon0601"]

HIDDEN_DIM = 256  # large enough to make SpMM compute-intensive
NUM_LAYERS = 2
BLK_H = 16
BLK_W = 8
LOGIT_PAD_MULTIPLE = 16

WARMUP_EPOCHS = 10
BENCHMARK_EPOCHS = int(os.environ.get('KH_BENCHMARK', '0')) or 200
CORRECTNESS_EPOCHS = 50
MAX_LOSS_RATIO = 1.1
CORRECTNESS_SEEDS = [123, 456, 789]
BENCHMARK_SEED = 2025
TORCH_CUDA_ARCH_LIST = "8.0+PTX"
AMPERE_GENCODE_FLAGS = [
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_80,code=compute_80",
]


def resolve_graph_names():
    raw = os.environ.get("TCGNN_GRAPHS", "").strip()
    if not raw:
        return DEFAULT_GRAPH_ORDER
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in GRAPH_SPECS]
    if unknown:
        raise ValueError(f"Unknown TCGNN_GRAPHS entries: {', '.join(unknown)}")
    return names


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple

# ── Graph Loading ─────────────────────────────────────────────────────

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

# ── TC-GNN Preprocessing ─────────────────────────────────────────────

def preprocess_graph(module, column_index, row_pointers, num_nodes):
    """Call TC-GNN CPU preprocessing to build edgeToRow, edgeToColumn, blockPartition."""
    num_edges = len(column_index)
    num_windows = (num_nodes + BLK_H - 1) // BLK_H

    blockPartition = torch.zeros(num_windows, dtype=torch.int32)
    edgeToColumn = torch.zeros(num_edges, dtype=torch.int32)
    edgeToRow = torch.zeros(num_edges, dtype=torch.int32)

    col_idx_cpu = torch.from_numpy(column_index).int()
    row_ptr_cpu = torch.from_numpy(row_pointers).int()

    module.preprocess(col_idx_cpu, row_ptr_cpu, num_nodes, BLK_H, BLK_W,
                      blockPartition, edgeToColumn, edgeToRow)

    return blockPartition, edgeToColumn, edgeToRow

# ── Autograd Function for GCN layer (GEMM + SpMM) ────────────────────
# Mirrors TC-GNN's TCGNNFunction: forward does GEMM then SpMM,
# backward does SpMM then GEMM (valid for symmetric/undirected graphs).

def make_gcn_function(spmm_fn):
    """Create autograd Function that captures spmm_fn in closure."""

    class GCNFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, weights, row_pointers, column_index,
                    blockPartition, edgeToColumn, edgeToRow):
            ctx.save_for_backward(X, weights, row_pointers, column_index,
                                  blockPartition, edgeToColumn, edgeToRow)
            X_prime = torch.mm(X, weights)
            X_prime = spmm_fn(X_prime, row_pointers, column_index,
                              blockPartition, edgeToColumn, edgeToRow)[0]
            return X_prime

        @staticmethod
        def backward(ctx, d_output):
            (X, weights, row_pointers, column_index,
             blockPartition, edgeToColumn, edgeToRow) = ctx.saved_tensors
            # SpMM backward (A^T @ grad = A @ grad for symmetric A)
            d_input_prime = spmm_fn(d_output, row_pointers, column_index,
                                    blockPartition, edgeToColumn, edgeToRow)[0]
            d_input = torch.mm(d_input_prime, weights.transpose(0, 1))
            d_weights = torch.mm(X.transpose(0, 1), d_input_prime)
            return d_input, d_weights, None, None, None, None, None

    return GCNFunction

# ── GCN Model ────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    """Multi-layer GCN: each layer does GEMM -> SpMM -> ReLU."""

    def __init__(self, gcn_fn, graph_data, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.gcn_fn = gcn_fn
        self.graph_data = graph_data   # tuple of GPU tensors

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(dims[i], dims[i + 1]) *
                               (2.0 / (dims[i] + dims[i + 1])) ** 0.5)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for i, w in enumerate(self.weights):
            x = self.gcn_fn.apply(x, w, *self.graph_data)
            if i < len(self.weights) - 1:
                x = F.relu(x)
        return x

# ── Training loop ────────────────────────────────────────────────────

def train_epochs(model, x, labels, num_classes, num_epochs, optimizer):
    """Train for num_epochs, return list of loss values."""
    losses = []
    for _ in range(num_epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = compute_loss(out, labels, num_classes)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    return losses

def benchmark_training(model, x, labels, num_classes, warmup_epochs, timed_epochs):
    """Warmup then time timed_epochs, return per-epoch times in ms."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(warmup_epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = compute_loss(out, labels, num_classes)
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
        loss = compute_loss(out, labels, num_classes)
        loss.backward()
        optimizer.step()
        end_events[i].record()

    torch.cuda.synchronize()
    return [start_events[i].elapsed_time(end_events[i]) for i in range(timed_epochs)]


def all_finite(values):
    """Return True iff every numeric value is finite."""
    return all(math.isfinite(v) for v in values)


def geometric_mean(values):
    vals = [v for v in values if v > 0 and math.isfinite(v)]
    if not vals:
        return -1.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def compute_loss(logits, labels, num_classes):
    """Use only real class logits; the tail is padding for kernel alignment."""
    if logits.size(1) < num_classes:
        raise ValueError(
            f"logit dimension {logits.size(1)} is smaller than num_classes {num_classes}"
        )
    return F.nll_loss(F.log_softmax(logits[:, :num_classes], dim=1), labels)


def make_trial_inputs(num_nodes, input_dim, num_classes, seed):
    feature_gen = torch.Generator(device="cuda")
    feature_gen.manual_seed(seed)
    label_gen = torch.Generator(device="cuda")
    label_gen.manual_seed(seed + 10_000)
    x = torch.randn(num_nodes, input_dim, device="cuda", generator=feature_gen)
    labels = torch.randint(0, num_classes, (num_nodes,), device="cuda", generator=label_gen)
    return x, labels


def run_correctness_trial(seed, graph_data, input_dim, output_dim, num_classes, ref_gcn_fn, sol_gcn_fn, x, labels):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ref_model = GCN(ref_gcn_fn, graph_data, input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS).cuda()
    sol_model = GCN(sol_gcn_fn, graph_data, input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS).cuda()
    sol_model.load_state_dict(ref_model.state_dict())

    ref_opt = torch.optim.Adam(ref_model.parameters(), lr=0.01)
    sol_opt = torch.optim.Adam(sol_model.parameters(), lr=0.01)

    ref_losses = train_epochs(ref_model, x, labels, num_classes, CORRECTNESS_EPOCHS, ref_opt)
    sol_losses = train_epochs(sol_model, x, labels, num_classes, CORRECTNESS_EPOCHS, sol_opt)

    del ref_model, sol_model, ref_opt, sol_opt
    torch.cuda.empty_cache()

    ref_first, ref_last = ref_losses[0], ref_losses[-1]
    sol_first, sol_last = sol_losses[0], sol_losses[-1]
    loss_ratio = sol_last / ref_last if ref_last > 0 else float("inf")

    return {
        "seed": seed,
        "ref_loss": {"first": ref_first, "last": ref_last},
        "solution_loss": {"first": sol_first, "last": sol_last},
        "loss_ratio": loss_ratio,
        "correct": True,
    }


def evaluate_graph(graph_name, graph_spec, ref_mod, ref_gcn_fn, sol_gcn_fn):
    graph_file = os.path.join(TASK_DIR, "data", f"{graph_name}.npz")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph data not found: {graph_file}")

    input_dim = graph_spec["input_dim"]
    num_classes = graph_spec["num_classes"]
    output_dim = round_up(num_classes, LOGIT_PAD_MULTIPLE)

    print(f"\n=== Dataset: {graph_name} ===", flush=True)
    print(f"Loading graph: {graph_file}...", flush=True)
    num_nodes, row_pointers, column_index = load_graph_csr(graph_file)
    num_edges = len(column_index)
    print(
        f"Nodes: {num_nodes}, Edges: {num_edges}, "
        f"Input dim: {input_dim}, Classes: {num_classes}, Padded output dim: {output_dim}"
    )

    blockPartition, edgeToColumn, edgeToRow = preprocess_graph(
        ref_mod, column_index, row_pointers, num_nodes
    )

    row_ptr_gpu = torch.from_numpy(row_pointers).int().cuda()
    col_idx_gpu = torch.from_numpy(column_index).int().cuda()
    bp_gpu = blockPartition.int().cuda()
    etc_gpu = edgeToColumn.int().cuda()
    etr_gpu = edgeToRow.int().cuda()
    graph_data = (row_ptr_gpu, col_idx_gpu, bp_gpu, etc_gpu, etr_gpu)

    print(f"=== GCN Training: {graph_name} ===")

    local_passed = True
    seed_results = []

    for seed in CORRECTNESS_SEEDS:
        x, labels = make_trial_inputs(num_nodes, input_dim, num_classes, seed)
        trial = run_correctness_trial(
            seed, graph_data, input_dim, output_dim, num_classes, ref_gcn_fn, sol_gcn_fn, x, labels
        )
        seed_results.append(trial)

        ref_first = trial["ref_loss"]["first"]
        ref_last = trial["ref_loss"]["last"]
        sol_first = trial["solution_loss"]["first"]
        sol_last = trial["solution_loss"]["last"]
        loss_ratio = trial["loss_ratio"]

        print(
            f"Seed {seed} [{graph_name}] Ref loss: {ref_first:.4f} -> {ref_last:.4f}"
        )
        print(
            f"Seed {seed} [{graph_name}] Solution loss: {sol_first:.4f} -> {sol_last:.4f}"
        )
        if math.isfinite(loss_ratio):
            print(f"Seed {seed} [{graph_name}] Loss ratio: {loss_ratio:.4f}")

        if not all_finite([ref_first, ref_last]):
            print(f"FAILED [{graph_name}] (reference produced non-finite loss for seed {seed})")
            trial["correct"] = False

        if not all_finite([sol_first, sol_last]):
            print(f"FAILED [{graph_name}] (solution produced non-finite loss for seed {seed})")
            trial["correct"] = False

        if all_finite([sol_first, sol_last]) and sol_last >= sol_first:
            print(f"FAILED [{graph_name}] (solution did not converge for seed {seed})")
            trial["correct"] = False

        if all_finite([ref_last, sol_last]) and ref_last > 0 and loss_ratio > MAX_LOSS_RATIO:
            print(
                f"FAILED [{graph_name}] (solution final loss ratio for seed {seed} "
                f"{loss_ratio:.4f} > {MAX_LOSS_RATIO:.2f})"
            )
            trial["correct"] = False

        local_passed = local_passed and trial["correct"]
        del x, labels
        torch.cuda.empty_cache()

    worst_seed_result = max(seed_results, key=lambda r: r["loss_ratio"])
    avg_loss_ratio = sum(r["loss_ratio"] for r in seed_results) / len(seed_results)

    print(
        f"Ref loss [{graph_name}]: {worst_seed_result['ref_loss']['first']:.4f} "
        f"-> {worst_seed_result['ref_loss']['last']:.4f} "
        f"(worst-ratio seed: {worst_seed_result['seed']})"
    )
    print(
        f"Solution loss [{graph_name}]: {worst_seed_result['solution_loss']['first']:.4f} "
        f"-> {worst_seed_result['solution_loss']['last']:.4f} "
        f"(worst-ratio seed: {worst_seed_result['seed']})"
    )
    print(
        f"Loss ratio [{graph_name}]: {worst_seed_result['loss_ratio']:.4f} "
        f"(worst seed), avg={avg_loss_ratio:.4f} over {len(seed_results)} seeds"
    )

    bench = {
        "ref_avg_ms": -1.0,
        "ref_min_ms": -1.0,
        "kernel_avg_ms": -1.0,
        "kernel_min_ms": -1.0,
        "speedup": -1.0,
    }

    if BENCHMARK_EPOCHS > 0 and local_passed:
        x, labels = make_trial_inputs(num_nodes, input_dim, num_classes, BENCHMARK_SEED)
        torch.manual_seed(BENCHMARK_SEED)
        torch.cuda.manual_seed_all(BENCHMARK_SEED)
        ref_model_b = GCN(ref_gcn_fn, graph_data, input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS).cuda()
        sol_model_b = GCN(sol_gcn_fn, graph_data, input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS).cuda()
        sol_model_b.load_state_dict(ref_model_b.state_dict())

        ref_times = benchmark_training(ref_model_b, x, labels, num_classes, WARMUP_EPOCHS, BENCHMARK_EPOCHS)
        sol_times = benchmark_training(sol_model_b, x, labels, num_classes, WARMUP_EPOCHS, BENCHMARK_EPOCHS)

        bench["ref_avg_ms"] = sum(ref_times) / len(ref_times)
        bench["ref_min_ms"] = min(ref_times)
        bench["kernel_avg_ms"] = sum(sol_times) / len(sol_times)
        bench["kernel_min_ms"] = min(sol_times)
        if bench["kernel_min_ms"] > 0:
            bench["speedup"] = bench["ref_min_ms"] / bench["kernel_min_ms"]

        print(
            f"Ref time [{graph_name}]: {bench['ref_avg_ms']:.4f} ms "
            f"(avg over {BENCHMARK_EPOCHS} epochs, min: {bench['ref_min_ms']:.4f} ms)"
        )
        print(
            f"Kernel time [{graph_name}]: {bench['kernel_avg_ms']:.4f} ms "
            f"(avg over {BENCHMARK_EPOCHS} epochs, min: {bench['kernel_min_ms']:.4f} ms)"
        )
        if bench["speedup"] > 0:
            print(f"Speedup [{graph_name}]: {bench['speedup']:.4f}x (ref_min / kernel_min)")

        del ref_model_b, sol_model_b
        torch.cuda.empty_cache()
        del x, labels

    del row_ptr_gpu, col_idx_gpu, bp_gpu, etc_gpu, etr_gpu
    torch.cuda.empty_cache()

    return {
        "graph": graph_name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "output_dim": output_dim,
        "correct": local_passed,
        "correctness_epochs": CORRECTNESS_EPOCHS,
        "seeds": seed_results,
        "seed_count": len(seed_results),
        "worst_seed": worst_seed_result["seed"],
        "avg_loss_ratio": avg_loss_ratio,
        "ref_loss": worst_seed_result["ref_loss"],
        "solution_loss": worst_seed_result["solution_loss"],
        "loss_ratio": worst_seed_result["loss_ratio"],
        "benchmark": bench,
    }

# ── Main ──────────────────────────────────────────────────────────────

def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_cpp = os.path.join(task_dir, 'wrapper.cpp')
    ref_cu = os.path.join(task_dir, 'ref_kernel.cu')
    sol_cu = os.path.join(task_dir, 'solution.cu')
    os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST

    cuda_flags = [
        '-O3',
        '--use_fast_math',
        '--expt-relaxed-constexpr',
        *AMPERE_GENCODE_FLAGS,
    ]

    print("Building reference extension...", flush=True)
    ref_mod = load(name='tcgnn_ref', sources=[wrapper_cpp, ref_cu],
                   extra_cuda_cflags=cuda_flags, verbose=False)

    print("Building solution extension...", flush=True)
    sol_mod = load(name='tcgnn_sol', sources=[wrapper_cpp, sol_cu],
                   extra_cuda_cflags=cuda_flags, verbose=False)

    # Create autograd functions
    ref_gcn_fn = make_gcn_function(ref_mod.forward)
    sol_gcn_fn = make_gcn_function(sol_mod.forward)

    graph_names = resolve_graph_names()
    print(f"Evaluating graphs: {', '.join(graph_names)}", flush=True)

    results = []
    all_passed = True
    for graph_name in graph_names:
        graph_result = evaluate_graph(graph_name, GRAPH_SPECS[graph_name], ref_mod, ref_gcn_fn, sol_gcn_fn)
        results.append(graph_result)
        all_passed = all_passed and graph_result["correct"]

    valid_bench = [r for r in results if r["benchmark"]["speedup"] > 0]
    agg_ref_time = (
        sum(r["benchmark"]["ref_avg_ms"] for r in valid_bench) / len(valid_bench)
        if valid_bench and len(valid_bench) == len(results) else -1.0
    )
    agg_kernel_time = (
        sum(r["benchmark"]["kernel_avg_ms"] for r in valid_bench) / len(valid_bench)
        if valid_bench and len(valid_bench) == len(results) else -1.0
    )
    agg_speedup = (
        geometric_mean([r["benchmark"]["speedup"] for r in valid_bench])
        if valid_bench and len(valid_bench) == len(results) else -1.0
    )

    worst_graph = max(results, key=lambda r: r["loss_ratio"])
    print("\n=== Aggregate Summary ===")
    print("Passed" if all_passed else "FAILED")
    print(
        f"Ref loss: {worst_graph['ref_loss']['first']:.4f} "
        f"-> {worst_graph['ref_loss']['last']:.4f} "
        f"(worst-ratio graph: {worst_graph['graph']})"
    )
    print(
        f"Solution loss: {worst_graph['solution_loss']['first']:.4f} "
        f"-> {worst_graph['solution_loss']['last']:.4f} "
        f"(worst-ratio graph: {worst_graph['graph']}, seed: {worst_graph['worst_seed']})"
    )
    print(f"Loss ratio: {worst_graph['loss_ratio']:.4f} (worst final ratio across graphs)")
    print(f"Average loss ratio: {worst_graph['avg_loss_ratio']:.4f} over {worst_graph['seed_count']} seeds")
    print(f"Correctness epochs: {CORRECTNESS_EPOCHS}")

    if agg_ref_time > 0 and agg_kernel_time > 0:
        print(f"Ref time: {agg_ref_time:.4f} ms (avg over {len(valid_bench)} graphs)")
        print(f"Kernel time: {agg_kernel_time:.4f} ms (avg over {len(valid_bench)} graphs)")
    if agg_speedup > 0:
        print(f"Speedup: {agg_speedup:.4f}x (geomean of per-graph speedups)")

    summary = {
        "graphs": results,
        "correct": all_passed,
        "aggregate": {
            "ref_time_ms": agg_ref_time,
            "kernel_time_ms": agg_kernel_time,
            "speedup": agg_speedup,
            "worst_graph": worst_graph["graph"],
            "loss_ratio": worst_graph["loss_ratio"],
            "avg_loss_ratio": worst_graph["avg_loss_ratio"],
            "worst_seed": worst_graph["worst_seed"],
            "seed_count": worst_graph["seed_count"],
            "correctness_epochs": CORRECTNESS_EPOCHS,
            "ref_loss": worst_graph["ref_loss"],
            "solution_loss": worst_graph["solution_loss"],
        },
    }
    print("RUN_SUMMARY_JSON " + json.dumps(summary, sort_keys=True))

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
