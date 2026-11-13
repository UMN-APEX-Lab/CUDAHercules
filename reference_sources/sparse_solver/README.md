# Sparse direct solver reference

Two programs that solve large complex-symmetric sparse systems `A x = b` (several right-hand sides) on one GPU.

- `ref_order`: GPU nested-dissection ordering. Reads A, builds a 2x2-pivot matching, compresses the graph,
  dissects it (GPU separators for the top levels, METIS for the leaves) and writes a permutation and the
  block tree of the dissection.
- `ref_solve`: supernodal multifrontal complex-symmetric LDL^T in FP32 on the GPU (GPU symbolic analysis,
  numeric factorization, supernodal triangular solve). The FP32 factors are the preconditioner of a
  right-preconditioned FP64 FGMRES with double-double residuals, which delivers an FP64-accurate solution.

## Build

Requires the CUDA toolkit and a conda env (or any prefix) with METIS, SuiteSparse (CHOLMOD) and MKL.

```
make                                   # SOLVER_ENV=<prefix> CUDA=/usr/local/cuda ARCH=sm_XY[a]
```

`ARCH` defaults to the GPU of the build machine (`nvidia-smi --query-gpu=compute_cap`; the `a` variants are
used for compute capability 90/100/101/103/110/120/121). `SOLVER_ENV` defaults to the `ref-solver` conda env (see `environment.yml`). The
binaries link the libraries from `$(SOLVER_ENV)/lib` through an rpath. Outputs: `build/<ARCH>/ref_order`, `build/<ARCH>/ref_solve`.
The nvcc warnings about the deprecated `double4` type are harmless.

## Input

A directory `<dir>` holding two files in the same binary layout:

```
header   32 bytes: uint32 info (bit fields: data type, index type, base, format, store, symmetry),
                   4 bytes padding, int64 rows, int64 cols, int64 nnz
rows     int32[nnz]   0-based row indices
cols     int32[nnz]   0-based column indices
values   complex128[nnz]   (re, im)
crc      uint32  CRC32 over everything before it (header included)
```

- `<dir>/A.bin`: the upper triangle of an n x n complex-symmetric matrix, entries sorted by row and then by
  column. File size must be exactly `32 + nnz * 16 + nnz * 8 + 4`.
- `<dir>/B.bin`: the sparse right-hand sides, `cols` = number of columns (ports), rows = n.

## Run

```
ref_order <dir> --threads 64 --gpu -o <plan_prefix>
ref_solve --whole --reps 1 --streams 16 --gmres --mp-tol 1e-8 --mp-iters 200 --no-verify \
          --user-perm <plan_prefix>.perm --user-tree <plan_prefix>.rowtree <dir>
```

`ref_order` writes `<plan_prefix>.perm` (gather permutation, int32), `<plan_prefix>.rowtree` (block sizes of
the dissection tree, int32) and `<plan_prefix>.match` (int32 involution of the 2x2 pivot pairs).
`ref_solve` needs the `.perm` and `.rowtree` files.

Without `--no-verify`, `ref_solve` also checks the result on the host. With `REF_EXT_RESIDUAL=1` it
additionally computes the residual with long double sums.

## Report lines

`ref_order` (stdout):

```
  ordering total           :     1.71 s
```

`ref_solve` (stdout, last lines):

```
[testbench] once init <s> read <s> convert <s> upload <s> analysis <s> verify <s> wall <s>
[testbench] freq <id> factor <s> solve <s> solves <count> [residual <value> factor-check <value>]
```

`factor` is the numeric factorization, `solve` the whole FP64 solve (preconditioner solves and FGMRES), `solves`
the number of FP32 preconditioner solves. With verification enabled, these lines are printed as well:

```
  ||b-Ax||/||b||           : <rel. residual of every right-hand side>
  residual, long double sum: <same, long double accumulation>      (only with REF_EXT_RESIDUAL=1)
```
