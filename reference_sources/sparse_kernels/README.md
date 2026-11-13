# Sparse reference kernels

Reference implementations of the CSR sparse tasks `class1/general/90_csr_spmv`, `91_csr_spmm` and
`92_csr_spgemm`: one tuned kernel per test matrix (and per K for SpMM), each implementing the task's
`*_preprocess` / `*_compute` / `*_cleanup` contract.

- `spmv/<matrix>.cu`, `spmm/<matrix>_k<K>.cu`, `spgemm/<matrix>.cu`: the per-case kernels
- `sparse_formats.{cu,cuh}`: host-side format conversions (COO, ELL, SELL, BSR, ...) used by some
  kernels in their preprocess step
- `export_{spmv,spmm,spgemm}.cu`: re-export a kernel's contract functions as
  `kh_ref_preprocess` / `kh_ref_compute` / `kh_ref_cleanup`

The task Makefiles build every kernel as its own shared library under `build/<arch hash>/` (cached,
git-ignored); the harness loads the library of the case it runs. These sources are not shown to the
model under evaluation.
