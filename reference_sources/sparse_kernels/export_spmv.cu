// Exports one SpMV reference kernel under fixed C names, so the harness can load every
// per-matrix reference as its own shared library.
void *spmv_preprocess(const int *, const int *, const float *, int, int, int);
void spmv_compute(void *, const int *, const int *, const float *, const float *, float *, int, int, int);
void spmv_cleanup(void *);

#define KH_EXPORT extern "C" __attribute__((visibility("default")))
KH_EXPORT void *kh_ref_preprocess(const int *rp, const int *ci, const float *v, int rows, int cols, int nnz) {
  return spmv_preprocess(rp, ci, v, rows, cols, nnz);
}
KH_EXPORT void kh_ref_compute(void *ctx, const int *rp, const int *ci, const float *v, const float *x, float *y,
                              int rows, int cols, int nnz) {
  spmv_compute(ctx, rp, ci, v, x, y, rows, cols, nnz);
}
KH_EXPORT void kh_ref_cleanup(void *ctx) { spmv_cleanup(ctx); }
