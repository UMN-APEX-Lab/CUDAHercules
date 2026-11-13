// Exports one SpMM reference kernel under fixed C names, so the harness can load every
// per-matrix, per-K reference as its own shared library.
void *spmm_preprocess(const int *, const int *, const float *, int, int, int, int);
void spmm_compute(void *, const int *, const int *, const float *, const float *, float *, int, int, int, int);
void spmm_cleanup(void *);

#define KH_EXPORT extern "C" __attribute__((visibility("default")))
KH_EXPORT void *kh_ref_preprocess(const int *rp, const int *ci, const float *v, int rows, int cols, int nnz,
                                  int K) {
  return spmm_preprocess(rp, ci, v, rows, cols, nnz, K);
}
KH_EXPORT void kh_ref_compute(void *ctx, const int *rp, const int *ci, const float *v, const float *B, float *C,
                              int rows, int cols, int nnz, int K) {
  spmm_compute(ctx, rp, ci, v, B, C, rows, cols, nnz, K);
}
KH_EXPORT void kh_ref_cleanup(void *ctx) { spmm_cleanup(ctx); }
