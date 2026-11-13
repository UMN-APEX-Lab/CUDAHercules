// Exports one SpGEMM reference kernel under fixed C names, so the harness can load every
// per-matrix reference as its own shared library.
void *spgemm_preprocess(const int *, const int *, const float *, int, int, int);
void spgemm_compute(void *, const int *, const int *, const float *, int **, int **, float **, int *, int, int,
                    int);
void spgemm_cleanup(void *);

#define KH_EXPORT extern "C" __attribute__((visibility("default")))
KH_EXPORT void *kh_ref_preprocess(const int *rp, const int *ci, const float *v, int rows, int cols, int nnz) {
  return spgemm_preprocess(rp, ci, v, rows, cols, nnz);
}
KH_EXPORT void kh_ref_compute(void *ctx, const int *rp, const int *ci, const float *v, int **c_rp, int **c_ci,
                              float **c_v, int *c_nnz, int rows, int cols, int nnz) {
  spgemm_compute(ctx, rp, ci, v, c_rp, c_ci, c_v, c_nnz, rows, cols, nnz);
}
KH_EXPORT void kh_ref_cleanup(void *ctx) { spgemm_cleanup(ctx); }
