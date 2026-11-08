#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
void HedgehogAttention(
    __nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O,
    __nv_bfloat16 *Q_map, __nv_bfloat16 *K_map,
    float *alphas, float *betas,
    int B, int H, int N);
