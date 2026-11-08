#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
void BasedLinearAttention(
    __nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O,
    int B, int H, int N);
