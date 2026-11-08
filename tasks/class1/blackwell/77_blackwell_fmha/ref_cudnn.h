#pragma once
// Reference implementation of the FMHA task: cuDNN scaled-dot-product attention through the
// cudnn-frontend graph API (cudnn-frontend: header-only, Apache-2.0 / MIT; cuDNN >= 9.x). Compiled
// in its own translation unit (ref_cudnn.cpp) so that neither the solution nor the harness TU sees
// any cuDNN header.
#include <cstdint>

struct KhCudnnSdpa;

// Builds the cuDNN graph (heuristic engine selection, plan construction, workspace allocation) for
// BSHD tensors: Q, O [B, S, Hq, D], K, V [B, S, Hkv, D], LSE [B, Hq, S] FP32 (natural log of the
// softmax denominator including the max, i.e. logsumexp of the scaled scores); scale 1/sqrt(D).
// dtype: 0 = FP16, 1 = BF16. Aborts the process with a message on any cuDNN failure.
KhCudnnSdpa *khcudnn_create(int B, int Hq, int Hkv, int S, int D, bool causal, int dtype,
                            void const *Q, void const *K, void const *V, void *O, float *LSE);
// Launches the attention kernel(s) on the default stream.
void khcudnn_run(KhCudnnSdpa *p);
void khcudnn_destroy(KhCudnnSdpa *p);
