#pragma once

#include "icicle/curves/params/bn254.h"

extern "C" {

int kh_custom_backend_init();
void kh_custom_backend_shutdown();
const char* kh_custom_backend_last_error();

int kh_custom_ntt_forward_bn254(
    const bn254::scalar_t* input,
    int log_n,
    bn254::scalar_t* output);

int kh_custom_msm_bn254(
    const bn254::scalar_t* scalars,
    const bn254::affine_t* points,
    int log_n,
    bn254::projective_t* output);

}
