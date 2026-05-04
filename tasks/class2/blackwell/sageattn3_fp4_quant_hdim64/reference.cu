/*
 * SageAttention v3 reference: FP4 quantization, head_dim=64
 */
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// ─── Inline FP4 conversion (from SageAttention v3) ─────────────────────────
inline __device__ uint32_t fp32_vec_to_e2m1(float2 *array) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// ─── Type helpers ───────────────────────────────────────────────────────────
template <typename T> struct TypeConverter { using Type = half2; };
template <> struct TypeConverter<half2> { using Type = half; };
template <> struct TypeConverter<half> { using Type = half2; };

template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[8];
};

constexpr int CVT_FP4_ELTS_PER_THREAD = 16;

// ─── Reference quantization kernel (from SageAttention v3) ─────────────────
template <uint32_t head_dim, uint32_t BLOCK_SIZE, bool permute, typename T>
__global__ void ref_fp4_quant_kernel(
    const T* input, uint8_t* output, uint8_t* output_sf,
    int batch_size, int num_heads, int num_tokens,
    int stride_bz_input, int stride_h_input, int stride_seq_input,
    int stride_bz_output, int stride_h_output, int stride_seq_output,
    int stride_bz_output_sf, int stride_h_output_sf, int stride_seq_output_sf) {
  using PV = PackedVec<T>;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;
  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

  int load_token_id;
  if constexpr (!permute) {
    load_token_id = token_id;
  } else {
    int local_token_id = threadIdx.x / NUM_THREADS_PER_TOKEN;
    int local_token_id_residue = local_token_id % 32;
    load_token_id = token_block_id * BLOCK_SIZE + (local_token_id / 32) * 32 +
                    (local_token_id_residue / 8) * 2 +
                    ((local_token_id_residue % 8) / 2) * 8 +
                    (local_token_id_residue % 8) % 2;
  }

  PV in_vec;
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++)
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;

  if (load_token_id < num_tokens) {
    in_vec = reinterpret_cast<PV const*>(input +
                batch_id * stride_bz_input +
                head_id * stride_h_input +
                load_token_id * stride_seq_input +
                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0];
  }

  auto localMax = __habs2(in_vec.elts[0]);
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++)
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8)
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);

  float vecMax = float(__hmax(localMax.x, localMax.y));
  float SFValue = vecMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = __half22float2(in_vec.elts[i]);
    fp2Vals[i].x *= SFValueInv;
    fp2Vals[i].y *= SFValueInv;
  }

  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++)
    e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output +
                batch_id * stride_bz_output +
                head_id * stride_h_output +
                token_id * stride_seq_output +
                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output +
                batch_id * stride_bz_output +
                head_id * stride_h_output +
                token_id * stride_seq_output +
                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }

  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf + (token_id / 64) * 64 * stride_seq_output_sf;
  uint32_t token_id_local = token_id % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = (threadIdx.x % NUM_THREADS_PER_TOKEN) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                            (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }
}

// ─── Entry point ────────────────────────────────────────────────────────────
extern "C"
void sageattn3_fp4_quant_hdim64(
    const void* input,
    void* output,
    void* output_sf,
    int batch_size,
    int num_heads,
    int num_tokens,
    cudaStream_t stream
) {
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 128;

    int stride_bz_input  = num_heads * num_tokens * HEAD_DIM;
    int stride_h_input   = num_tokens * HEAD_DIM;
    int stride_seq_input = HEAD_DIM;

    int stride_bz_output  = num_heads * num_tokens * (HEAD_DIM / 2);
    int stride_h_output   = num_tokens * (HEAD_DIM / 2);
    int stride_seq_output = HEAD_DIM / 2;

    int stride_bz_output_sf  = num_heads * num_tokens * (HEAD_DIM / 16);
    int stride_h_output_sf   = num_tokens * (HEAD_DIM / 16);
    int stride_seq_output_sf = HEAD_DIM / 16;

    dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
    dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

    ref_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, false, half>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(input),
            reinterpret_cast<uint8_t*>(output),
            reinterpret_cast<uint8_t*>(output_sf),
            batch_size, num_heads, num_tokens,
            stride_bz_input, stride_h_input, stride_seq_input,
            stride_bz_output, stride_h_output, stride_seq_output,
            stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf);
}
