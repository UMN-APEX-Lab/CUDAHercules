// vLLM Marlin launcher for the W4A16 reference (bf16 activations, uint4b8 weights, group size 128).
// Launch logic adapted from vllm/csrc/quantization/marlin/marlin.cu (marlin_mm), weight repack kernel
// from gptq_marlin_repack.cu, scale permutation from marlin_utils.py::marlin_permute_scales
// (all Apache-2.0, see marlin/LICENSE); torch dispatch removed, plan computed once in the constructor.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "marlin_ref.h"
#include "kernel.h"
#include "marlin_template.h"

using namespace marlin;

#define MREF_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

// Marlin weight repack (verbatim kernel from gptq_marlin_repack.cu)
#include "marlin_repack_kernel.inc"

namespace {

struct ThreadCfg { int thread_k, thread_n, num_threads; };
// vLLM's priority lists (marlin.cu)
const ThreadCfg kSmallCfgs[] = {{128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
const ThreadCfg kLargeCfgs[] = {{64, 256, 256}, {64, 128, 128}, {128, 64, 128}};
constexpr int kStages = 4, kGroupSize = 128, kNumBits = 4;

int cache_size(ThreadCfg const &th, int thread_m_blocks, int stages) {
  int pack_factor = 32 / kNumBits;
  int tb_k = th.thread_k, tb_n = th.thread_n, tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * 2;
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);
  int tb_groups = div_ceil(tb_k, kGroupSize);
  int sh_s_size = tb_groups * tb_n * 2 * stages;
  return tmp_size + sh_a_size + sh_s_size;
}
bool valid_config(ThreadCfg const &th, int thread_m_blocks, int prob_n, int prob_k, int max_shared_mem) {
  if (prob_k % th.thread_k != 0 || prob_n % th.thread_n != 0) return false;
  if (th.thread_n < min_thread_n || th.thread_k < min_thread_k) return false;
  if (th.num_threads < 128) return false;
  return cache_size(th, thread_m_blocks, kStages) <= max_shared_mem;
}

using KernelFn = void (*)(MARLIN_KERNEL_PARAMS);

// Only the kernels needed by the benchmark shapes are instantiated (each takes ~1 min to compile);
// anything else returns nullptr and the harness falls back to the CUTLASS reference.
KernelFn get_kernel(int thread_m_blocks, int thread_n_blocks, int thread_k_blocks, bool m8, int threads) {
#define MREF_K(TH, TM, TN, TK, M8)                                                              \
  if (threads == TH && thread_m_blocks == TM && thread_n_blocks == TN && thread_k_blocks == TK && \
      m8 == M8)                                                                                  \
    return Marlin<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), \
                  TH, TM, TN, TK, M8, kStages, 8, false>;
  MREF_K(256, 1, 8, 8, true)     // m <= 8     (thread tile k128 x n128)
  MREF_K(256, 1, 8, 8, false)    // m <= 16
  MREF_K(256, 4, 16, 4, false)   // m 17..64  (k64 x n256, 256 threads)
#undef MREF_K
  return nullptr;
}

struct Launch {
  KernelFn fn; int blocks, threads, smem;
  int4 const *A; int4 *C; int m, n, k;
};

// harness layout [N][K/8] uint32 (two's complement nibbles) -> GPTQ layout [K/8][N] uint32, offset binary
__global__ void to_gptq_layout(uint32_t const *in, uint32_t *out, int N, int K8) {
  size_t tot = size_t(N) * K8;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < tot; i += size_t(gridDim.x) * blockDim.x) {
    int n = int(i / K8), k8 = int(i % K8);
    out[size_t(k8) * N + n] = in[i] ^ 0x88888888u;   // q in [-8,7] -> q + 8 in [0,15] (uint4b8)
  }
}
// harness scales [N][G] -> Marlin scales [G][N] with the (8x8) scale permutation of marlin_permute_scales
__global__ void permute_scales(__nv_bfloat16 const *in, __nv_bfloat16 *out, int N, int G) {
  size_t tot = size_t(N) * G;
  for (size_t o = blockIdx.x * size_t(blockDim.x) + threadIdx.x; o < tot; o += size_t(gridDim.x) * blockDim.x) {
    size_t row = o / 64; int c = int(o % 64);
    int perm = c / 8 + 8 * (c % 8);     // scale_perm[i * 8 + j] = i + 8 * j
    size_t src = row * 64 + perm;       // flat index into the [G][N] scale matrix
    int g = int(src / N), n = int(src % N);
    out[o] = in[size_t(n) * G + g];
  }
}

// vLLM's launch heuristics (marlin_mm) evaluated once. Returns false if a needed kernel is not built.
bool make_plan(int M, int N, int K, int sms, int max_shared_mem, void const *A, void *C,
               std::vector<Launch> *plan) {
  if (K % 128 != 0 || N % 64 != 0) return false;
  int max_par = 16;
  if (N <= 4096) max_par = 16 * 8;
  int rest_m = M, max_tmb = 4;
  int4 const *A_ptr = reinterpret_cast<int4 const *>(A);
  int4 *C_ptr = reinterpret_cast<int4 *>(C);
  while (rest_m) {
    int par_count = rest_m / (max_tmb * 16);
    if (par_count > max_par) par_count = max_par;
    int m_split = par_count > 0 ? par_count * (max_tmb * 16) : rest_m;
    int tmb = std::min(div_ceil(m_split, 16), max_tmb);
    bool m8 = m_split <= 8;
    ThreadCfg const *cfgs = tmb > 1 ? kLargeCfgs : kSmallCfgs;
    ThreadCfg tc{-1, -1, -1};
    for (int i = 0; i < 3; i++)
      if (valid_config(cfgs[i], tmb, N, K, max_shared_mem - 512)) { tc = cfgs[i]; break; }
    if (tc.thread_n != -1) {
      if (N / tc.thread_n * div_ceil(m_split, tmb * 16) * 4 <= sms)
        if (valid_config({128, 64, 128}, tmb, N, K, max_shared_mem)) tc = {128, 64, 128};
    }
    // Tuned for B200 (sweep, see description): for 17..64 rows vLLM's switch to the 128-thread 64-wide tile
    // above is 1.3x slower than its own (64, 256, 256) kernel at the benchmark shape.
    if (tmb > 1 && m_split <= 64 && valid_config({64, 256, 256}, tmb, N, K, max_shared_mem - 512))
      tc = {64, 256, 256};
    if (tc.thread_k == -1 && max_tmb > 1) { max_tmb--; continue; }
    if (tc.thread_k == -1) return false;
    KernelFn fn = get_kernel(tmb, tc.thread_n / 16, tc.thread_k / 16, m8, tc.num_threads);
    if (!fn) return false;
    plan->push_back({fn, sms, tc.num_threads, max_shared_mem, A_ptr, C_ptr, m_split, N, K});
    A_ptr += size_t(m_split) * (K / 8);
    C_ptr += size_t(m_split) * (N / 8);
    rest_m -= m_split;
  }
  return true;
}

int device_sms() { int v = 0; MREF_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0)); return v; }
int device_smem() { int v = 0; MREF_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)); return v; }

}  // namespace

struct MarlinRefImpl {
  std::vector<Launch> plan;
  uint32_t *Bm = nullptr; __nv_bfloat16 *Sp = nullptr; float *ctmp = nullptr; int *locks = nullptr;
  ~MarlinRefImpl() { cudaFree(Bm); cudaFree(Sp); cudaFree(ctmp); cudaFree(locks); }
};

bool marlin_ref_supported(int M, int N, int K) {
  std::vector<Launch> plan;
  return make_plan(M, N, K, device_sms(), device_smem(), nullptr, nullptr, &plan);
}

MarlinRef::MarlinRef(uint8_t const *B, __nv_bfloat16 const *S, __nv_bfloat16 const *A, __nv_bfloat16 *D,
                     int M, int N, int K) : impl_(new MarlinRefImpl) {
  int sms = device_sms(), smem = device_smem(), G = K / 128;
  if (!make_plan(M, N, K, sms, smem, A, D, &impl_->plan)) { std::fprintf(stderr, "Marlin: unsupported shape\n"); std::exit(2); }
  uint32_t *Bg = nullptr;
  size_t wbytes = size_t(N) * K / 2;
  MREF_CHECK(cudaMalloc(&Bg, wbytes));
  MREF_CHECK(cudaMalloc(&impl_->Bm, wbytes));
  MREF_CHECK(cudaMalloc(&impl_->Sp, size_t(N) * G * 2));
  MREF_CHECK(cudaMalloc(&impl_->ctmp, size_t(sms) * 64 * max_thread_n * sizeof(float)));
  MREF_CHECK(cudaMalloc(&impl_->locks, sms * sizeof(int)));
  MREF_CHECK(cudaMemset(impl_->locks, 0, sms * sizeof(int)));   // the kernel leaves the locks zeroed
  to_gptq_layout<<<4096, 256>>>(reinterpret_cast<uint32_t const *>(B), Bg, N, K / 8);
  MREF_CHECK(cudaFuncSetAttribute(gptq_marlin_repack_kernel<repack_threads, 4>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  gptq_marlin_repack_kernel<repack_threads, 4><<<sms, repack_threads, smem>>>(Bg, impl_->Bm, K, N);
  permute_scales<<<1024, 256>>>(S, impl_->Sp, N, G);
  MREF_CHECK(cudaGetLastError());
  MREF_CHECK(cudaDeviceSynchronize());
  cudaFree(Bg);
  for (auto &l : impl_->plan)
    MREF_CHECK(cudaFuncSetAttribute(l.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, l.smem));
}
MarlinRef::~MarlinRef() = default;

void MarlinRef::run() {
  for (auto &l : impl_->plan) {
    l.fn<<<l.blocks, l.threads, l.smem>>>(l.A, reinterpret_cast<int4 const *>(impl_->Bm), l.C,
                                         reinterpret_cast<int4 *>(impl_->ctmp), nullptr,
                                         reinterpret_cast<int4 const *>(impl_->Sp), nullptr, nullptr,
                                         l.m, l.n, l.k, l.k, impl_->locks, false, /*use_atomic_add=*/false,
                                         /*use_fp32_reduce=*/true, l.smem);
  }
  MREF_CHECK(cudaGetLastError());
}
