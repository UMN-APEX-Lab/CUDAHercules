// Task tk_based: Based (2nd-order Taylor) causal linear attention on B200.
//
// Reference: ThunderKittens-style chunked kernel (64-token chunks, TMA double buffering, warp-level
// tensor-core MMA so that it runs on sm_100a). Solution under test (KH_TEST_SOLUTION): solution.h
//   void BasedLinearAttention(Q, K, V, O, B, H, N)
// Inputs are generated on the device; reference and solution write to different, NaN-poisoned
// buffers; every output element is compared.

#ifdef KH_TEST_SOLUTION
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "solution.h"   // must stay before the reference code / kh_verify.h
#endif

#include "kittens.cuh"

// Reference kernel: Based linear attention (2nd-order Taylor feature map), ThunderKittens-style, warp-level
// tensor-core MMA so that it runs on sm_100a. One CTA per (batch, head); 64-token chunks; the running
// state (sum v, sum k^T v, sum (k (x) k)^T v) is kept in FP32 registers and exposed to the MMAs as BF16.
namespace khref_based {
using namespace kittens;

constexpr int CHUNK = 64;
constexpr int DQK = 16;
constexpr int DVO = 64;
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;

using qk_tile = st_bf<CHUNK, DQK>;
using vo_tile = st_bf<CHUNK, DVO>;
using st_state = st_bf<16, DVO>;           // one 16 x 64 slice of the second-order state (or the first-order state)
using qk_gl = gl<bf16, -1, -1, -1, DQK, qk_tile>;
using vo_gl = gl<bf16, -1, -1, -1, DVO, vo_tile>;

struct globals {
  qk_gl q, k;
  vo_gl v, o;
  int n;
};

constexpr size_t SMEM_BYTES = 4 * sizeof(qk_tile) + 3 * sizeof(vo_tile) + 17 * sizeof(st_state) + 1024 * 2 + 256 + 2048;

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }

__global__ __launch_bounds__(NTHREADS, 1) void based_kernel(const __grid_constant__ globals g) {
  extern __shared__ alignment_dummy __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);
  qk_tile(&q_s)[2] = al.allocate<qk_tile, 2>();
  qk_tile(&k_s)[2] = al.allocate<qk_tile, 2>();
  vo_tile(&v_s)[2] = al.allocate<vo_tile, 2>();
  vo_tile &o_s = al.allocate<vo_tile>();
  st_state(&st_s)[17] = al.allocate<st_state, 17>();   // [0..15]: A2 slices (e = 0..15), [16]: A1
  __shared__ float a0s[DVO];
  __shared__ semaphore bar;

  const int batch = blockIdx.y, head = blockIdx.x;
  const int w = warpid(), lane = laneid();
  const int rr = w & 3, ch = w >> 2, row0 = rr * 16;
  const int nblk = g.n / CHUNK;
  constexpr uint32_t BYTES = 2 * sizeof(qk_tile) + sizeof(vo_tile);

  if (threadIdx.x == 0) {
    init_semaphore(bar, 0, 1);
    tma::expect_bytes(bar, BYTES);
    tma::load_async(q_s[0], g.q, {batch, head, 0, 0}, bar);
    tma::load_async(k_s[0], g.k, {batch, head, 0, 0}, bar);
    tma::load_async(v_s[0], g.v, {batch, head, 0, 0}, bar);
  }
  for (int i = threadIdx.x; i < 17 * 16 * DVO; i += NTHREADS) (&st_s[0].data[0])[i] = __float2bfloat16(0.f);
  if (threadIdx.x < DVO) a0s[threadIdx.x] = 0.f;
  rt_fl<16, 32> a2[2][2];   // A2 slices e = 2w, 2w+1 (rows f, columns v split in halves)
  rt_fl<16, 32> a1h;        // warps 0,1: column half w of A1
#pragma unroll
  for (int i = 0; i < 2; i++) { warp::zero(a2[i][0]); warp::zero(a2[i][1]); }
  warp::zero(a1h);
  __syncthreads();

  int tic = 0;
  for (int blk = 0; blk < nblk; blk++, tic ^= 1) {
    wait(bar, tic);
    if (threadIdx.x == 0 && blk + 1 < nblk) {
      tma::expect_bytes(bar, BYTES);
      tma::load_async(q_s[tic ^ 1], g.q, {batch, head, blk + 1, 0}, bar);
      tma::load_async(k_s[tic ^ 1], g.k, {batch, head, blk + 1, 0}, bar);
      tma::load_async(v_s[tic ^ 1], g.v, {batch, head, blk + 1, 0}, bar);
    }

    // ---- local part: P = causal(1 + s + s^2/2), s = q.k / sqrt(16);  O = P V
    rt_bf<16, 16> Qr;
    warp::load(Qr, q_s[tic].template subtile<16, 16>({rr, 0}));
    rt_fl<16, 64> S;
    warp::zero(S);
    {
      rt_bf<64, 16> Kb;
      warp::load(Kb, k_s[tic]);
      warp::mma_ABt(S, Qr, Kb, S);
    }
    warp::apply(S, S, [&](int row, int col, float val) {
      float s = val * 0.25f;
      return (col <= row0 + row) ? 1.f + s + 0.5f * s * s : 0.f;
    });
    rt_bf<16, 64> P;
    warp::copy(P, S);
    rt_fl<16, 32> O;
    warp::zero(O);
    {
      rt_bf<64, 32, ducks::rt_layout::col> vh;
      warp::load(vh, v_s[tic].template subtile<64, 32>({0, ch}));
      warp::mma_AB(O, P, vh, O);
    }
    // ---- zeroth order: running sum of V
    warp::apply(O, O, [&](int row, int col, float val) { return val + a0s[ch * 32 + col]; });
    // ---- first order: (q/4) @ A1
    {
      rt_bf<16, 16> Aq;
      warp::mul(Aq, Qr, __float2bfloat16(0.25f));
      rt_bf<16, 32, ducks::rt_layout::col> b;
      warp::load(b, st_s[16].template subtile<16, 32>({0, ch}));
      warp::mma_AB(O, Aq, b, O);
    }
    // ---- second order: sum_e ((q_e q) / 32) @ A2[e]
    const int rt_ = row0 + lane / 4;
#pragma unroll
    for (int e = 0; e < 16; e++) {
      float qt = b2f(q_s[tic][{rt_, e}]) * (1.f / 32.f);
      float qb = b2f(q_s[tic][{rt_ + 8, e}]) * (1.f / 32.f);
      rt_bf<16, 16> Ae;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float sc = (k % 2 == 0) ? qt : qb;
        float2 x = __bfloat1622float2(Qr.tiles[0][0].data[k]);
        Ae.tiles[0][0].data[k] = __floats2bfloat162_rn(sc * x.x, sc * x.y);
      }
      rt_bf<16, 32, ducks::rt_layout::col> b;
      warp::load(b, st_s[e].template subtile<16, 32>({0, ch}));
      warp::mma_AB(O, Ae, b, O);
    }
    { auto o_sub = o_s.template subtile<16, 32>({rr, ch}); warp::store(o_sub, O); }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();   // o_s complete; all warps done reading the state
    if (threadIdx.x == 0) tma::store_async(g.o, o_s, {batch, head, blk, 0});

    // ---- state update with this chunk
    if (threadIdx.x < DVO) {
      float s = 0.f;
#pragma unroll 8
      for (int m = 0; m < CHUNK; m++) s += b2f(v_s[tic][{m, int(threadIdx.x)}]);
      a0s[threadIdx.x] += s;
    }
    {
      rt_bf<64, 16, ducks::rt_layout::col> kcol;
      warp::load(kcol, k_s[tic]);
      rt_bf<64, 32, ducks::rt_layout::col> vh[2];
      warp::load(vh[0], v_s[tic].template subtile<64, 32>({0, 0}));
      warp::load(vh[1], v_s[tic].template subtile<64, 32>({0, 1}));
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int e = 2 * w + i;
        rt_bf<64, 16, ducks::rt_layout::col> ke;
        warp::apply(ke, kcol, [&](int m, int f, bf16 val) {
          return __float2bfloat16(b2f(val) * b2f(k_s[tic][{m, e}]));
        });
        warp::mma_AtB(a2[i][0], ke, vh[0], a2[i][0]);
        warp::mma_AtB(a2[i][1], ke, vh[1], a2[i][1]);
      }
      if (w < 2) {
        if (w == 0) warp::mma_AtB(a1h, kcol, vh[0], a1h);
        else        warp::mma_AtB(a1h, kcol, vh[1], a1h);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      auto s0 = st_s[2 * w + i].template subtile<16, 32>({0, 0});
      auto s1 = st_s[2 * w + i].template subtile<16, 32>({0, 1});
      warp::store(s0, a2[i][0]);
      warp::store(s1, a2[i][1]);
    }
    if (w < 2) { auto s1 = st_s[16].template subtile<16, 32>({0, w}); warp::store(s1, a1h); }
    if (threadIdx.x == 0) tma::store_async_read_wait();
    __syncthreads();
  }
  if (threadIdx.x == 0) tma::store_async_wait();
}

inline globals make_globals(bf16 *q, bf16 *k, bf16 *v, bf16 *o, int B, int H, int N) {
  size_t b_ = B, h_ = H, n_ = N;
  qk_gl qg{q, b_, h_, n_, nullptr}, kg{k, b_, h_, n_, nullptr};
  vo_gl vg{v, b_, h_, n_, nullptr}, og{o, b_, h_, n_, nullptr};
  return globals{qg, kg, vg, og, N};
}

inline void launch(globals const &g, int B, int H, cudaStream_t s = 0) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(based_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
    init = true;
  }
  based_kernel<<<dim3(H, B), NTHREADS, SMEM_BYTES, s>>>(g);
}
}  // namespace khref_based


#include "kh_verify.h"
#include <cstdio>
#include <string>
#include <vector>

namespace {
using namespace khref_based;

// Tolerances: |x - ref| <= max_rms * RMS(ref) + rel * |ref| for every element, and rel. Frobenius
// error <= fro. Measured with the shipped FP32 quadratic kernel on all sizes: max|err|/rms <= 0.05,
// fro = 2.0e-3 (the reference deviates 1.9e-3 from an independent FP32 PyTorch computation).
// Bad variants (fro on the timed sizes): current token excluded 1.1e-2 (N=16K) .. 1.6e-2 (N=8K);
// Q,K rounded to FP8 2.4e-2; wrong scale (0.2 instead of 1/4) 0.15; one 128-token history tile
// skipped 0.12-0.18; no quadratic / no linear / no constant term 0.32 / 0.49 / 0.65; square
// without the 1/2 0.32; unscaled q.k 5.0; half of the outputs unwritten / empty: non-finite.
// (Taylor weights rounded to BF16 are as accurate as the reference: 2.9e-3, passes.)
KhTol kTol(0.2, 6.0e-3, 0.03);

struct Case { const char *label; int B, H, N; };

void dump(const char *dir, const char *name, const void *dptr, size_t bytes) {
  std::vector<char> h(bytes);
  KH_CHECK(cudaMemcpy(h.data(), dptr, bytes, cudaMemcpyDeviceToHost));
  std::string path = std::string(dir) + "/" + name;
  FILE *f = std::fopen(path.c_str(), "wb");
  std::fwrite(h.data(), 1, bytes, f);
  std::fclose(f);
}

bool run_case(Case const &c, bool timed, int trials) {
  const int B = c.B, H = c.H, N = c.N;
  const size_t n_qk = size_t(B) * H * N * DQK, n_vo = size_t(B) * H * N * DVO;
  bf16 *q, *k, *v, *o_ref;
  KH_CHECK(cudaMalloc(&q, n_qk * 2)); KH_CHECK(cudaMalloc(&k, n_qk * 2));
  KH_CHECK(cudaMalloc(&v, n_vo * 2)); KH_CHECK(cudaMalloc(&o_ref, n_vo * 2));
  // Q, K, V are regenerated with a new seed before every timed call and before the final re-check.
  unsigned seq = 1;
  auto refresh = [&](unsigned seed) {
    kh_fill_uniform(q, n_qk, 11u + 7919u * seed, 1.5f);
    kh_fill_uniform(k, n_qk, 22u + 7919u * seed, 1.5f);
    kh_fill_uniform(v, n_vo, 33u + 7919u * seed, 1.0f);
  };
  refresh(seq);
  kh_poison(o_ref, n_vo * 2);
  KH_CHECK(cudaDeviceSynchronize());

  globals g = make_globals(q, k, v, o_ref, B, H, N);
  auto run_ref = [&] { launch(g, B, H); };
  run_ref();
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {
    KhResult r = kh_compare(o_ref, o_ref, n_vo, kTol);
    if (!r.pass) { std::fprintf(stderr, "Reference output incomplete or non-finite (non-finite = %llu)\n", r.nonfinite); ok = false; }
  }
  if (const char *dir = std::getenv("KH_DUMP_DIR")) {
    dump(dir, "q.bin", q, n_qk * 2); dump(dir, "k.bin", k, n_qk * 2); dump(dir, "v.bin", v, n_vo * 2);
    dump(dir, "o_ref.bin", o_ref, n_vo * 2);
  }

#ifdef KH_TEST_SOLUTION
  bf16 *o_sol;
  KH_CHECK(cudaMalloc(&o_sol, n_vo * 2));
  auto reset_sol = [&] { kh_poison(o_sol, n_vo * 2); };
  auto call_sol = [&] {
    BasedLinearAttention(reinterpret_cast<__nv_bfloat16 *>(q), reinterpret_cast<__nv_bfloat16 *>(k),
                         reinterpret_cast<__nv_bfloat16 *>(v), reinterpret_cast<__nv_bfloat16 *>(o_sol), B, H, N);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "Solution kernel failed: %s\n", cudaGetErrorString(e)); std::exit(1); }
  };
  if (ok) {
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(o_sol, o_ref, n_vo, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
#ifdef KH_TOL_SCAN   // development aid: how many elements would fail at other (max_rms, rel) settings
    for (double m : {0.01, 0.02, 0.03, 0.05, 0.1}) for (double rl : {0.0, 0.005, 0.01, 0.02, 0.04, 0.08}) {
      KhResult rs = kh_compare(o_sol, o_ref, n_vo, KhTol(m, 1.0, rl));
      std::printf("SCAN max_rms=%.3f rel=%.3f nbad=%llu max_err_rel=%.4e fro=%.4e\n", m, rl, rs.nbad, rs.max_err_rel, rs.fro_rel);
    }
#endif
  }
#endif
  if (ok && timed && trials > 0) {
    KhTime tr = kh_time(run_ref, trials, [&] { refresh(++seq); kh_poison(o_ref, n_vo * 2); });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { refresh(++seq); reset_sol(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#endif
  }
#ifdef KH_TEST_SOLUTION
  // Final re-check, unconditionally (also without timing): fresh inputs, reference recomputed for them,
  // freshly poisoned solution output. A solution that memoises its output fails here.
  if (ok) {
    refresh(++seq);   // fresh inputs: recompute the reference, then run the solution on them
    run_ref();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult rr = kh_compare(o_sol, o_ref, n_vo, kTol);
    kh_print_result("solution (re-check, fresh inputs)", rr, kTol);
    ok = rr.pass;
  }
#endif
#ifdef KH_TEST_SOLUTION
  cudaFree(o_sol);
#endif
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o_ref);
  return ok;
}
}  // namespace

int main(int argc, char **argv) {
  cudaDeviceProp props;
  KH_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10 || props.minor != 0) {
    std::fprintf(stderr, "This task requires a Blackwell sm_100 GPU (B200).\n");
    return 0;
  }
  const int trials = kh_trials();
  const bool keep_going = std::getenv("KH_KEEP_GOING") != nullptr;
  bool failed = false;

  if (argc >= 4) {   // ./test B H N : single custom shape
    Case c{"custom", std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};
    std::printf("\n=== custom: B=%d, H=%d, N=%d ===\n", c.B, c.H, c.N);
    if (!run_case(c, true, trials)) { std::printf("Incorrect\n"); return 1; }
    std::printf("Passed\n");
    return 0;
  }
  std::vector<Case> cases = {
      {"b8_n8k", 8, 32, 8192},
      {"b8_n16k", 8, 32, 16384},
      {"b4_n16k", 4, 32, 16384},
  };

  // Correctness-only cases (odd batch / head counts, short sequences); not timed.
  std::vector<Case> small = {
      {"odd_shape", 3, 5, 1408},
      {"short_seq", 2, 8, 128},
  };
  for (auto const &c : small) {
    std::printf("\n--- %s: B=%d, H=%d, N=%d (correctness only) ---\n", c.label, c.B, c.H, c.N);
    if (!run_case(c, false, 0)) { std::printf("Incorrect\n"); if (!keep_going) return 1; failed = true; }
    else std::printf("Passed\n");
  }

  for (auto const &c : cases) {
    std::printf("\n=== %s: B=%d, H=%d, N=%d ===\n", c.label, c.B, c.H, c.N);
    if (!run_case(c, true, trials)) { std::printf("Incorrect\n"); if (!keep_going) return 1; failed = true; }
    else std::printf("Passed\n");
  }
  return failed ? 1 : 0;
}
