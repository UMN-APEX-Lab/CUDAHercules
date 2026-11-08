// Task tk_mamba2: Mamba-2 (SSD) forward pass on B200.
//
// Reference: ThunderKittens-style chunked kernel (64-token chunks, TMA double buffering, warp-level
// tensor-core MMA so that it runs on sm_100a). Solution under test (KH_TEST_SOLUTION): solution.h
//   void Mamba2Forward(Q, K, V, O, A, B, H, N, d_state)
// Inputs are generated on the device; reference and solution write to different, NaN-poisoned
// buffers; every output element is compared.

#ifdef KH_TEST_SOLUTION
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "solution.h"   // must stay before the reference code / kh_verify.h
#endif

#include "kittens.cuh"

// Reference kernel: Mamba-2 SSD forward (chunked state-space duality form), ThunderKittens-style,
// warp-level tensor-core MMA so that it runs on sm_100a. One CTA per (batch, 2 heads) sharing the
// Q/K (C/B) tiles; 64-token chunks; 8 warps = 2 heads x 4 row slices. d_state is a template parameter.
namespace khref_m2 {
using namespace kittens;

constexpr int CHUNK = 64;
constexpr int DH = 64;         // head dim (P)
constexpr int HPB = 2;         // heads per CTA
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;

template <int DS>
struct cfg {
  using qk_tile = st_bf<CHUNK, DS>;
  using v_tile = st_bf<CHUNK, DH>;
  using state_t = st_bf<DS, DH>;
  using qk_gl = gl<bf16, -1, -1, -1, DS, qk_tile>;
  using v_gl = gl<bf16, -1, -1, -1, DH, v_tile>;
  static constexpr size_t SMEM = 4 * sizeof(qk_tile) + 4 * sizeof(v_tile) + HPB * sizeof(v_tile) +
                                 HPB * sizeof(state_t) + 4096;
};

template <int DS>
struct globals {
  typename cfg<DS>::qk_gl q, k;     // [B, 1, N, DS]
  typename cfg<DS>::v_gl v, o;      // [B, H, N, DH]
  const float *a;                   // [B, H, N] log-decay (<= 0)
  int n, heads;
};

template <int DS>
__global__ __launch_bounds__(NTHREADS, 1) void m2_kernel(const __grid_constant__ globals<DS> g) {
  using C = cfg<DS>;
  using qk_tile = typename C::qk_tile;
  using v_tile = typename C::v_tile;
  using state_t = typename C::state_t;
  constexpr int NS = DS / 64;   // 16-row state slices per warp (warp owns DS/4 rows)
  extern __shared__ alignment_dummy __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);
  qk_tile(&q_s)[2] = al.allocate<qk_tile, 2>();
  qk_tile(&k_s)[2] = al.allocate<qk_tile, 2>();
  v_tile(&v_s)[2][HPB] = al.allocate<v_tile, 2, HPB>();
  v_tile(&vd_s)[HPB] = al.allocate<v_tile, HPB>();
  state_t(&st_s)[HPB] = al.allocate<state_t, HPB>();
  __shared__ float cum_s[HPB][CHUNK];
  __shared__ semaphore bar;

  const int batch = blockIdx.y;
  const int w = warpid(), lane = laneid();
  const int hl = w >> 2, rr = w & 3;
  const int head = blockIdx.x * HPB + hl;
  const int row0 = rr * 16;
  const int nblk = g.n / CHUNK;
  constexpr uint32_t BYTES = 2 * sizeof(qk_tile) + HPB * sizeof(v_tile);
  const float *a_ptr = g.a + (size_t(batch) * g.heads + head) * g.n;

  if (threadIdx.x == 0) {
    init_semaphore(bar, 0, 1);
    tma::expect_bytes(bar, BYTES);
    tma::load_async(q_s[0], g.q, {batch, 0, 0, 0}, bar);
    tma::load_async(k_s[0], g.k, {batch, 0, 0, 0}, bar);
#pragma unroll
    for (int i = 0; i < HPB; i++) tma::load_async(v_s[0][i], g.v, {batch, int(blockIdx.x) * HPB + i, 0, 0}, bar);
  }
  for (int i = threadIdx.x; i < HPB * DS * DH; i += NTHREADS) (&st_s[0].data[0])[i] = __float2bfloat16(0.f);
  rt_fl<16, 64> S[NS];   // state rows [DS/4 * rr + 16 i, ...) x DH, fp32
#pragma unroll
  for (int i = 0; i < NS; i++) warp::zero(S[i]);
  float2 a_cur = make_float2(0.f, 0.f);
  if (rr == 0) a_cur = reinterpret_cast<const float2 *>(a_ptr)[lane];   // chunk 0 (warps 0 and 4 do the scans)
  __syncthreads();

  int tic = 0;
  for (int blk = 0; blk < nblk; blk++, tic ^= 1) {
    wait(bar, tic);
    if (threadIdx.x == 0 && blk + 1 < nblk) {
      tma::expect_bytes(bar, BYTES);
      tma::load_async(q_s[tic ^ 1], g.q, {batch, 0, blk + 1, 0}, bar);
      tma::load_async(k_s[tic ^ 1], g.k, {batch, 0, blk + 1, 0}, bar);
#pragma unroll
      for (int i = 0; i < HPB; i++) tma::load_async(v_s[tic ^ 1][i], g.v, {batch, int(blockIdx.x) * HPB + i, blk + 1, 0}, bar);
    }
    // ---- inclusive cumulative sum of the log-decays of this chunk (one warp per head)
    float2 a_nxt = make_float2(0.f, 0.f);
    if (rr == 0) {
      if (blk + 1 < nblk) a_nxt = reinterpret_cast<const float2 *>(a_ptr + (blk + 1) * CHUNK)[lane];
      float s = a_cur.x + a_cur.y, incl = s;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        float t = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += t;
      }
      float c0 = incl - s + a_cur.x;
      cum_s[hl][2 * lane] = c0;
      cum_s[hl][2 * lane + 1] = c0 + a_cur.y;
      a_cur = a_nxt;
    }
    __syncthreads();
    const float tot = cum_s[hl][CHUNK - 1];

    // ---- intra-chunk: P = (Q K^T) * exp(cum_i - cum_j) [j <= i];  O = P V
    rt_fl<16, 64> Sc;
    warp::zero(Sc);
#pragma unroll
    for (int kk = 0; kk < DS / 32; kk++) {
      rt_bf<16, 32> a;
      rt_bf<64, 32> b;
      warp::load(a, q_s[tic].template subtile<16, 32>({rr, kk}));
      warp::load(b, k_s[tic].template subtile<64, 32>({0, kk}));
      warp::mma_ABt(Sc, a, b, Sc);
    }
    warp::apply(Sc, Sc, [&](int row, int col, float val) {
      return col <= row0 + row ? val * __expf(cum_s[hl][row0 + row] - cum_s[hl][col]) : 0.f;
    });
    rt_bf<16, 64> P;
    warp::copy(P, Sc);
    rt_fl<16, 64> O, Oi;
    warp::zero(O);
    warp::zero(Oi);
    {
      rt_bf<64, 64, ducks::rt_layout::col> vb;
      warp::load(vb, v_s[tic][hl]);
      warp::mma_AB(O, P, vb, O);
    }
    // ---- inter-chunk: O += exp(cum_i) * (Q S_prev)
#pragma unroll
    for (int kk = 0; kk < DS / 32; kk++) {
      rt_bf<16, 32> a;
      rt_bf<32, 64, ducks::rt_layout::col> b;
      warp::load(a, q_s[tic].template subtile<16, 32>({rr, kk}));
      warp::load(b, st_s[hl].template subtile<32, 64>({kk, 0}));
      warp::mma_AB(Oi, a, b, Oi);
    }
    warp::apply(Oi, Oi, [&](int row, int col, float val) { return val * __expf(cum_s[hl][row0 + row]); });
    warp::add(O, O, Oi);
    {
      auto og = g.o;
      warp::store(og, O, {batch, head, blk * 4 + rr, 0});
    }
    // ---- decayed V for the state update: vd[m] = v[m] * exp(tot - cum[m])
    for (int idx = (threadIdx.x & 127); idx < CHUNK * DH / 2; idx += 128) {
      if ((threadIdx.x >> 7) != hl) break;   // each head handled by its own 4 warps
      int m = idx / (DH / 2), c2 = idx % (DH / 2);
      float d = __expf(tot - cum_s[hl][m]);
      float2 x = __bfloat1622float2(*reinterpret_cast<bf16_2 *>(&v_s[tic][hl][{m, 2 * c2}]));
      *reinterpret_cast<bf16_2 *>(&vd_s[hl][{m, 2 * c2}]) = __floats2bfloat162_rn(x.x * d, x.y * d);
    }
    __syncthreads();   // vd complete, all warps done reading the state

    // ---- state update: S = exp(tot) S + K^T vd
    const float etot = __expf(tot);
#pragma unroll
    for (int i = 0; i < NS; i++) {
      warp::mul(S[i], S[i], etot);
      rt_bf<64, 16, ducks::rt_layout::col> a;
      rt_bf<64, 64, ducks::rt_layout::col> b;
      warp::load(a, k_s[tic].template subtile<64, 16>({0, NS * rr + i}));
      warp::load(b, vd_s[hl]);
      warp::mma_AtB(S[i], a, b, S[i]);
      auto sub = st_s[hl].template subtile<16, 64>({NS * rr + i, 0});
      warp::store(sub, S[i]);
    }
    __syncthreads();
  }
}

template <int DS>
inline globals<DS> make_globals(bf16 *q, bf16 *k, bf16 *v, bf16 *o, const float *a, int B, int H, int N) {
  size_t b_ = B, h_ = H, n_ = N, one = 1;
  typename cfg<DS>::qk_gl qg{q, b_, one, n_, nullptr}, kg{k, b_, one, n_, nullptr};
  typename cfg<DS>::v_gl vg{v, b_, h_, n_, nullptr}, og{o, b_, h_, n_, nullptr};
  return globals<DS>{qg, kg, vg, og, a, N, H};
}

template <int DS>
inline void launch(globals<DS> const &g, int B, int H, cudaStream_t s = 0) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(m2_kernel<DS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cfg<DS>::SMEM);
    init = true;
  }
  m2_kernel<DS><<<dim3(H / HPB, B), NTHREADS, cfg<DS>::SMEM, s>>>(g);
}
}  // namespace khref_m2


#include "kh_verify.h"
#include <cstdio>
#include <string>
#include <vector>

namespace {
using namespace khref_m2;

// Tolerances: |x - ref| <= max_rms * RMS(ref) + rel * |ref| for every element, and rel. Frobenius
// error <= fro. Measured with the shipped FP32 recurrence on all sizes: max|err|/rms <= 0.05,
// fro = 2.8e-3 (the reference deviates 2.5e-3 from the official FP32 `ssd_minimal_discrete`).
// Bad variants (fro on the timed sizes): decay 1+a instead of exp(a) 1.4e-2; decay of the
// previous token 2.6e-2; log-decay x1.05 / x0.95 3.4e-2 / 3.6e-2; Q,K rounded to FP8 3.3e-2;
// state reset every 2048 / 512 / 64 tokens 0.1 / 0.24 / 0.63; current token excluded 0.19; only
// the first half of d_state used 0.71; Q and K swapped 1.4; no decay 8-17; half of the outputs
// unwritten / empty: non-finite. State rounded to BF16 every step: 9.8e-3 (fails, narrowly); kept in
// FP16 every step: 3.1e-3 (passes, as accurate as the reference).
KhTol kTol(0.2, 8.5e-3, 0.03);

struct Case { const char *label; int B, H, N, DS; };

void dump(const char *dir, const char *name, const void *dptr, size_t bytes) {
  std::vector<char> h(bytes);
  KH_CHECK(cudaMemcpy(h.data(), dptr, bytes, cudaMemcpyDeviceToHost));
  std::string path = std::string(dir) + "/" + name;
  FILE *f = std::fopen(path.c_str(), "wb");
  std::fwrite(h.data(), 1, bytes, f);
  std::fclose(f);
}

__global__ void gen_decay(float *a, size_t n, int H, int N, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    int h = int((i / N) % H);
    // dt log-uniform in [1e-3, 3e-2], A_h in [1, 4]; a = -A_h * dt (per-token log decay <= 0)
    float u = kh_uniform_pos(i, seed, 0.f, 1.f);
    float dt = __expf(logf(1e-3f) + u * (logf(3e-2f) - logf(1e-3f)));
    float Ah = 1.f + 3.f * kh_uniform_pos(h, seed + 1u, 0.f, 1.f);
    a[i] = -Ah * dt;
  }
}

template <int DS>
bool run_case_ds(Case const &c, bool timed, int trials) {
  const int B = c.B, H = c.H, N = c.N;
  const size_t n_qk = size_t(B) * N * DS, n_v = size_t(B) * H * N * DH, n_a = size_t(B) * H * N;
  bf16 *q, *k, *v, *o_ref;
  float *a;
  KH_CHECK(cudaMalloc(&q, n_qk * 2)); KH_CHECK(cudaMalloc(&k, n_qk * 2));
  KH_CHECK(cudaMalloc(&v, n_v * 2)); KH_CHECK(cudaMalloc(&o_ref, n_v * 2));
  KH_CHECK(cudaMalloc(&a, n_a * 4));
  const float qk_scale = 1.732f * std::pow(float(DS), -0.25f);   // unit-variance q.k
  // C (Q), B (K), x*dt (V) and the per-token decay A are regenerated with a new seed before every
  // timed call and before the final re-check.
  unsigned seq = 1;
  auto refresh = [&](unsigned seed) {
    kh_fill_uniform(q, n_qk, 11u + 7919u * seed, qk_scale);
    kh_fill_uniform(k, n_qk, 22u + 7919u * seed, qk_scale);
    kh_fill_uniform(v, n_v, 33u + 7919u * seed, 1.7f);
    gen_decay<<<4096, 256>>>(a, n_a, H, N, 44u + 7919u * seed);
    KH_CHECK(cudaGetLastError());
  };
  refresh(seq);
  kh_poison(o_ref, n_v * 2);
  KH_CHECK(cudaDeviceSynchronize());

  globals<DS> g = make_globals<DS>(q, k, v, o_ref, a, B, H, N);
  auto run_ref = [&] { launch<DS>(g, B, H); };
  run_ref();
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {
    KhResult r = kh_compare(o_ref, o_ref, n_v, kTol);
    if (!r.pass) { std::fprintf(stderr, "Reference output incomplete or non-finite (non-finite = %llu)\n", r.nonfinite); ok = false; }
  }
  if (const char *dir = std::getenv("KH_DUMP_DIR")) {
    dump(dir, "q.bin", q, n_qk * 2); dump(dir, "k.bin", k, n_qk * 2); dump(dir, "v.bin", v, n_v * 2);
    dump(dir, "a.bin", a, n_a * 4); dump(dir, "o_ref.bin", o_ref, n_v * 2);
  }

#ifdef KH_TEST_SOLUTION
  bf16 *o_sol;
  KH_CHECK(cudaMalloc(&o_sol, n_v * 2));
  auto reset_sol = [&] { kh_poison(o_sol, n_v * 2); };
  auto call_sol = [&] {
    Mamba2Forward(reinterpret_cast<__nv_bfloat16 *>(q), reinterpret_cast<__nv_bfloat16 *>(k),
                  reinterpret_cast<__nv_bfloat16 *>(v), reinterpret_cast<__nv_bfloat16 *>(o_sol), a, B, H, N, DS);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "Solution kernel failed: %s\n", cudaGetErrorString(e)); std::exit(1); }
  };
  if (ok) {
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(o_sol, o_ref, n_v, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
#ifdef KH_TOL_SCAN   // development aid: how many elements would fail at other (max_rms, rel) settings
    for (double m : {0.01, 0.02, 0.03, 0.05, 0.1}) for (double rl : {0.0, 0.005, 0.01, 0.02, 0.04, 0.08}) {
      KhResult rs = kh_compare(o_sol, o_ref, n_v, KhTol(m, 1.0, rl));
      std::printf("SCAN max_rms=%.3f rel=%.3f nbad=%llu max_err_rel=%.4e fro=%.4e\n", m, rl, rs.nbad, rs.max_err_rel, rs.fro_rel);
    }
#endif
  }
#endif
  if (ok && timed && trials > 0) {
    KhTime tr = kh_time(run_ref, trials, [&] { refresh(++seq); kh_poison(o_ref, n_v * 2); });
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
    KhResult rr = kh_compare(o_sol, o_ref, n_v, kTol);
    kh_print_result("solution (re-check, fresh inputs)", rr, kTol);
    ok = rr.pass;
  }
#endif
#ifdef KH_TEST_SOLUTION
  cudaFree(o_sol);
#endif
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o_ref); cudaFree(a);
  return ok;
}

bool run_case(Case const &c, bool timed, int trials) {
  return c.DS == 64 ? run_case_ds<64>(c, timed, trials) : run_case_ds<128>(c, timed, trials);
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

  if (argc >= 5) {   // ./test B H N d_state : single custom shape
    Case c{"custom", std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
    std::printf("\n=== custom: B=%d, H=%d, N=%d, d_state=%d ===\n", c.B, c.H, c.N, c.DS);
    if (!run_case(c, true, trials)) { std::printf("Incorrect\n"); return 1; }
    std::printf("Passed\n");
    return 0;
  }
  std::vector<Case> cases = {
      {"b8_n4k", 8, 80, 4096, 128},
      {"b4_n8k", 4, 80, 8192, 128},
      {"b2_n16k", 2, 80, 16384, 128},
  };

  // Correctness-only cases (odd batch / head counts, short sequences); not timed.
  std::vector<Case> small = {
      {"d_state64", 3, 6, 1344, 64},
      {"odd_heads", 3, 10, 1344, 128},
      {"short_seq", 2, 8, 64, 128},
  };
  for (auto const &c : small) {
    std::printf("\n--- %s: B=%d, H=%d, N=%d, d_state=%d (correctness only) ---\n", c.label, c.B, c.H, c.N, c.DS);
    if (!run_case(c, false, 0)) { std::printf("Incorrect\n"); if (!keep_going) return 1; failed = true; }
    else std::printf("Passed\n");
  }

  for (auto const &c : cases) {
    std::printf("\n=== %s: B=%d, H=%d, N=%d, d_state=%d ===\n", c.label, c.B, c.H, c.N, c.DS);
    if (!run_case(c, true, trials)) { std::printf("Incorrect\n"); if (!keep_going) return 1; failed = true; }
    else std::printf("Passed\n");
  }
  return failed ? 1 : 0;
}
