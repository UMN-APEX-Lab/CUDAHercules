// Task tk_hedgehog: Hedgehog attention (sliding window softmax + learned feature-map linear attention) on B200.
//
// Reference: ThunderKittens-style chunked kernel (64-token chunks, TMA double buffering, warp-level
// tensor-core MMA so that it runs on sm_100a). Solution under test (KH_TEST_SOLUTION): solution.h
//   void HedgehogAttention(Q, K, V, O, Q_map, K_map, alphas, betas, B, H, N)
// Inputs are generated on the device; reference and solution write to different, NaN-poisoned
// buffers; every output element is compared.

#ifdef KH_TEST_SOLUTION
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "solution.h"   // must stay before the reference code / kh_verify.h
#endif

#include "kittens.cuh"

// Reference kernel: Hedgehog attention (block sliding window softmax + learned-feature-map linear attention,
// jointly normalised), ThunderKittens-style, warp-level tensor-core MMA so that it runs on sm_100a.
// One CTA per (batch, head); 64-token chunks; 8 warps = 4 row slices x 2 output-column halves.
namespace khref_hh {
using namespace kittens;

constexpr int CHUNK = 64;
constexpr int HD = 128;     // head dim (q, k, v, o)
constexpr int NF = 64;      // feature-map width per sign (feature dim = 2 * NF = 128)
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;

using tile_t = st_bf<CHUNK, HD>;
using map_t = st_bf<HD, NF>;
using tile_gl = gl<bf16, -1, -1, -1, -1, tile_t>;
using map_gl = gl<bf16, -1, -1, -1, -1, map_t>;
using kv_t = st_bf<2 * NF, HD>;

struct globals {
  tile_gl q, k, v, o;
  map_gl qmap, kmap;
  const float *alphas, *betas;   // [H]
  int n;
};

constexpr size_t SMEM_BYTES = (2 + 3 + 3 + 1) * sizeof(tile_t) + 2 * sizeof(map_t) + sizeof(kv_t) + 1024 * 2 + 1024;

template <ducks::rt::all RT>
__device__ __forceinline__ void softmax_rows(RT &t) {   // rows are already scaled by log2(e)
  typename RT::col_vec mx, sm;
  warp::row_max(mx, t);
  warp::sub_row(t, t, mx);
  warp::exp2(t, t);
  warp::row_sum(sm, t);
  warp::div_row(t, t, sm);
}

__global__ __launch_bounds__(NTHREADS, 1) void hh_kernel(const __grid_constant__ globals g) {
  extern __shared__ alignment_dummy __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);
  tile_t(&q_s)[2] = al.allocate<tile_t, 2>();
  tile_t(&k_s)[3] = al.allocate<tile_t, 3>();
  tile_t(&v_s)[3] = al.allocate<tile_t, 3>();
  tile_t &phik_s = al.allocate<tile_t>();          // featurised previous K block (64 tokens x 128 features)
  map_t &qmap_s = al.allocate<map_t>();
  map_t &kmap_s = al.allocate<map_t>();
  kv_t &kv_s = al.allocate<kv_t>();                // KV state, bf16 copy (128 features x 128 v)
  __shared__ float ksum_s[2 * NF];                 // sum of phi(k) over the linear region (fp32)
  __shared__ semaphore bar;

  const int batch = blockIdx.y, head = blockIdx.x;
  const int w = warpid();
  const int rr = w & 3, c = w >> 2;
  const float alpha = g.alphas[head], beta = g.betas[head];
  const int nblk = g.n / CHUNK;
  constexpr float SCALE = 0.08838834764831845f * 1.4426950408889634f;   // 1/sqrt(128) * log2(e)
  constexpr float LOG2E = 1.4426950408889634f;

  if (threadIdx.x == 0) {
    init_semaphore(bar, 0, 1);
    tma::expect_bytes(bar, 3 * sizeof(tile_t) + 2 * sizeof(map_t));
    tma::load_async(qmap_s, g.qmap, {0, head, 0, 0}, bar);
    tma::load_async(kmap_s, g.kmap, {0, head, 0, 0}, bar);
    tma::load_async(q_s[0], g.q, {batch, head, 0, 0}, bar);
    tma::load_async(k_s[0], g.k, {batch, head, 0, 0}, bar);
    tma::load_async(v_s[0], g.v, {batch, head, 0, 0}, bar);
  }
  for (int i = threadIdx.x; i < 2 * NF * HD; i += NTHREADS) kv_s.data[i] = __float2bfloat16(0.f);
  if (threadIdx.x < 2 * NF) ksum_s[threadIdx.x] = 0.f;
  rt_fl<16, 32> kv[4];   // KV state rows [16w, 16w+16): feature rows x 128 value columns
#pragma unroll
  for (int j = 0; j < 4; j++) warp::zero(kv[j]);
  __syncthreads();

  int tic = 0, cur = 0;   // cur = ring slot of K/V block `blk`
  for (int blk = 0; blk < nblk; blk++, tic ^= 1, cur = (cur == 2 ? 0 : cur + 1)) {
    const int prv = (cur == 0 ? 2 : cur - 1);
    wait(bar, tic);
    if (threadIdx.x == 0 && blk + 1 < nblk) {
      tma::expect_bytes(bar, 3 * sizeof(tile_t));
      tma::load_async(q_s[tic ^ 1], g.q, {batch, head, blk + 1, 0}, bar);
      tma::load_async(k_s[(cur + 1) % 3], g.k, {batch, head, blk + 1, 0}, bar);
      tma::load_async(v_s[(cur + 1) % 3], g.v, {batch, head, blk + 1, 0}, bar);
    }

    // ---------------- sliding window (previous + current 64-token block, causal) ----------------
    rt_fl<16, 64> S0, S1;
    rt_fl<16, 64>::col_vec mx, den;
    warp::zero(S1);
#pragma unroll
    for (int kk = 0; kk < HD / 32; kk++) {
      rt_bf<16, 32> a;
      rt_bf<64, 32> b;
      warp::load(a, q_s[tic].template subtile<16, 32>({rr, kk}));
      warp::load(b, k_s[cur].template subtile<64, 32>({0, kk}));
      warp::mma_ABt(S1, a, b, S1);
    }
    if (blk >= 1) {
      warp::zero(S0);
#pragma unroll
      for (int kk = 0; kk < HD / 32; kk++) {
        rt_bf<16, 32> a;
        rt_bf<64, 32> b;
        warp::load(a, q_s[tic].template subtile<16, 32>({rr, kk}));
        warp::load(b, k_s[prv].template subtile<64, 32>({0, kk}));
        warp::mma_ABt(S0, a, b, S0);
      }
    } else {
      warp::neg_infty(S0);
    }
    {
      const int row0 = rr * 16;
      warp::apply(S1, S1, [&](int row, int col, float val) { return col <= row0 + row ? val * SCALE : -INFINITY; });
    }
    warp::mul(S0, S0, SCALE);
    warp::row_max(mx, S1);
    warp::row_max(mx, S0, mx);
    warp::sub_row(S0, S0, mx);
    warp::sub_row(S1, S1, mx);
    warp::exp2(S0, S0);
    warp::exp2(S1, S1);
    warp::mul(S0, S0, beta);
    warp::mul(S1, S1, beta);
    warp::row_sum(den, S1);
    warp::row_sum(den, S0, den);

    rt_fl<16, 64> O;
    warp::zero(O);
    {
      rt_bf<16, 64> P;
      rt_bf<64, 64, ducks::rt_layout::col> vb;
      warp::copy(P, S1);
      warp::load(vb, v_s[cur].template subtile<64, 64>({0, c}));
      warp::mma_AB(O, P, vb, O);
      if (blk >= 1) {
        warp::copy(P, S0);
        warp::load(vb, v_s[prv].template subtile<64, 64>({0, c}));
        warp::mma_AB(O, P, vb, O);
      }
    }

    // ---------------- linear attention over blocks <= blk-2 with the learned feature map ----------------
    if (blk >= 1) {
      rt_fl<16, 64> z;
      warp::zero(z);
#pragma unroll
      for (int kk = 0; kk < HD / 32; kk++) {
        rt_bf<16, 32> a;
        rt_bf<32, 64, ducks::rt_layout::col> b;
        warp::load(a, q_s[tic].template subtile<16, 32>({rr, kk}));
        warp::load(b, qmap_s.template subtile<32, 64>({kk, 0}));
        warp::mma_AB(z, a, b, z);
      }
#pragma unroll
      for (int sgn = 0; sgn < 2; sgn++) {   // sgn 0: softmax(z) features, sgn 1: softmax(-z) features
        rt_fl<16, 64> phi;
        warp::mul(phi, z, sgn == 0 ? LOG2E : -LOG2E);
        softmax_rows(phi);
        warp::mul(phi, phi, alpha);
        // normaliser: den += phi . ksum (FP32)
        {
          rt_fl<16, 64> t;
          warp::apply(t, phi, [&](int row, int col, float val) { return val * ksum_s[sgn * NF + col]; });
          warp::row_sum(den, t, den);
        }
        rt_bf<16, 64> L;
        warp::copy(L, phi);
        rt_bf<64, 64, ducks::rt_layout::col> b;
        warp::load(b, kv_s.template subtile<64, 64>({sgn, c}));
        warp::mma_AB(O, L, b, O);
      }
    }
    warp::div_row(O, O, den);
    {
      // direct store of the register tile (rows rr*16.., columns c*64..)
      auto og = g.o;
      warp::store(og, O, {batch, head, blk * 4 + rr, c});   // coordinates in units of the 16 x 64 register tile
    }

    // ---------------- featurise the previous K block and fold it into the state ----------------
    if (blk >= 1) {
      rt_fl<16, 64> z;
      warp::zero(z);
#pragma unroll
      for (int kk = 0; kk < HD / 32; kk++) {
        rt_bf<16, 32> a;
        rt_bf<32, 64, ducks::rt_layout::col> b;
        warp::load(a, k_s[prv].template subtile<16, 32>({rr, kk}));
        warp::load(b, kmap_s.template subtile<32, 64>({kk, 0}));
        warp::mma_AB(z, a, b, z);
      }
      warp::mul(z, z, c == 0 ? LOG2E : -LOG2E);
      softmax_rows(z);
      { auto sub = phik_s.template subtile<16, 64>({rr, c}); warp::store(sub, z); }
    }
    __syncthreads();   // phi(K) complete; everyone is done reading kv_s / ksum_s
    if (blk >= 1) {
      if (threadIdx.x < 2 * NF) {
        float s = 0.f;
#pragma unroll 8
        for (int m = 0; m < CHUNK; m++) s += __bfloat162float(phik_s[{m, int(threadIdx.x)}]);
        ksum_s[threadIdx.x] += s;
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        rt_bf<64, 16, ducks::rt_layout::col> a;
        rt_bf<64, 32, ducks::rt_layout::col> b;
        warp::load(a, phik_s.template subtile<64, 16>({0, w}));
        warp::load(b, v_s[prv].template subtile<64, 32>({0, j}));
        warp::mma_AtB(kv[j], a, b, kv[j]);
      }
#pragma unroll
      for (int j = 0; j < 4; j++) { auto sub = kv_s.template subtile<16, 32>({w, j}); warp::store(sub, kv[j]); }
    }
    __syncthreads();
  }
}

inline globals make_globals(bf16 *q, bf16 *k, bf16 *v, bf16 *o, bf16 *qmap, bf16 *kmap,
                            const float *alphas, const float *betas, int B, int H, int N) {
  size_t b_ = B, h_ = H, n_ = N, d_ = HD, one = 1, nf = NF;
  tile_gl qg{q, b_, h_, n_, d_}, kg{k, b_, h_, n_, d_}, vg{v, b_, h_, n_, d_}, og{o, b_, h_, n_, d_};
  map_gl qm{qmap, one, h_, d_, nf}, km{kmap, one, h_, d_, nf};
  return globals{qg, kg, vg, og, qm, km, alphas, betas, N};
}

inline void launch(globals const &g, int B, int H, cudaStream_t s = 0) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(hh_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
    init = true;
  }
  hh_kernel<<<dim3(H, B), NTHREADS, SMEM_BYTES, s>>>(g);
}
}  // namespace khref_hh


#include "kh_verify.h"
#include <cstdio>
#include <string>
#include <vector>

namespace {
using namespace khref_hh;

// Tolerances: |x - ref| <= max_rms * RMS(ref) + rel * |ref| for every element, and rel. Frobenius
// error <= fro. Measured with the shipped FP32 token-by-token kernel on all sizes: max|err|/rms
// <= 0.17 (heavy-tailed outputs: the error scales with |ref|, hence the rel term), fro = 2.7e-3
// (the reference deviates 2.3e-3 .. 2.6e-3 from an independent FP32 PyTorch computation).
// Bad variants (fro on the timed sizes): linear state rounded to BF16 1.0e-2 .. 2.0e-2 (grows
// with N); Q rounded to FP8 2.5e-2; no alpha 0.14; no beta 0.15-0.18; no negative features 0.2;
// linear region includes the previous block 0.2; unnormalised features 0.5; alpha/beta swapped
// 0.65; window = current block only 0.8; no max shift 1.7-2.3; no linear part 2.6-5.4; scale 1
// instead of 1/sqrt(128) 1.7; no final normalisation 33-120; half of the outputs unwritten / empty:
// non-finite.
KhTol kTol(0.2, 8.0e-3, 0.03);

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
  const size_t n_el = size_t(B) * H * N * HD, n_map = size_t(H) * HD * NF;
  bf16 *q, *k, *v, *o_ref, *qmap, *kmap;
  float *alphas, *betas;
  KH_CHECK(cudaMalloc(&q, n_el * 2)); KH_CHECK(cudaMalloc(&k, n_el * 2));
  KH_CHECK(cudaMalloc(&v, n_el * 2)); KH_CHECK(cudaMalloc(&o_ref, n_el * 2));
  KH_CHECK(cudaMalloc(&qmap, n_map * 2)); KH_CHECK(cudaMalloc(&kmap, n_map * 2));
  KH_CHECK(cudaMalloc(&alphas, H * 4)); KH_CHECK(cudaMalloc(&betas, H * 4));
  // Q, K, V are regenerated with a new seed before every timed call and before the final re-check
  // (the feature maps, alphas and betas are model weights and stay fixed).
  unsigned seq = 1;
  auto refresh = [&](unsigned seed) {
    kh_fill_uniform(q, n_el, 11u + 7919u * seed, 2.0f);
    kh_fill_uniform(k, n_el, 22u + 7919u * seed, 2.0f);
    kh_fill_uniform(v, n_el, 33u + 7919u * seed, 1.0f);
  };
  refresh(seq);
  kh_fill_uniform(qmap, n_map, 44u, 0.25f);
  kh_fill_uniform(kmap, n_map, 55u, 0.25f);
  {
    std::vector<float> ha(H), hb(H);
    for (int h = 0; h < H; h++) { ha[h] = kh_uniform_pos(h, 66u, 0.5f, 2.0f); hb[h] = kh_uniform_pos(h, 77u, 0.25f, 1.0f); }
    KH_CHECK(cudaMemcpy(alphas, ha.data(), H * 4, cudaMemcpyHostToDevice));
    KH_CHECK(cudaMemcpy(betas, hb.data(), H * 4, cudaMemcpyHostToDevice));
  }
  kh_poison(o_ref, n_el * 2);
  KH_CHECK(cudaDeviceSynchronize());

  globals g = make_globals(q, k, v, o_ref, qmap, kmap, alphas, betas, B, H, N);
  auto run_ref = [&] { launch(g, B, H); };
  run_ref();
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {
    KhResult r = kh_compare(o_ref, o_ref, n_el, kTol);
    if (!r.pass) { std::fprintf(stderr, "Reference output incomplete or non-finite (non-finite = %llu)\n", r.nonfinite); ok = false; }
  }
  if (const char *dir = std::getenv("KH_DUMP_DIR")) {
    dump(dir, "q.bin", q, n_el * 2); dump(dir, "k.bin", k, n_el * 2); dump(dir, "v.bin", v, n_el * 2);
    dump(dir, "qmap.bin", qmap, n_map * 2); dump(dir, "kmap.bin", kmap, n_map * 2);
    dump(dir, "alphas.bin", alphas, H * 4); dump(dir, "betas.bin", betas, H * 4);
    dump(dir, "o_ref.bin", o_ref, n_el * 2);
  }

#ifdef KH_TEST_SOLUTION
  bf16 *o_sol;
  KH_CHECK(cudaMalloc(&o_sol, n_el * 2));
  auto reset_sol = [&] { kh_poison(o_sol, n_el * 2); };
  auto call_sol = [&] {
    HedgehogAttention(reinterpret_cast<__nv_bfloat16 *>(q), reinterpret_cast<__nv_bfloat16 *>(k),
                      reinterpret_cast<__nv_bfloat16 *>(v), reinterpret_cast<__nv_bfloat16 *>(o_sol),
                      reinterpret_cast<__nv_bfloat16 *>(qmap), reinterpret_cast<__nv_bfloat16 *>(kmap),
                      alphas, betas, B, H, N);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "Solution kernel failed: %s\n", cudaGetErrorString(e)); std::exit(1); }
  };
  if (ok) {
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(o_sol, o_ref, n_el, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
#ifdef KH_TOL_SCAN   // development aid: how many elements would fail at other (max_rms, rel) settings
    for (double m : {0.01, 0.02, 0.03, 0.05, 0.1}) for (double rl : {0.0, 0.005, 0.01, 0.02, 0.04, 0.08}) {
      KhResult rs = kh_compare(o_sol, o_ref, n_el, KhTol(m, 1.0, rl));
      std::printf("SCAN max_rms=%.3f rel=%.3f nbad=%llu max_err_rel=%.4e fro=%.4e\n", m, rl, rs.nbad, rs.max_err_rel, rs.fro_rel);
    }
#endif
  }
#endif
  if (ok && timed && trials > 0) {
    KhTime tr = kh_time(run_ref, trials, [&] { refresh(++seq); kh_poison(o_ref, n_el * 2); });
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
    KhResult rr = kh_compare(o_sol, o_ref, n_el, kTol);
    kh_print_result("solution (re-check, fresh inputs)", rr, kTol);
    ok = rr.pass;
  }
#endif
#ifdef KH_TEST_SOLUTION
  cudaFree(o_sol);
#endif
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o_ref); cudaFree(qmap); cudaFree(kmap);
  cudaFree(alphas); cudaFree(betas);
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
    std::printf("\n=== custom: B=%d, H=%d, N=%d, D=%d ===\n", c.B, c.H, c.N, HD);
    if (!run_case(c, true, trials)) { std::printf("Incorrect\n"); return 1; }
    std::printf("Passed\n");
    return 0;
  }
  std::vector<Case> cases = {
      {"b8_n4k", 8, 32, 4096},
      {"b8_n8k", 8, 32, 8192},
      {"b4_n16k", 4, 32, 16384},
  };

  // Correctness-only cases (odd batch / head counts, short sequences); not timed.
  std::vector<Case> small = {
      {"odd_shape", 3, 5, 1344},
      {"short_seq", 2, 8, 128},
  };
  for (auto const &c : small) {
    std::printf("\n--- %s: B=%d, H=%d, N=%d, D=128 (correctness only) ---\n", c.label, c.B, c.H, c.N);
    if (!run_case(c, false, 0)) { std::printf("Incorrect\n"); if (!keep_going) return 1; failed = true; }
    else std::printf("Passed\n");
  }

  for (auto const &c : cases) {
    std::printf("\n=== %s: B=%d, H=%d, N=%d, D=%d ===\n", c.label, c.B, c.H, c.N, HD);
    if (!run_case(c, true, trials)) { std::printf("Incorrect\n"); if (!keep_going) return 1; failed = true; }
    else std::printf("Passed\n");
  }
  return failed ? 1 : 0;
}
