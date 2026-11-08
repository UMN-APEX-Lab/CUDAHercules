// Task tk_linear_attention: causal linear attention with per-head exponential decay on B200.
//
// Reference: ThunderKittens-style chunked kernel (64-token chunks, TMA double buffering, warp-level
// tensor-core MMA so that it runs on sm_100a). Solution under test (KH_TEST_SOLUTION): solution.h
//   void LinearAttention(Q, K, V, O, slopes, B, H, N, D)
// Inputs are generated on the device; reference and solution write to different, NaN-poisoned
// buffers; every output element is compared.

#ifdef KH_TEST_SOLUTION
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "solution.h"   // must stay before the reference code / kh_verify.h
#endif

// Reference kernel: causal linear attention with per-head exponential decay (ThunderKittens,
// warp-level tensor-core MMA so that it runs on sm_100a). One CTA per (batch, head), 64-token chunks.
#include "kittens.cuh"

namespace khref_la {
using namespace kittens;

constexpr int CHUNK = 64;     // tokens per chunk
constexpr int HD = 128;       // head dimension (q/k/v/o)
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;

using tile_t = st_bf<CHUNK, HD>;               // 64 x 128 bf16, TMA tile for q / k / v / o
using gl_t = gl<bf16, -1, -1, -1, -1, tile_t>;

struct globals {
  gl_t q, k, v, o;
  const float *slopes;   // [H] decay rate per head, > 0
  int n;                 // sequence length (multiple of CHUNK)
};

// smem: q,k,v double buffered (96 KB) + o (16 KB) + decayed k (16 KB) + kv state bf16 (32 KB)
constexpr size_t SMEM_BYTES = 6 * sizeof(tile_t) + 2 * sizeof(tile_t) + sizeof(st_bf<HD, HD>) + 1024;

__global__ __launch_bounds__(NTHREADS, 1) void la_kernel(const __grid_constant__ globals g) {
  extern __shared__ alignment_dummy __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);
  tile_t(&q_s)[2] = al.allocate<tile_t, 2>();
  tile_t(&k_s)[2] = al.allocate<tile_t, 2>();
  tile_t(&v_s)[2] = al.allocate<tile_t, 2>();
  tile_t &o_s = al.allocate<tile_t>();
  tile_t &kd_s = al.allocate<tile_t>();
  st_bf<HD, HD> &kv_s = al.allocate<st_bf<HD, HD>>();
  __shared__ semaphore bar;

  const int batch = blockIdx.y, head = blockIdx.x;
  const int w = warpid(), lane = laneid();
  const int rr = w & 3;     // which 16-row slice of the chunk this warp produces
  const int ch = w >> 2;    // which 64-column half of the output / which half... of V
  const float slope = g.slopes[head];
  const int nblk = g.n / CHUNK;
  constexpr uint32_t TILE_BYTES = sizeof(tile_t);

  if (threadIdx.x == 0) {
    init_semaphore(bar, 0, 1);
    tma::expect_bytes(bar, 3 * TILE_BYTES);
    tma::load_async(q_s[0], g.q, {batch, head, 0, 0}, bar);
    tma::load_async(k_s[0], g.k, {batch, head, 0, 0}, bar);
    tma::load_async(v_s[0], g.v, {batch, head, 0, 0}, bar);
  }
  // zero the kv state (bf16 copy in smem) and the fp32 register state
  for (int i = threadIdx.x; i < HD * HD; i += NTHREADS) kv_s.data[i] = __float2bfloat16(0.f);
  rt_fl<16, 32> kv[4];   // state rows [16w, 16w+16), all 128 value columns
#pragma unroll
  for (int j = 0; j < 4; j++) warp::zero(kv[j]);
  __syncthreads();

  const float block_decay = __expf(-slope * float(CHUNK));
  int tic = 0;
  for (int blk = 0; blk < nblk; blk++, tic ^= 1) {
    wait(bar, tic);
    __syncthreads();   // everyone is past the previous chunk (buffers [tic^1] free)
    if (threadIdx.x == 0 && blk + 1 < nblk) {
      tma::expect_bytes(bar, 3 * TILE_BYTES);
      tma::load_async(q_s[tic ^ 1], g.q, {batch, head, blk + 1, 0}, bar);
      tma::load_async(k_s[tic ^ 1], g.k, {batch, head, blk + 1, 0}, bar);
      tma::load_async(v_s[tic ^ 1], g.v, {batch, head, blk + 1, 0}, bar);
    }

    // ---- S = Q_r K^T (16 x 64), decayed and causally masked -> bf16
    rt_fl<16, 64> S;
    warp::zero(S);
#pragma unroll
    for (int kk = 0; kk < HD / 32; kk++) {
      rt_bf<16, 32> a;
      rt_bf<64, 32> b;
      warp::load(a, q_s[tic].template subtile<16, 32>({rr, kk}));
      warp::load(b, k_s[tic].template subtile<64, 32>({0, kk}));
      warp::mma_ABt(S, a, b, S);
    }
    const int row0 = rr * 16;
    warp::apply(S, S, [&](int row, int col, float val) {
      int r = row0 + row;
      return col <= r ? val * __expf(-slope * float(r - col)) : 0.f;
    });
    rt_bf<16, 64> P;
    warp::copy(P, S);

    // ---- O_local = P V[:, half]
    rt_fl<16, 64> O, Oi;
    warp::zero(O);
    warp::zero(Oi);
    {
      rt_bf<64, 64, ducks::rt_layout::col> vb;
      warp::load(vb, v_s[tic].template subtile<64, 64>({0, ch}));
      warp::mma_AB(O, P, vb, O);
    }
    // ---- O_inter = decay_q * (Q_r @ KV[:, half])
#pragma unroll
    for (int kk = 0; kk < HD / 32; kk++) {
      rt_bf<16, 32> a;
      rt_bf<32, 64, ducks::rt_layout::col> b;
      warp::load(a, q_s[tic].template subtile<16, 32>({rr, kk}));
      warp::load(b, kv_s.template subtile<32, 64>({kk, ch}));
      warp::mma_AB(Oi, a, b, Oi);
    }
    warp::apply(Oi, Oi, [&](int row, int col, float val) {
      return val * __expf(-slope * float(row0 + row));
    });
    warp::add(O, O, Oi);
    { auto o_sub = o_s.template subtile<16, 64>({rr, ch}); warp::store(o_sub, O); }

    // ---- decayed K for the state update: kd[m, :] = k[m, :] * exp(-slope * (CHUNK - m))
    for (int idx = threadIdx.x; idx < CHUNK * HD / 2; idx += NTHREADS) {
      int m = idx / (HD / 2);
      int c2 = idx % (HD / 2);
      bf16_2 x = *reinterpret_cast<bf16_2 *>(&k_s[tic][{m, 2 * c2}]);
      float d = __expf(-slope * float(CHUNK - m));
      float2 xf = __bfloat1622float2(x);
      bf16_2 y = __floats2bfloat162_rn(xf.x * d, xf.y * d);
      *reinterpret_cast<bf16_2 *>(&kd_s[{m, 2 * c2}]) = y;
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();   // o_s, kd_s complete; everybody finished reading kv_s
    if (threadIdx.x == 0) tma::store_async(g.o, o_s, {batch, head, blk, 0});

    // ---- state update: KV_w = decay^64 * KV_w + Kd^T[16w..16w+16, :] @ V
#pragma unroll
    for (int j = 0; j < 4; j++) {
      warp::mul(kv[j], kv[j], block_decay);
      rt_bf<64, 16, ducks::rt_layout::col> a;
      rt_bf<64, 32, ducks::rt_layout::col> b;
      warp::load(a, kd_s.template subtile<64, 16>({0, w}));
      warp::load(b, v_s[tic].template subtile<64, 32>({0, j}));
      warp::mma_AtB(kv[j], a, b, kv[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) { auto kv_sub = kv_s.template subtile<16, 32>({w, j}); warp::store(kv_sub, kv[j]); }
    if (threadIdx.x == 0) tma::store_async_read_wait();
    __syncthreads();
  }
  if (threadIdx.x == 0) tma::store_async_wait();
}

inline globals make_globals(bf16 *q, bf16 *k, bf16 *v, bf16 *o, const float *slopes, int B, int H, int N) {
  size_t b_ = B, h_ = H, n_ = N, d_ = HD;
  gl_t qg{q, b_, h_, n_, d_}, kg{k, b_, h_, n_, d_}, vg{v, b_, h_, n_, d_}, og{o, b_, h_, n_, d_};
  return globals{qg, kg, vg, og, slopes, N};
}

inline void launch(globals const &g, int B, int H, cudaStream_t s = 0) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(la_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
    init = true;
  }
  la_kernel<<<dim3(H, B), NTHREADS, SMEM_BYTES, s>>>(g);
}
}  // namespace khref_la

#include "kh_verify.h"
#include <cstdio>
#include <string>
#include <vector>

namespace {
using namespace khref_la;

// Tolerances: |x - ref| <= max_rms * RMS(ref) + rel * |ref| for every element, and rel. Frobenius
// error <= fro. Measured with the shipped FP32 recurrence on all sizes: max|err|/rms <= 0.072,
// fro = 2.9e-3 (the reference and the solution both round to BF16 at different places; the
// reference itself deviates 2.7e-3 from an independent FP32 PyTorch computation). Bad variants
// (fro / elements failing at max_rms=0.2, rel=0.03; timed sizes): state rounded to BF16 every
// step 1.4e-2; Q,K rounded to FP8 4.1e-2; slope x1.1 6.5e-2; slope x0.9 7.5e-2; slope of the
// wrong head 0.29-0.55; state reset every 2048 / 512 / 64 tokens 0.16 / 0.35 / 0.75; current
// token excluded 0.2; no decay 13+; half of the outputs unwritten / empty solution: non-finite.
// (State kept in FP16 is as accurate as the reference: 3.4e-3, passes.)
KhTol kTol(0.2, 9.0e-3, 0.03);

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
  const size_t n_el = size_t(B) * H * N * HD;
  bf16 *q, *k, *v, *o_ref;
  float *slopes;
  KH_CHECK(cudaMalloc(&q, n_el * 2)); KH_CHECK(cudaMalloc(&k, n_el * 2));
  KH_CHECK(cudaMalloc(&v, n_el * 2)); KH_CHECK(cudaMalloc(&o_ref, n_el * 2));
  KH_CHECK(cudaMalloc(&slopes, H * sizeof(float)));
  // q.k has std ~ 1/sqrt(HD) and v has unit std, so O has std ~ 1
  // Q, K, V are regenerated with a new seed before every timed call and before the final re-check.
  unsigned seq = 1;
  auto refresh = [&](unsigned seed) {
    kh_fill_uniform(q, n_el, 11u + 7919u * seed, 0.153f);
    kh_fill_uniform(k, n_el, 22u + 7919u * seed, 0.153f);
    kh_fill_uniform(v, n_el, 33u + 7919u * seed, 1.7f);
  };
  refresh(seq);
  {
    std::vector<float> hs(H);
    for (int h = 0; h < H; h++)   // ALiBi / lightning-attention style slopes, jittered
      hs[h] = std::exp2(-8.0f * (h + 1) / H) * kh_uniform_pos(h, 44u, 0.9f, 1.1f);
    KH_CHECK(cudaMemcpy(slopes, hs.data(), H * sizeof(float), cudaMemcpyHostToDevice));
  }
  kh_poison(o_ref, n_el * 2);
  KH_CHECK(cudaDeviceSynchronize());

  globals g = make_globals(q, k, v, o_ref, slopes, B, H, N);
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
    dump(dir, "slopes.bin", slopes, H * 4); dump(dir, "o_ref.bin", o_ref, n_el * 2);
  }

#ifdef KH_TEST_SOLUTION
  bf16 *o_sol;
  KH_CHECK(cudaMalloc(&o_sol, n_el * 2));
  auto reset_sol = [&] { kh_poison(o_sol, n_el * 2); };
  auto call_sol = [&] {
    LinearAttention(reinterpret_cast<__nv_bfloat16 *>(q), reinterpret_cast<__nv_bfloat16 *>(k),
                    reinterpret_cast<__nv_bfloat16 *>(v), reinterpret_cast<__nv_bfloat16 *>(o_sol),
                    slopes, B, H, N, HD);
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
  cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o_ref); cudaFree(slopes);
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
      {"b8_n8k", 8, 32, 8192},
      {"b4_n16k", 4, 64, 16384},
      {"b1_n64k", 1, 64, 65536},
  };

  // Correctness-only cases (odd batch / head counts, short sequences); not timed.
  std::vector<Case> small = {
      {"odd_shape", 3, 5, 1536},
      {"short_seq", 2, 8, 64},
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
