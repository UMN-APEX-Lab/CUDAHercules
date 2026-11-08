/*
  Task tk_bf16_gemm: BF16 GEMM on Blackwell (sm_100a / B200), C = A * B^T.

  Reference: the ThunderKittens Blackwell BF16 GEMM (kernels/gemm/bf16_b200: 2-CTA clusters, tcgen05
  MMA, TMA, cluster launch control) with one template configuration chosen per problem shape from a
  sweep on B200. It differs from the upstream kernel only in the tensor-memory release handshake
  (both CTAs of a cluster synchronise before tensor memory is deprovisioned; without it the kernel can
  fault with "unspecified launch failure").

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    void Bf16Gemm(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K);

    A: row-major [M, K], B: row-major [N, K] (the GEMM is C = A * B^T), C: row-major [M, N].

  Verification: A and B are full-mantissa random BF16; the reference and the solution write to
  different buffers that are pre-filled with NaN; every output element must be finite and within
  tolerance of the reference. The solution is re-checked after the timing loop.
*/

#ifdef KH_TEST_SOLUTION
#include "solution.h"   // first: the solution must not see ThunderKittens or the harness helpers
#endif

#include "kittens.cuh"
#include "kh_verify.h"

#include <functional>
#include <memory>
#include <vector>

using namespace kittens;

#ifdef KH_TEST_SOLUTION
#define KH_PASS_MSG "Passed"
#else
#define KH_PASS_MSG "Reference ran (no solution under test)"
#endif

namespace bf16_gemm {
template <int _Mb, int _Nb, int _Kb, int _SUPERGROUP_SIZE, bool _OVERLAP_MMA_EPI, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH>
struct config {
    static_assert(_Mb == 256, "Mb must be 256");
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0, "Nb must be 16, 32, ..., 256");
    static_assert(_Kb >= 16 && _Kb % 16 == 0, "Kb must be a multiple of 16");
    static_assert(_SUPERGROUP_SIZE >= 1 && _SUPERGROUP_SIZE <= 16, "SUPERGROUP_SIZE must be 1-16");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 16, "LOAD_PIPE_DEPTH must be 1-16");
    static_assert(_EPI_PIPE_DEPTH >= 1 && _EPI_PIPE_DEPTH <= 16, "EPI_PIPE_DEPTH must be 1-16");

    static constexpr int Mb = _Mb; // per MMA
    static constexpr int Nb = _Nb; // per MMA
    static constexpr int Kb = _Kb;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr bool OVERLAP_MMA_EPI = _OVERLAP_MMA_EPI;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = OVERLAP_MMA_EPI ? 2 : 1;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = OVERLAP_MMA_EPI ? 1 : 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
};

template <typename C>
struct globals {
    using a_tile = st_bf<C::Mb/2, C::Kb>;
    using b_tile = st_bf<C::Nb/2, C::Kb>;
    using d_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() { return dim3(d.rows()/(C::NUM_CONSUMERS*C::Mb/2) * d.cols()/C::Nb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        constexpr size_t _dynamic_shared_memory = sizeof(a_tile) * C::LOAD_PIPE_DEPTH * C::NUM_CONSUMERS +
                                                  sizeof(b_tile) * C::LOAD_PIPE_DEPTH +
                                                  sizeof(d_tile) * C::NUM_D_TILES * C::NUM_CONSUMERS + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<typename G::a_tile>();
        g.b.template prefetch_tma<typename G::b_tile>();
        g.d.template prefetch_tma<typename G::d_tile>();
    }

    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / C::Kb;
    const int rblks = g.d.rows() / (C::Mb * C::NUM_CONSUMERS);
    const int cblks = g.d.cols() / C::Nb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[C::LOAD_PIPE_DEPTH][C::NUM_CONSUMERS] = al.allocate<G::a_tile, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    typename G::b_tile (&b_smem)[C::LOAD_PIPE_DEPTH]                   = al.allocate<G::b_tile, C::LOAD_PIPE_DEPTH>();
    typename G::d_tile (&d_smem)[C::NUM_CONSUMERS][C::NUM_D_TILES]     = al.allocate<G::d_tile, C::NUM_CONSUMERS, C::NUM_D_TILES>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, C::Mb/2, C::Nb>;

    __shared__ uint32_t tmem_addr;
    __shared__ clc::handle clc_handle[C::CLC_PIPE_DEPTH];
    __shared__ semaphore tmem_provisioned, tmem_finished, schedule_arrived[C::CLC_PIPE_DEPTH], schedule_finished[C::CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH], inputs_finished[C::LOAD_PIPE_DEPTH], outputs_arrived[C::NUM_CONSUMERS], outputs_finished[C::MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, (2+C::NUM_CONSUMERS)*C::CLUSTER_SIZE+C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, C::NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE*C::NUM_CONSUMERS);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    #pragma unroll
                    for (int i = 0; i < C::NUM_CONSUMERS; i++)
                        tma::cluster::load_async(a_smem[input_ring][i], g.a, {(tile_coord.x*2+cta_rank)*C::NUM_CONSUMERS+i, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b, {tile_coord.y*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
                else break;
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0) {
                    wait(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], ((task_iter+C::CLC_PIPE_DEPTH)/C::CLC_PIPE_DEPTH)%2);
                    clc::schedule(clc_handle[task_iter%C::CLC_PIPE_DEPTH], schedule_arrived[task_iter%C::CLC_PIPE_DEPTH]);
                }
                tma::expect_bytes(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter%C::CLC_PIPE_DEPTH]));
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        } else if (cta_rank == 0 && warpgroup::warpid() < C::NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt[C::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
                if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::warpid())*C::Nb);
                else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::warpid())*C::Nb);
            }
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                wait(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], ((task_iter+C::MMA_PIPE_DEPTH)/C::MMA_PIPE_DEPTH)%2);
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::expect_bytes(inputs_arrived[input_ring], (C::CLUSTER_SIZE*C::NUM_CONSUMERS*sizeof(G::a_tile) + 2*sizeof(G::b_tile))/C::NUM_CONSUMERS);
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    if (idx == 0) mm2_ABt (d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    update_phasebit<0>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                detail::tcgen05::commit<C::CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
                if (!schedule.success) break;
            }
        }
    }
    else {
        using epilogue_group = group<WARPGROUP_WARPS*C::NUM_CONSUMERS>;
        if constexpr (!C::OVERLAP_MMA_EPI)
            warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();
        if (epilogue_group::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt[C::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::groupid())*C::Nb);
            else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::groupid())*C::Nb);
        }
        int2 tile_coord, next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
        for(int task_iter = 0; true; task_iter++) {
            tile_coord = next_tile_coord;
            wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
            auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
            if (schedule.success) next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
            wait(outputs_arrived[warpgroup::groupid()], task_iter%2);
            if constexpr (C::OVERLAP_MMA_EPI) {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg;
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::load_async(d_reg, d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        warpgroup::sync(warpgroup::groupid()+1);
                        if (!schedule.success) warpgroup::pdl::arrive();
                        warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                    }
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            } else {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                    warpgroup::load_async(d_reg[i], d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                tensor_load_wait();
                warpgroup::sync(warpgroup::groupid()+1);
                if (!schedule.success) warpgroup::pdl::arrive();
                warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg[i]);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            }
            if (!schedule.success) break;
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) {
            // Both CTAs of the cluster must be done with tensor memory before it is released
            // (the 2-CTA MMA of CTA 0 also writes CTA 1's tensor memory).
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_rank);
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

}  // namespace bf16_gemm


struct Buffers { __nv_bfloat16 *A = nullptr, *B = nullptr, *D = nullptr; };

static void alloc_and_fill(Buffers &b, int M, int N, int K) {
  size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;
  KH_CHECK(cudaMalloc(&b.A, nA * 2));
  KH_CHECK(cudaMalloc(&b.B, nB * 2));
  KH_CHECK(cudaMalloc(&b.D, nC * 2));
  kh_fill_uniform(b.A, nA, 11u);
  kh_fill_uniform(b.B, nB, 22u);
}
/// Regenerates the per-call input A (new random values) with a new seed.
static void refill_inputs(Buffers &b, int M, int N, int K, unsigned seed) {
  (void)N;
  kh_fill_uniform(b.A, size_t(M) * K, seed);
}
static void free_buffers(Buffers &b) { cudaFree(b.A); cudaFree(b.B); cudaFree(b.D); }

// Reference configurations, selected per shape below (see the KH_SWEEP development target).
using Cfg0 = bf16_gemm::config<256, 256, 64, 8, false, 4, 8>;
using Cfg1 = bf16_gemm::config<256, 256, 64, 4, false, 4, 8>;
using Cfg2 = bf16_gemm::config<256, 256, 128, 8, true, 3, 4>;
using Cfg3 = bf16_gemm::config<256, 128, 128, 2, true, 4, 2>;
using Cfg4 = bf16_gemm::config<256, 256, 64, 16, false, 4, 8>;
constexpr int kNumCfg = 5;

/// One prepared reference launch: tensor-map (TMA descriptor) construction for the `globals` and the
/// launch configuration are done in the constructor, so run() is just the kernel launch.
struct RefGemm {
  virtual ~RefGemm() {}
  virtual void run() = 0;
  virtual int row_multiple() const = 0;
  virtual int col_multiple() const = 0;
  virtual int k_multiple() const = 0;
};


template <class C>
struct TkRef : RefGemm {
  using G = bf16_gemm::globals<C>;
  G g;
  std::unique_ptr<LaunchConfig<true, true>> lc;
  TkRef(Buffers const &b, int M, int N, int K)
      : g{typename G::a_gl{b.A, nullptr, nullptr, size_t(M), size_t(K)},
          typename G::b_gl{b.B, nullptr, nullptr, size_t(N), size_t(K)},
          typename G::d_gl{b.D, nullptr, nullptr, size_t(M), size_t(N)}} {
    KH_CHECK(cudaFuncSetAttribute(bf16_gemm::kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  g.dynamic_shared_memory()));
    lc.reset(new LaunchConfig<true, true>(g.grid(), g.block(), g.dynamic_shared_memory(),
                                          cudaStream_t(0), C::CLUSTER_SIZE));
  }
  void run() override { KH_CHECK(cudaLaunchKernelEx(*lc, bf16_gemm::kernel<C>, g)); }
  int row_multiple() const override { return C::NUM_CONSUMERS * C::Mb; }
  int col_multiple() const override { return C::Nb; }
  int k_multiple() const override { return C::Kb; }
};


static std::unique_ptr<RefGemm> make_ref(int cfg, Buffers const &b, int M, int N, int K) {
  switch (cfg) {
    case 0: return std::make_unique<TkRef<Cfg0>>(b, M, N, K);
    case 1: return std::make_unique<TkRef<Cfg1>>(b, M, N, K);
    case 2: return std::make_unique<TkRef<Cfg2>>(b, M, N, K);
    case 3: return std::make_unique<TkRef<Cfg3>>(b, M, N, K);
    case 4: return std::make_unique<TkRef<Cfg4>>(b, M, N, K);
    default: std::fprintf(stderr, "bad config index\n"); std::exit(2);
  }
}

// TOLERANCE_PLACEHOLDER
// Tolerances (measured; BF16 output, relative Frobenius error vs the reference over all shapes below).
// The reference's tensor-core accumulation order differs from a sequential FP32 kernel: the shipped
// baseline measures 2.0e-4 .. 4.0e-4 (max|err|/rms <= 3.6e-2 = single BF16-ulp flips on the largest
// elements, always inside the per-element rel. term). Bounds: Frobenius 1.5e-3 (3.7x), elements
// 1e-3*RMS + 1.6e-2*|ref| (two BF16 ulps). Bad variants, all FAIL on every shape: BF16 partial sums
// every 512 K 2.6e-3, FP16 accumulation 1.3e-2 .. 2.5e-2, FP8 E4M3 inputs 3.4e-2, skipped K block
// 6.7e-2 .. 1.3e-1, BF16 accumulation 1.0e-1 .. 1.9e-1, B not transposed 1.4, half the outputs
// unwritten (NaN), empty solution (NaN).
static KhTol kTol(1.0e-3, 1.5e-3, 1.6e-2);

struct Shape3 { const char *label; int M, N, K, cfg; };

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Shape3 const &s, bool timed, int trials) {
  int M = s.M, N = s.N, K = s.K;
  size_t nC = size_t(M) * N;
  Buffers b;
  alloc_and_fill(b, M, N, K);
  __nv_bfloat16 *D_ref = b.D;
  kh_poison(D_ref, nC * 2);
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(s.cfg, b, M, N, K);
  if (M % ref->row_multiple() || N % ref->col_multiple() || K % ref->k_multiple()) {
    std::fprintf(stderr, "shape %dx%dx%d is not a multiple of the reference tile (%d,%d,%d)\n", M, N, K,
                 ref->row_multiple(), ref->col_multiple(), ref->k_multiple());
#ifdef KH_SWEEP
    ref.reset(); free_buffers(b);
    return true;
#else
    std::exit(2);
#endif
  }
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {   // the reference must have written every element (it was pre-filled with NaN)
    KhResult r = kh_compare(D_ref, D_ref, nC, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite << ")" << std::endl;
      ok = false;
    }
  }

#ifdef KH_TEST_SOLUTION
  __nv_bfloat16 *D_sol = nullptr;
  KH_CHECK(cudaMalloc(&D_sol, nC * 2));
  auto reset_sol = [&] { kh_poison(D_sol, nC * 2); };
  auto call_sol = [&] { Bf16Gemm(b.A, b.B, D_sol, M, N, K); KH_CHECK(cudaGetLastError()); };
  if (ok) {
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(D_sol, D_ref, nC, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
  }
#endif

  if (ok && timed) {   // the fresh-input re-check also runs when timing is disabled (KH_BENCHMARK=0)
    // The activations A (including their block scales when quantised) are regenerated with a new seed,
    // outside the timed region and before the L2 flush, before every call of both the reference and the
    // solution, so a result or an A-dependent intermediate cannot be memoized across calls; the weights B
    // stay fixed. The reference output is re-poisoned before every call.
    unsigned seed = 1000u;
    auto refresh = [&] { refill_inputs(b, M, N, K, seed++); };
    if (trials > 0) {
    KhTime tr = kh_time([&] { ref->run(); }, trials, [&] { kh_poison(D_ref, nC * 2); refresh(); });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { reset_sol(); refresh(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#endif
    }
#ifdef KH_TEST_SOLUTION
    // Re-check after timing with yet another fresh input: recompute the reference for it, re-poison the
    // solution output, and require the solution to produce the right answer from scratch.
    refresh();
    kh_poison(D_ref, nC * 2);
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(D_sol, D_ref, nC, kTol);
    kh_print_result("solution (re-check after timing, fresh input)", r, kTol);
    ok = r.pass;
#endif
  }

#ifdef KH_TEST_SOLUTION
  cudaFree(D_sol);
#endif
  ref.reset();
  free_buffers(b);
  return ok;
}

int main(int argc, char **argv) {
  cudaDeviceProp props;
  KH_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This task requires a Blackwell sm_100 GPU (B200)." << std::endl;
    return 1;
  }
  int trials = kh_trials();
  bool const keep_going = std::getenv("KH_KEEP_GOING") != nullptr;   // development aid
  bool failed = false;

  // Llama-3-70B linear layers (hidden 8192, 64 q / 8 kv heads x 128 -> fused QKV 10240, FFN 28672 ->
  // fused gate+up 57344); M = tokens. The reference tile is 256 rows per CTA pair, so the smallest
  // supported batch is 256 tokens (a large decode / speculative batch); the rest are 8K / 16K-token
  // prefill chunks.
  std::vector<Shape3> shapes = {
      {"decode_batch256_qkv", 256, 10240, 8192, 2},
      {"decode_batch256_mlp_down", 256, 8192, 28672, 3},
      {"prefill_qkv", 8192, 10240, 8192, 0},
      {"prefill_o_proj", 8192, 8192, 8192, 0},
      {"prefill_mlp_up", 8192, 57344, 8192, 0},
      {"prefill_mlp_down", 8192, 8192, 28672, 1},
      {"prefill16k_o_proj", 16384, 8192, 8192, 4}
  };

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every shape.
  for (auto s : shapes) {
    for (int cfg = 0; cfg < kNumCfg; ++cfg) {
      s.cfg = cfg;
      std::printf("\n=== %s cfg=%d: M=%d, N=%d, K=%d ===\n", s.label, cfg, s.M, s.N, s.K);
      run_case(s, true, 10);
    }
  }
  return 0;
#endif

  if (argc >= 4) {   // ./test M N K : single custom shape
    Shape3 s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[1]) <= 512 ? 2 : 0};
    std::printf("\n=== custom: M=%d, N=%d, K=%d ===\n", s.M, s.N, s.K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << KH_PASS_MSG << std::endl;
    return 0;
  }

  for (auto const &s : shapes) {
    std::printf("\n=== %s: M=%d, N=%d, K=%d ===\n", s.label, s.M, s.N, s.K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << KH_PASS_MSG << std::endl;
  }
  return failed ? 1 : 0;
}
