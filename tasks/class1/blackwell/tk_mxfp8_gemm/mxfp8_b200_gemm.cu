/*
  Task tk_mxfp8_gemm: MXFP8 block-scaled GEMM on Blackwell (sm_100a / B200), D = (A*sA) * (B*sB)^T.

  Reference: the ThunderKittens Blackwell MXFP8 GEMM (kernels/gemm/mxfp8_b200: 2-CTA clusters,
  tcgen05 block-scaled MMA, TMA) with one template configuration chosen per problem shape from a
  sweep on B200. It differs from the upstream kernel only in the tensor-memory release handshake
  (both CTAs of a cluster synchronise before tensor memory is deprovisioned).

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    void Mxfp8Gemm(__nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, __nv_fp8_e8m0 *A_scale, __nv_fp8_e8m0 *B_scale,
                   __nv_bfloat16 *C, int M, int N, int K);

    A: row-major [M, K] FP8 E4M3, B: row-major [N, K] FP8 E4M3, C: row-major [M, N] BF16.
    One E8M0 scale per 32 consecutive K elements of a row of A / B (the OCP MX block size), stored in
    the 128x4 swizzled tile layout of tcgen05 / cuBLAS: for the scale of (row r, k-block kb = k / 32):
      index = ((r / 128) * (K / 128) + kb / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + kb % 4.

  Verification: A / B are random E4M3 values over the whole finite range with random mantissas,
  scales are random powers of two (2^-3 .. 2^3); the reference and the solution write to different
  buffers pre-filled with NaN; every output element must be finite and within tolerance of the
  reference. The solution is re-checked after the timing loop.
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

namespace mxfp8_gemm {

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_LOAD_PIPE_DEPTH > 0, "LOAD_PIPE_DEPTH must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH > 0, "EPI_PIPE_DEPTH must be greater than 0");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2, "NUM_D_TILES must be at least 2 if EPI_PIPE_DEPTH > 1");

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 128;
    static constexpr int B_SC_SIZE = Nb/128;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
};

template <typename C>
struct globals {
    using A_fp8_tile = st_fp8e4m3<C::Mb / 2, C::Kb>;
    using A_sc_tile  = st_fp8e8m0<32, 16, false>;
    using B_fp8_tile = st_fp8e4m3<C::Nb / 2, C::Kb>;
    using B_sc_tile  = st_fp8e8m0<32, 16, false>;
    using D_tile     = st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_gl    = gl<fp8e4m3,  1,  1, -1, -1, A_fp8_tile>;
    using A_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;
    using B_gl    = gl<fp8e4m3,  1,  1, -1, -1, B_fp8_tile>;
    using B_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, B_sc_tile>;
    using D_gl    = gl<bf16,     1,  1, -1, -1, D_tile>;

    A_gl A;       // M x K
    A_sc_gl A_sc; // (M // 128) x (K // 128) x 32 x 16
    B_gl B;       // N x K
    B_sc_gl B_sc; // (N // 128) x (K // 128) x 32 x 16
    D_gl D;       // M x N

    struct input_tiles_t {
        A_fp8_tile A;
        B_fp8_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        return dim3(min((D.rows()/(C::Mb/2))*(D.cols()/C::Nb), num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp8_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp8_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int num_col_blocks = g.D.cols() / C::Nb;
    const int num_blocks = num_col_blocks * num_row_blocks;
    const int num_iters_per_block = g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    // Declare tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                #pragma unroll 2
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            // Load input scales to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                #pragma unroll 2
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx * 2 + cta_id, i, 0, 0}, scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    if constexpr (C::B_SC_SIZE == 2) tma::cluster::load_async(input_scales[stage].B[cta_id], g.B_sc, {col_block_idx * 2 + cta_id, i, 0, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    else if (cta_id == 0)            tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // Launch tensor core matrix multiply
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::LOAD_PIPE_DEPTH>>(384);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                #pragma unroll 2
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*16);
                    load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A);
                    auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*32);
                    load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[0]);
                    if constexpr (C::B_SC_SIZE == 2) {
                        auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*32+16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[1]);
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                        inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                update_phasebit<1>(phasebits, 0);
                tensor_commit<2>(outputs_arrived);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        // Consumer group
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Load the output from tensor memory into registers and store to HBM
            if constexpr (C::OVERLAP_EPI) {
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg;
                    warpgroup::load_async(D_reg, out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0
                    }
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                }
            } else {
                rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                    warpgroup::load_async(D_reg[i], out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                }
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            // Both CTAs of the cluster must be done with tensor memory before it is released.
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_id);
            wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }
}

} // namespace mxfp8_gemm

namespace mxfp8_gemm {
template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_entrypoint(const __grid_constant__ globals<C> g) { kernel<C>(g); }
}  // namespace mxfp8_gemm


struct Buffers {
  __nv_fp8_e4m3 *A = nullptr, *B = nullptr;
  __nv_fp8_e8m0 *Asc = nullptr, *Bsc = nullptr;
  __nv_bfloat16 *D = nullptr;
};

/// Random E8M0 scale bytes: exponent byte uniform in [lo, hi] (value 2^(byte - 127)).
__global__ void kh_fill_e8m0(uint8_t *p, size_t n, unsigned int seed, int lo, int hi) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = uint8_t(lo + int(kh_hash64(i, seed) % unsigned(hi - lo + 1)));
}

static void alloc_and_fill(Buffers &b, int M, int N, int K) {
  size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;
  KH_CHECK(cudaMalloc(&b.A, nA));
  KH_CHECK(cudaMalloc(&b.B, nB));
  KH_CHECK(cudaMalloc(&b.Asc, nA / 32));
  KH_CHECK(cudaMalloc(&b.Bsc, nB / 32));
  KH_CHECK(cudaMalloc(&b.D, nC * 2));
  kh_fill_uniform(b.A, nA, 11u, 448.0f);
  kh_fill_uniform(b.B, nB, 22u, 448.0f);
  kh_fill_e8m0<<<2048, 256>>>(reinterpret_cast<uint8_t *>(b.Asc), nA / 32, 33u, 124, 130);
  kh_fill_e8m0<<<2048, 256>>>(reinterpret_cast<uint8_t *>(b.Bsc), nB / 32, 44u, 124, 130);
  KH_CHECK(cudaGetLastError());
}
/// Regenerates the per-call input A (E4M3 data and its E8M0 block scales) with a new seed.
static void refill_inputs(Buffers &b, int M, int N, int K, unsigned seed) {
  (void)N;
  size_t nA = size_t(M) * K;
  kh_fill_uniform(b.A, nA, seed, 448.0f);
  kh_fill_e8m0<<<2048, 256>>>(reinterpret_cast<uint8_t *>(b.Asc), nA / 32, seed + 7919u, 124, 130);
  KH_CHECK(cudaGetLastError());
}
static void free_buffers(Buffers &b) {
  cudaFree(b.A); cudaFree(b.B); cudaFree(b.Asc); cudaFree(b.Bsc); cudaFree(b.D);
}

// Reference configurations, selected per shape below (see the KH_SWEEP development target).
using Cfg0 = mxfp8_gemm::config<256, 6, 16, 12, 4, false>;
using Cfg1 = mxfp8_gemm::config<256, 5, 8, 8, 2, false>;
using Cfg2 = mxfp8_gemm::config<256, 6, 16, 16, 4, false>;
using Cfg3 = mxfp8_gemm::config<128, 6, 8, 8, 2, false>;
constexpr int kNumCfg = 4;

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
  using G = mxfp8_gemm::globals<C>;
  G g;
  std::unique_ptr<LaunchConfig<true, true>> lc;
  TkRef(Buffers const &b, int M, int N, int K)
      : g{typename G::A_gl{b.A, nullptr, nullptr, size_t(M), size_t(K)},
          typename G::A_sc_gl{b.Asc, size_t(M / 128), size_t(K / 128), nullptr, nullptr},
          typename G::B_gl{b.B, nullptr, nullptr, size_t(N), size_t(K)},
          typename G::B_sc_gl{b.Bsc, size_t(N / 128), size_t(K / 128), nullptr, nullptr},
          typename G::D_gl{b.D, nullptr, nullptr, size_t(M), size_t(N)}} {
    KH_CHECK(cudaFuncSetAttribute(mxfp8_gemm::kernel_entrypoint<C>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    lc.reset(new LaunchConfig<true, true>(g.grid(), g.block(), g.dynamic_shared_memory(),
                                          cudaStream_t(0), C::CLUSTER_SIZE));
  }
  void run() override { KH_CHECK(cudaLaunchKernelEx(*lc, mxfp8_gemm::kernel_entrypoint<C>, g)); }
  int row_multiple() const override { return C::Mb; }
  int col_multiple() const override { return C::Nb; }
  int k_multiple() const override { return C::Kb; }
};


static std::unique_ptr<RefGemm> make_ref(int cfg, Buffers const &b, int M, int N, int K) {
  switch (cfg) {
    case 0: return std::make_unique<TkRef<Cfg0>>(b, M, N, K);
    case 1: return std::make_unique<TkRef<Cfg1>>(b, M, N, K);
    case 2: return std::make_unique<TkRef<Cfg2>>(b, M, N, K);
    case 3: return std::make_unique<TkRef<Cfg3>>(b, M, N, K);
    default: std::fprintf(stderr, "bad config index\n"); std::exit(2);
  }
}

// TOLERANCE_PLACEHOLDER
// Tolerances (measured; BF16 output, relative Frobenius error vs the reference over all shapes below).
// The shipped baseline (sequential FP32 FMA over exact products) measures 1.0e-4 .. 2.1e-4
// (max|err|/rms <= 3.0e-2 = single BF16-ulp flips, inside the per-element rel. term). Bounds:
// Frobenius 1.0e-3 (4.7x), elements 1e-3*RMS + 1.6e-2*|ref| (two BF16 ulps). Bad variants, all FAIL on
// every shape: skipped K block 6.7e-2 .. 1.3e-1, BF16 accumulation 6.7e-2 .. 1.1e-1, E5M2 inputs
// 8.9e-2, FP16 accumulation (overflows to Inf), ignored A / B scales 8.4e-1, scale layout not swizzled
// 1.3, scale block 128 instead of 32 1.1, half the outputs unwritten (NaN), empty solution (NaN).
static KhTol kTol(1.0e-3, 1.0e-3, 1.6e-2);

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
  auto call_sol = [&] { Mxfp8Gemm(b.A, b.B, b.Asc, b.Bsc, D_sol, M, N, K); KH_CHECK(cudaGetLastError()); };
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

  // Llama-3-70B linear layers in MXFP8 (hidden 8192, fused QKV 10240, FFN 28672 -> fused gate+up
  // 57344); M = tokens. The reference tile is 256 rows per CTA pair, so the smallest supported batch
  // is 256 tokens; the rest are 8K / 16K-token prefill chunks.
  std::vector<Shape3> shapes = {
      {"decode_batch256_qkv", 256, 10240, 8192, 0},
      {"decode_batch256_mlp_down", 256, 8192, 28672, 3},
      {"prefill_qkv", 8192, 10240, 8192, 2},
      {"prefill_o_proj", 8192, 8192, 8192, 0},
      {"prefill_mlp_up", 8192, 57344, 8192, 2},
      {"prefill_mlp_down", 8192, 8192, 28672, 1},
      {"prefill16k_o_proj", 16384, 8192, 8192, 2}
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
    Shape3 s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[1]) <= 512 ? 0 : 2};
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
