#include "kittens.cuh"
#include "prototype.cuh"

#ifdef TORCH_COMPILE
#define TK_COMPILE_MAMBA2
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;
struct mamba2_fwd_layout {
	using q_tile   = st_bf<64, 128>;
	using k_tile   = st_bf<64, 128>;
	using v_tile   = st_bf<64, 64>;
	using o_tile   = st_bf<64, 64>;
    using a_vec    = sv_fl<64>; // decays
	using q_global = kittens::gl<bf16, -1, -1, -1, 128, q_tile>; // B, H, N, S
	using k_global = kittens::gl<bf16, -1, -1, -1, 128, k_tile>;
	using v_global = kittens::gl<bf16, -1, -1, -1, 64, v_tile>;
	using o_global = kittens::gl<bf16, -1, -1, -1, 64, o_tile>;
    using a_global = kittens::gl<float, -1, -1, 1, -1, a_vec>;
	struct globals { q_global Q; k_global K; v_global V; o_global O; a_global A; };
	struct input_block    {
        q_tile q;
        k_tile k;
        v_tile v[2];
        a_vec  a[2];
        a_vec  padding[6];
    };
    struct output_block {
        o_tile o[2];
    };
	struct scratch_block  {
        st_bf<64, 64> kv[2][2], k[2][2];
        a_vec         a_cumsum[2];
        a_vec         padding[6];
    };
    struct common_state {
        int batch, head;
    };
	struct consumer_state {
		rt_fl<16, 64> o_reg;
		rt_fl<16, 64> att_block;
		rt_bf<16, 64> att_block_mma;
        rt_fl<16, 64> local_decay;
        rt_bf<16, 128> q_reg, k_reg;
        rt_fl<16, 64> kv[2];
	};
};
struct mamba2_fwd_template {
	static constexpr int NUM_CONSUMER_WARPS=8, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=2, PRODUCER_BARRIER_ARRIVALS=1, CONSUMER_BARRIER_ARRIVALS=NUM_CONSUMER_WARPS/4;
	using layout = mamba2_fwd_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        // args.common.batch = blockIdx.y;
		// args.common.head = blockIdx.x*NUM_CONSUMER_WARPS/4; // stride 2 on heads
		// args.num_iters = args.task_iter == 0 ? args.globals.K.rows/layout::k_tile::rows : -1;
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
		args.common.batch = task_id / (args.globals.V.depth()/(NUM_CONSUMER_WARPS/4)); // batch = id / heads.
		task_id -= args.common.batch*(args.globals.V.depth()/(NUM_CONSUMER_WARPS/4));
		args.common.head = task_id*2; // stride 2 on heads
		args.num_iters = args.common.batch < args.globals.Q.batch() ? args.globals.K.rows()/layout::k_tile::rows : -1;
    }
	struct producer {
		__device__ static void setup(producer_setup_args<layout> args) {
			warpgroup::producer_registers();
		}
		__device__ static void load(producer_load_args<layout> args) {
			if(warpgroup::warpid() == args.iter%4) {
                warp::tma::expect(args.inputs_arrived, args.input.q, args.input.k, args.input.v[0], args.input.a[0], args.input.v[1], args.input.a[1]);
                warp::tma::load_async(args.input.q, args.globals.Q, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                warp::tma::load_async(args.input.k, args.globals.K, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                #pragma unroll
                for(int i = 0; i < NUM_CONSUMER_WARPS/4; i++) {
                    warp::tma::load_async(args.input.v[i], args.globals.V, {args.common.batch,  args.common.head+i, args.iter, 0}, args.inputs_arrived);
                    warp::tma::load_async(args.input.a[i], args.globals.A, {args.common.batch,  args.common.head+i, 0, args.iter}, args.inputs_arrived);
                }
                __syncwarp();
            }
		}
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                #pragma unroll
                for(int i = 0; i < NUM_CONSUMER_WARPS/4; i++) {
                    warp::tma::store_async(args.globals.O, args.output.o[i], {args.common.batch, args.common.head+i, args.iter, 0});
                }
                warp::tma::store_async_read_wait();
                __syncwarp();
                if(laneid() == 0) arrive(args.outputs_finished);
                __syncwarp();
            }
        }
	};
	struct consumer {
		__device__ static void setup(consumer_setup_args<layout> args) {
			warpgroup::consumer_registers<NUM_CONSUMER_WARPS/WARPGROUP_WARPS>();
            warp::zero(args.state.kv[0]);warp::zero(args.state.kv[1]);
		}
		__device__ static void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::groupid();
            // Start by doing cumsum into shared memory
            warpgroup::sync(warpgroupid);
            warpgroup::copy(args.scratch.a_cumsum[warpgroupid], args.input.a[warpgroupid]);
            warpgroup::sync(warpgroupid);
            if(warpgroup::warpid() <= 1) {
                int tid = warpgroup::laneid();
                // Perform the prefix sum (Hillis-Steele scan)
                for (int offset = 1; offset < 64; offset *= 2) {
                    float temp = (tid >= offset) ? args.scratch.a_cumsum[warpgroupid][tid - offset] : 0.0f;
                    group<2>::sync(warpgroupid+2);
                    args.scratch.a_cumsum[warpgroupid][tid] += temp;
                    group<2>::sync(warpgroupid+2);
                }
            }
            warpgroup::sync(warpgroupid); // cumulative sum done
            // Calculate decays
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                int base_col = i*16 + (laneid()%4)*2;
                args.state.local_decay.tiles[0][i].data[0].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[0].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[1].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[1].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[2].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[2].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
                args.state.local_decay.tiles[0][i].data[3].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[3].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
            }
            warp::exp(args.state.local_decay, args.state.local_decay);
            // causal mask
            #pragma unroll
            for(int i = 0; i < 4; i++) { // causal mask
                auto &decay_subtile = reinterpret_cast<rt_fl<16,16>&>(args.state.local_decay.tiles[0][i]);
                if      (i >  warpgroup::warpid()) { warp::zero       (decay_subtile); }
                else if (i == warpgroup::warpid()) { warp::make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero()); }
            }
      		// A = Q @ K.T
            warpgroup::load(args.state.q_reg, args.input.q); // we need this later, anyways
			warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k);
			warpgroup::mma_async_wait();
            warp::mul(args.state.att_block, args.state.att_block, args.state.local_decay);
            warp::copy(args.state.att_block_mma, args.state.att_block);
            warpgroup::mm_AB(args.state.o_reg, args.state.att_block_mma, args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();
            // // multiply q by decays
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row +8]));
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    args.state.q_reg.tiles[0][i].data[0].x *= top;
                    args.state.q_reg.tiles[0][i].data[0].y *= top;
                    args.state.q_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.q_reg.tiles[0][i].data[2].x *= top;
                    args.state.q_reg.tiles[0][i].data[2].y *= top;
                    args.state.q_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            rt_bf<16,64> q_low,q_high;
            for(int c=0;c<4;++c) for(int j=0;j<4;++j) {
              q_low.tiles[0][c].data[j]=args.state.q_reg.tiles[0][c].data[j];
              q_high.tiles[0][c].data[j]=args.state.q_reg.tiles[0][c+4].data[j];
            }
            warpgroup::store(args.scratch.kv[warpgroupid][0],args.state.kv[0]);
            warpgroup::store(args.scratch.kv[warpgroupid][1],args.state.kv[1]);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AB(args.state.o_reg,q_low,args.scratch.kv[warpgroupid][0]);
            warpgroup::mma_async_wait();
            warpgroup::mma_AB(args.state.o_reg,q_high,args.scratch.kv[warpgroupid][1]);
            warpgroup::mma_async_wait();
            warpgroup::store(args.output.o[warpgroupid], args.state.o_reg);
            warpgroup::sync(warpgroupid);
            float last_decay = args.scratch.a_cumsum[warpgroupid][args.scratch.a_cumsum[warpgroupid].length-1]; // last element
            float total_decay = expf(last_decay);
            warp::mul(args.state.kv[0],args.state.kv[0],total_decay);
            warp::mul(args.state.kv[1],args.state.kv[1],total_decay); // decay kv
            warpgroup::load(args.state.k_reg, args.input.k); // multiply k's by decays
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row +8]));
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    args.state.k_reg.tiles[0][i].data[0].x *= top;
                    args.state.k_reg.tiles[0][i].data[0].y *= top;
                    args.state.k_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.k_reg.tiles[0][i].data[2].x *= top;
                    args.state.k_reg.tiles[0][i].data[2].y *= top;
                    args.state.k_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            rt_bf<16,64> k_low,k_high;
            for(int c=0;c<4;++c) for(int j=0;j<4;++j) {
              k_low.tiles[0][c].data[j]=args.state.k_reg.tiles[0][c].data[j];
              k_high.tiles[0][c].data[j]=args.state.k_reg.tiles[0][c+4].data[j];
            }
            warpgroup::store(args.scratch.k[warpgroupid][0],k_low);
            warpgroup::store(args.scratch.k[warpgroupid][1],k_high);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AtB(args.state.kv[0],args.scratch.k[warpgroupid][0],args.input.v[warpgroupid]);
            warpgroup::mma_AtB(args.state.kv[1],args.scratch.k[warpgroupid][1],args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) {
                arrive(args.outputs_arrived);
                arrive(args.inputs_finished);
            }
            __syncwarp();
		}
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(warpgroup::laneid() == 0) arrive(args.finish_finished);
            __syncwarp();
        }
	};
};


#include "harness.impl"
