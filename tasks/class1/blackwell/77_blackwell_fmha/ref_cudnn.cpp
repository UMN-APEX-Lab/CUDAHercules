// cuDNN SDPA reference for the FMHA task (see ref_cudnn.h).
#include "ref_cudnn.h"

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_map>

namespace fe = cudnn_frontend;

struct KhCudnnSdpa {
  cudnnHandle_t handle = nullptr;
  std::shared_ptr<fe::graph::Graph> graph;
  std::unordered_map<int64_t, void *> pack;
  void *workspace = nullptr;
};

static void die(char const *what, std::string const &msg) {
  std::fprintf(stderr, "cuDNN reference failed (%s): %s\n", what, msg.c_str());
  std::exit(2);
}

KhCudnnSdpa *khcudnn_create(int B, int Hq, int Hkv, int S, int D, bool causal, int dtype,
                            void const *Q, void const *K, void const *V, void *O, float *LSE) {
  auto *p = new KhCudnnSdpa();
  if (cudnnCreate(&p->handle) != CUDNN_STATUS_SUCCESS) die("cudnnCreate", "");
  int64_t b = B, hq = Hq, hk = Hkv, s = S, d = D;
  auto g = std::make_shared<fe::graph::Graph>();
  g->set_io_data_type(dtype ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  // Logical dims are BHSD; the strides describe the BSHD memory layout of the task.
  auto q = g->tensor(fe::graph::Tensor_attributes().set_name("Q").set_uid(1).set_dim({b, hq, s, d}).set_stride({s * hq * d, d, hq * d, 1}));
  auto k = g->tensor(fe::graph::Tensor_attributes().set_name("K").set_uid(2).set_dim({b, hk, s, d}).set_stride({s * hk * d, d, hk * d, 1}));
  auto v = g->tensor(fe::graph::Tensor_attributes().set_name("V").set_uid(3).set_dim({b, hk, s, d}).set_stride({s * hk * d, d, hk * d, 1}));
  auto opt = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(true).set_attn_scale(1.0f / std::sqrt(float(D)));
  if (causal) opt.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT).set_diagonal_band_right_bound(0);
  auto [o, stats] = g->sdpa(q, k, v, opt);
  o->set_output(true).set_dim({b, hq, s, d}).set_stride({s * hq * d, d, hq * d, 1}).set_uid(4);
  stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(5);
  auto st = g->build(p->handle, {fe::HeurMode_t::A});
  if (!st.is_good()) die("graph build", st.get_message());
  int64_t ws = 0;
  st = g->get_workspace_size(ws);
  if (!st.is_good()) die("workspace size", st.get_message());
  if (ws > 0 && cudaMalloc(&p->workspace, size_t(ws)) != cudaSuccess) die("workspace alloc", "");
  p->graph = g;
  p->pack = {{1, const_cast<void *>(Q)}, {2, const_cast<void *>(K)}, {3, const_cast<void *>(V)}, {4, O}, {5, LSE}};
  return p;
}

void khcudnn_run(KhCudnnSdpa *p) {
  auto st = p->graph->execute(p->handle, p->pack, p->workspace);
  if (!st.is_good()) die("graph execute", st.get_message());
}

void khcudnn_destroy(KhCudnnSdpa *p) {
  if (!p) return;
  cudaFree(p->workspace);
  cudnnDestroy(p->handle);
  delete p;
}
