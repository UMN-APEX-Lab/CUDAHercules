/**
 * CUDA-Hercules Class 3: Icicle ZK Benchmark
 *
 * CPU and reference CUDA runs use Icicle's public APIs.
 * Custom solution runs load a fixed-ABI shared library directly via dlopen().
 */
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "icicle/backend/ntt_config.h"
#include "icicle/curves/params/bn254.h"
#include "icicle/msm.h"
#include "icicle/ntt.h"
#include "icicle/runtime.h"

#include "../custom_backend_api.h"

using namespace bn254;
using Clock = std::chrono::high_resolution_clock;
using FpMs = std::chrono::duration<float, std::chrono::milliseconds::period>;

static const std::vector<int> NTT_LOG_SIZES = {16, 18, 20, 22, 24};
static const std::vector<int> MSM_LOG_SIZES = {14, 16, 18, 20, 22};

struct OpResult {
  std::string name;
  int log_size;
  float time_ms;
  bool correct;
  std::string error;
};

struct CustomBackendApi {
  void* handle = nullptr;
  int (*init)() = nullptr;
  void (*shutdown)() = nullptr;
  const char* (*last_error)() = nullptr;
  int (*ntt_forward)(const scalar_t*, int, scalar_t*) = nullptr;
  int (*msm)(const scalar_t*, const affine_t*, int, projective_t*) = nullptr;
  std::string load_error;
};

void save_field_elements(const char* path, const scalar_t* data, size_t n)
{
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data), n * sizeof(scalar_t));
}

bool load_and_compare_field(const char* ref_path, const scalar_t* data, size_t n)
{
  auto ref = std::make_unique<scalar_t[]>(n);
  std::ifstream f(ref_path, std::ios::binary);
  if (!f.good()) return true;
  f.read(reinterpret_cast<char*>(ref.get()), n * sizeof(scalar_t));
  for (size_t i = 0; i < n; i++) {
    if (ref[i] != data[i]) return false;
  }
  return true;
}

void save_projective(const char* path, const projective_t* data, size_t n)
{
  auto affine = std::make_unique<affine_t[]>(n);
  for (size_t i = 0; i < n; i++) {
    projective_t tmp = data[i];
    affine[i] = tmp.to_affine();
  }
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(affine.get()), n * sizeof(affine_t));
}

bool load_and_compare_proj(const char* ref_path, const projective_t* data, size_t n)
{
  auto ref_affine = std::make_unique<affine_t[]>(n);
  std::ifstream f(ref_path, std::ios::binary);
  if (!f.good()) return true;
  f.read(reinterpret_cast<char*>(ref_affine.get()), n * sizeof(affine_t));
  for (size_t i = 0; i < n; i++) {
    projective_t tmp = data[i];
    affine_t a = tmp.to_affine();
    if (!(ref_affine[i] == a)) return false;
  }
  return true;
}

std::string custom_last_error(const CustomBackendApi* backend)
{
  if (!backend || !backend->last_error) return "";
  const char* msg = backend->last_error();
  return msg ? std::string(msg) : std::string();
}

template <typename Fn>
bool load_symbol(void* handle, const char* name, Fn& out, std::string& error)
{
  dlerror();
  void* sym = dlsym(handle, name);
  const char* err = dlerror();
  if (err != nullptr || sym == nullptr) {
    error = std::string("missing symbol: ") + name;
    return false;
  }
  out = reinterpret_cast<Fn>(sym);
  return true;
}

bool load_custom_backend(const char* so_path, CustomBackendApi& backend)
{
  backend.handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
  if (!backend.handle) {
    backend.load_error = dlerror();
    return false;
  }

  if (!load_symbol(backend.handle, "kh_custom_backend_init", backend.init, backend.load_error) ||
      !load_symbol(backend.handle, "kh_custom_backend_shutdown", backend.shutdown, backend.load_error) ||
      !load_symbol(
        backend.handle, "kh_custom_backend_last_error", backend.last_error, backend.load_error) ||
      !load_symbol(
        backend.handle, "kh_custom_ntt_forward_bn254", backend.ntt_forward, backend.load_error) ||
      !load_symbol(backend.handle, "kh_custom_msm_bn254", backend.msm, backend.load_error)) {
    dlclose(backend.handle);
    backend.handle = nullptr;
    return false;
  }
  return true;
}

OpResult run_ntt_icicle(int log_n, const char* ref_dir, bool save_ref)
{
  const int n = 1 << log_n;
  auto input = std::make_unique<scalar_t[]>(n);
  auto output = std::make_unique<scalar_t[]>(n);

  srand(42 + log_n);
  for (int i = 0; i < n; i++) {
    input[i] = scalar_t::from(static_cast<uint32_t>(rand()));
  }

  scalar_t root = scalar_t::omega(log_n);
  auto init_cfg = default_ntt_init_domain_config();
  ntt_init_domain(root, init_cfg);

  NTTConfig<scalar_t> config = default_ntt_config<scalar_t>();
  ntt(input.get(), n, NTTDir::kForward, config, output.get());

  auto t0 = Clock::now();
  auto err = ntt(input.get(), n, NTTDir::kForward, config, output.get());
  auto t1 = Clock::now();
  float ms = FpMs(t1 - t0).count();

  bool correct = true;
  char path[512];
  if (save_ref) {
    snprintf(path, sizeof(path), "%s/ntt_%d.bin", ref_dir, log_n);
    save_field_elements(path, output.get(), n);
  } else {
    snprintf(path, sizeof(path), "%s/ntt_%d.bin", ref_dir, log_n);
    correct = load_and_compare_field(path, output.get(), n);
  }

  ntt_release_domain<scalar_t>();

  char name[64];
  snprintf(name, sizeof(name), "NTT 2^%d", log_n);
  return {name, log_n, ms, correct && (err == eIcicleError::SUCCESS), ""};
}

OpResult run_ntt_custom(const CustomBackendApi& backend, int log_n, const char* ref_dir)
{
  const int n = 1 << log_n;
  auto input = std::make_unique<scalar_t[]>(n);
  auto output = std::make_unique<scalar_t[]>(n);

  srand(42 + log_n);
  for (int i = 0; i < n; i++) {
    input[i] = scalar_t::from(static_cast<uint32_t>(rand()));
  }

  int warmup_err = backend.ntt_forward(input.get(), log_n, output.get());
  cudaError_t warmup_cuda_err = cudaDeviceSynchronize();

  auto t0 = Clock::now();
  int err = backend.ntt_forward(input.get(), log_n, output.get());
  cudaError_t cuda_err = cudaDeviceSynchronize();
  auto t1 = Clock::now();
  float ms = FpMs(t1 - t0).count();

  char path[512];
  snprintf(path, sizeof(path), "%s/ntt_%d.bin", ref_dir, log_n);
  bool correct = load_and_compare_field(path, output.get(), n);
  bool ok = (warmup_err == 0) && (err == 0) && (warmup_cuda_err == cudaSuccess) &&
            (cuda_err == cudaSuccess) && correct;

  char name[64];
  snprintf(name, sizeof(name), "NTT 2^%d", log_n);
  return {name, log_n, ms, ok, custom_last_error(&backend)};
}

OpResult run_msm_icicle(int log_n, const char* ref_dir, bool save_ref)
{
  const int n = 1 << log_n;
  auto scalars = std::make_unique<scalar_t[]>(n);
  auto points = std::make_unique<affine_t[]>(n);
  projective_t result;

  char scalars_path[512], points_path[512];
  snprintf(scalars_path, sizeof(scalars_path), "%s/msm_scalars_%d.bin", ref_dir, log_n);
  snprintf(points_path, sizeof(points_path), "%s/msm_points_%d.bin", ref_dir, log_n);

  std::ifstream fs(scalars_path, std::ios::binary);
  if (fs.good()) {
    fs.read(reinterpret_cast<char*>(scalars.get()), n * sizeof(scalar_t));
    fs.close();
    std::ifstream fp(points_path, std::ios::binary);
    fp.read(reinterpret_cast<char*>(points.get()), n * sizeof(affine_t));
  } else {
    scalar_t::rand_host_many(scalars.get(), n);
    projective_t::rand_host_many(points.get(), n);
    {
      std::ofstream f(scalars_path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(scalars.get()), n * sizeof(scalar_t));
    }
    {
      std::ofstream f(points_path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(points.get()), n * sizeof(affine_t));
    }
  }

  auto config = default_msm_config();
  msm(scalars.get(), points.get(), n, config, &result);

  auto t0 = Clock::now();
  auto err = msm(scalars.get(), points.get(), n, config, &result);
  auto t1 = Clock::now();
  float ms = FpMs(t1 - t0).count();

  bool correct = true;
  char path[512];
  if (save_ref) {
    snprintf(path, sizeof(path), "%s/msm_%d.bin", ref_dir, log_n);
    save_projective(path, &result, 1);
  } else {
    snprintf(path, sizeof(path), "%s/msm_%d.bin", ref_dir, log_n);
    correct = load_and_compare_proj(path, &result, 1);
  }

  char name[64];
  snprintf(name, sizeof(name), "MSM 2^%d", log_n);
  return {name, log_n, ms, correct && (err == eIcicleError::SUCCESS), ""};
}

OpResult run_msm_custom(const CustomBackendApi& backend, int log_n, const char* ref_dir)
{
  const int n = 1 << log_n;
  auto scalars = std::make_unique<scalar_t[]>(n);
  auto points = std::make_unique<affine_t[]>(n);
  projective_t result;

  char scalars_path[512], points_path[512];
  snprintf(scalars_path, sizeof(scalars_path), "%s/msm_scalars_%d.bin", ref_dir, log_n);
  snprintf(points_path, sizeof(points_path), "%s/msm_points_%d.bin", ref_dir, log_n);

  std::ifstream fs(scalars_path, std::ios::binary);
  if (fs.good()) {
    fs.read(reinterpret_cast<char*>(scalars.get()), n * sizeof(scalar_t));
    fs.close();
    std::ifstream fp(points_path, std::ios::binary);
    fp.read(reinterpret_cast<char*>(points.get()), n * sizeof(affine_t));
  } else {
    scalar_t::rand_host_many(scalars.get(), n);
    projective_t::rand_host_many(points.get(), n);
    {
      std::ofstream f(scalars_path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(scalars.get()), n * sizeof(scalar_t));
    }
    {
      std::ofstream f(points_path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(points.get()), n * sizeof(affine_t));
    }
  }

  int warmup_err = backend.msm(scalars.get(), points.get(), log_n, &result);
  cudaError_t warmup_cuda_err = cudaDeviceSynchronize();

  auto t0 = Clock::now();
  int err = backend.msm(scalars.get(), points.get(), log_n, &result);
  cudaError_t cuda_err = cudaDeviceSynchronize();
  auto t1 = Clock::now();
  float ms = FpMs(t1 - t0).count();

  char path[512];
  snprintf(path, sizeof(path), "%s/msm_%d.bin", ref_dir, log_n);
  bool correct = load_and_compare_proj(path, &result, 1);
  bool ok = (warmup_err == 0) && (err == 0) && (warmup_cuda_err == cudaSuccess) &&
            (cuda_err == cudaSuccess) && correct;

  char name[64];
  snprintf(name, sizeof(name), "MSM 2^%d", log_n);
  return {name, log_n, ms, ok, custom_last_error(&backend)};
}

int main(int argc, char* argv[])
{
  const char* device = "CPU";
  const char* backend_dir = nullptr;
  const char* custom_so = nullptr;
  const char* ref_dir = "/tmp/icicle_ref";
  bool save_ref = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
    else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend_dir = argv[++i];
    else if (strcmp(argv[i], "--custom-so") == 0 && i + 1 < argc) custom_so = argv[++i];
    else if (strcmp(argv[i], "--ref-dir") == 0 && i + 1 < argc) ref_dir = argv[++i];
    else if (strcmp(argv[i], "--save-ref") == 0)
      save_ref = true;
  }

  if (backend_dir && custom_so) {
    std::cerr << "Use either --backend or --custom-so, not both." << std::endl;
    return 1;
  }

  CustomBackendApi custom_backend;
  if (custom_so) {
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count <= 0) {
      std::cerr << "CUDA device not available for custom backend: "
                << cudaGetErrorString(cuda_err) << std::endl;
      return 1;
    }
    cudaSetDevice(0);
    cudaFree(0);

    if (!load_custom_backend(custom_so, custom_backend)) {
      std::cerr << "Failed to load custom backend: " << custom_backend.load_error << std::endl;
      return 1;
    }
    if (custom_backend.init() != 0) {
      std::cerr << "Custom backend init failed: " << custom_last_error(&custom_backend) << std::endl;
      dlclose(custom_backend.handle);
      return 1;
    }
  } else if (backend_dir) {
    icicle_load_backend(backend_dir, true);
  }

  if (strcmp(device, "CPU") != 0 && custom_so == nullptr) {
    if (icicle_is_device_available(device) != eIcicleError::SUCCESS) {
      std::cerr << "Device " << device << " not available!" << std::endl;
      return 1;
    }
    icicle_set_device(device);
  }

  std::cout << "Device: " << device << std::endl;

  std::vector<OpResult> results;
  float ntt_total = 0, msm_total = 0;

  std::cout << "\n--- NTT ---" << std::endl;
  for (int log_n : NTT_LOG_SIZES) {
    OpResult r =
      custom_so ? run_ntt_custom(custom_backend, log_n, ref_dir) : run_ntt_icicle(log_n, ref_dir, save_ref);
    printf("  %s: %.2f ms [%s]\n", r.name.c_str(), r.time_ms, r.correct ? "PASS" : "FAIL");
    if (!r.correct && !r.error.empty()) {
      std::cout << "    error: " << r.error << std::endl;
    }
    ntt_total += r.time_ms;
    results.push_back(r);
  }

  std::cout << "\n--- MSM ---" << std::endl;
  for (int log_n : MSM_LOG_SIZES) {
    OpResult r =
      custom_so ? run_msm_custom(custom_backend, log_n, ref_dir) : run_msm_icicle(log_n, ref_dir, save_ref);
    printf("  %s: %.2f ms [%s]\n", r.name.c_str(), r.time_ms, r.correct ? "PASS" : "FAIL");
    if (!r.correct && !r.error.empty()) {
      std::cout << "    error: " << r.error << std::endl;
    }
    msm_total += r.time_ms;
    results.push_back(r);
  }

  bool all_passed = true;
  for (const auto& r : results) {
    if (!r.correct) all_passed = false;
  }

  float e2e_total = ntt_total + msm_total;
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << (all_passed ? "Passed" : "FAILED") << std::endl;
  printf("NTT total: %.2f ms\n", ntt_total);
  printf("MSM total: %.2f ms\n", msm_total);
  printf("Kernel time: %.4f ms\n", e2e_total);

  if (custom_backend.handle) {
    custom_backend.shutdown();
    dlclose(custom_backend.handle);
  }

  return all_passed ? 0 : 1;
}
