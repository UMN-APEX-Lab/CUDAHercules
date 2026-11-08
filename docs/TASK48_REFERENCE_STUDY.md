# Task 48 TF32 reference 调研与 H200 实测

结论：本次候选中，缓存 descriptor、workspace 和算法配置的 cuBLASLt
`CUBLAS_COMPUTE_32F_FAST_TF32` 最适合作为新的性能 reference。
**现已集成、通过正式 harness 正负控制并刷新 H200 baseline**。KDA agent 保持暂停。
全部 Hopper 任务的后续检查见 [HOPPER_REFERENCE_AUDIT.md](HOPPER_REFERENCE_AUDIT.md)。

当前正式 reference 的 decode/prefill/MLP 最小时间为 **0.0914 / 3.0173 / 11.5825 ms**。
保存的 KDA kernel 经新正式 harness 重测后仍 PASS，几何平均加速比为 **1.010645×**。
原始实验的 2.125234× 记录没有改写；新旧分数分别对应不同的 benchmark 版本。

## 生产实现依据

查阅并固定了四个仓库的源码版本，未将 agent 生成的 kernel 用作 reference：

- DeepGEMM `5081f8513af57e61696de4f4f04b0e707ffb503e` 的
  [smxx_cublaslt.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/impls/smxx_cublaslt.hpp)
  使用 cuBLASLt FAST_TF32 处理普通 GEMM。这比借用其特定输入布局和计算语义的 TF32 HC kernel 更符合任务 48。
- FlashInfer `74a72ead04c030c3970a349d21e43665a3ffe3e2` 的
  [mm_bf16_cublaslt.cuh](https://github.com/flashinfer-ai/flashinfer/blob/74a72ead04c030c3970a349d21e43665a3ffe3e2/include/flashinfer/gemm/mm_bf16_cublaslt.cuh)
  提供算法枚举及可保存的配置，可参考其准备、选择与执行分离的方法；这里没有使用 BF16 代替 TF32。
- 另查阅 vLLM `cbe948fe0c7f3db65b9f5bc9ec50754c706d0cd9` 和
  SGLang `46e296e628fc04c135f60eb35b8e93d9bef05ca0`。这些仓库的大量专用路径面向 BF16/FP8，
  不能直接替代本任务的 FP32 输入、TF32 计算语义。

## 实测

H200 NVL，CUDA 13.0，cuBLAS 13.1.0。保留任务的 A row-major、B/C column-major、
FP32 输入输出、逐 K 互逆指数缩放以及 alpha=1、beta=0。

以下为集成前的探索记录；当时尚未发现整个 Hopper 构建缺少 production release 参数的问题。
正式集成采用 release 构建，并重新运行全部验证。当前调优工具默认启用 release 模式。

第一轮比较了 17 个 cuBLASLt 算法配置、每种 shape 各 14 个 CUTLASS 配置及原 wrapper，
总计 62 条 shape/config 记录，全部通过实验数值检查。CUTLASS 覆盖不同 M/N/K tile 和
1×1、1×2、2×1、4×2 cluster。cuBLASLt 在三个默认 shape 上均胜出；这不是对所有可能配置的穷举。

随后使用 seed 0、42、314 对 cuBLASLt 重复验证，并以 seed 42 检查 CUDA Graph，
20/20 条记录通过。另把保存的 KDA kernel 与 cuBLASLt 放进同一程序，三个 seed 共
60/60 条记录通过。每条计时为 3 次 warmup、20 次试验，每次清理 128 MiB L2，保存 min/median。
算法选择、descriptor 创建、workspace 分配均在计时外。

下表为独立同机对照 seed 0 的最小 CUDA event 时间，单位 ms。
“旧 reference”仅用于展示原正式缓存的历史值，并非本轮同步重测值。

| M,N,K | 旧 reference（历史缓存） | cuBLASLt | 保存的 KDA kernel（本轮重测） | KDA 加速比 |
|---|---:|---:|---:|---:|
| 128,8192,8192 | 0.3907 | 0.091296 | 0.092032 | 0.992× |
| 8192,8192,8192 | 4.2220 | 3.018910 | 3.352060 | 0.901× |
| 8192,28672,8192 | 16.0631 | 11.569300 | 11.489100 | 1.007× |

三个 seed 的 min-time 几何平均加速比分别为 0.965×、0.960×、0.965×；
median-time 则为 0.972×、0.991×、0.988×。因此该 reference 下，当前 KDA kernel 大致与
库实现持平，原来的 2.125× 主要反映旧 reference 的性能空间。
GPU 时钟没有锁定，执行顺序固定，不能把几个百分点差异解释为稳定的优劣结论。
本轮重测也不能与历史 KDA 时间直接相除得出正式新分数。

decode/prefill 采用 heuristic index 0，MLP 采用 index 1（另有性能相近的 index 6）。
三个 shape 的 algorithm ID 都是 43，tile ID 分别为 18、23、24。
算法编号和 heuristic 顺序是库版本相关的。集成实现枚举至多 64 个合法配置，
用 scratch D 进行五次 cold-L2 计时后缓存选定算法，不硬编码上述编号。
scratch D 保证准备阶段不改写 beta 所引用的 C。256 MiB workspace、descriptor 和选择均在计时外。

## 数值验证与复现

独立 cuBLAS pedantic FP32 GEMM 生成完整 oracle，再显式执行 TF32 RN/RZ operand rounding
并用 FP32 GEMM 生成两个 rounded oracle。三个 oracle 各做独立 FP64 点积采样检查。
候选完整输出必须有限，且满足某个 oracle 的逐元素累计误差界
`0.003 + 3e-5 * abs(expected)`、NRMS ≤ 1e-4，以及对未量化 FP32 oracle 的 NRMS ≤ 1e-3。
实测对原始 FP32 的 NRMS 约 2.62e-4；对 rounded oracle 的 NRMS 约 1.93e-5。
正式任务现也使用该独立精度判定，避免用一份 TF32 近似结果错误拒绝更准确的 FP32 解。
它保留对真实 FP32 的 NRMS 限制，并分别约束 FP32、TF32 RN、TF32 RZ 的累加误差。
GPU 控制验证三类合法实现通过，零值/NaN/BF16 降精度失败，以及 beta 输入在准备时不被改变。
cuBLASLt numerical flags 额外排除了 FP16/BF16 输入实现，并要求 FP32 累加。

工具为 `scripts/tune_hopper_tf32_reference.cu`。基本 vendor-only 构建：

```sh
mkdir -p .cache/task48_reference
nvcc -O3 -std=c++17 -arch=sm_90a scripts/tune_hopper_tf32_reference.cu -lcublasLt -lcublas -o .cache/task48_reference/tune_lt
flock .cache/hopper_gpu.lock .cache/task48_reference/tune_lt
```

增加 `-DKH_TUNE_CUTLASS --expt-relaxed-constexpr` 和项目 CUTLASS include 路径可比较 CUTLASS。
其中 `current_reference` 是当前集成实现，早期原 wrapper 的历史数据仍在原始结果文件中。
`-DKH_TUNE_SOLUTION='"/absolute/path/to/archived/best_solution.h"'` 可重测保存的 candidate；
此时同样需要 CUTLASS include 路径。归档 candidate 和当前默认流 wrapper 保持普通 launch 计时，
Graph 模式只测支持显式 stream 的 vendor/原生 CUTLASS 配置。
环境变量 `KH_TUNE_SEED`、`KH_TUNE_TRIALS`、`KH_TUNE_CANDIDATE`、`KH_TUNE_GRAPH`
分别控制 seed、重复次数、单个配置过滤、Graph 计时。

原始结果位于工作区的 `results/task48_reference_candidates.jsonl`、
`results/task48_reference_validation.jsonl`、`results/task48_kda_fresh_comparison.jsonl`。
`results/task48_reference_study_summary.json` 保存逐 seed 加速比、依赖版本、源码版本及文件 SHA256。

正式验证及新旧版本证据已发布到 `docs/validation/hopper_reference_audit.json`，
正式缓存为 `reference_benchmarks/NVIDIA_H200_NVL.json`。
