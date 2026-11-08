# Hopper 性能 reference 检查与替换

覆盖全部 **85 个 Hopper 任务**：Class 1 的 21 个、Class 2 的 60 个 FA3 和 4 个 TK。
本轮替换弱 reference、补齐生产构建参数并刷新 H200 缓存。KDA agent 保持暂停；
只重测已保存的 task 48 kernel，没有模型调用，也没有改写历史分数。

## 共性问题和结果

FA3 缺少生产构建参数。上游
[hopper/setup.py](https://github.com/Dao-AILab/flash-attention/blob/b2edfab9f8fa81205884cdc9c0b8859cd32934d3/hopper/setup.py)
明确使用 `-DNDEBUG`，注明缺少它会严重影响性能，并启用
`CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED`。原 reference 两者均缺失。
现在所有 Hopper Make 构建启用 release 模式，60 个 FA3 同时补齐这两个参数；
class2 提交也获得对应参数。独立数学 oracle、有限性、RMS、哨兵和正负控制保留，
这些检查不依赖 device assert。

48 的固定 tile/cluster 和计时内硬件查询导致 decode 尤其弱，新 reference 使用缓存的
cuBLASLt TF32。同样的方法还找到了 FP16、INT8、FP8、pointer-batched、gather/scatter、
router 和 grouped W8A16 中的性能空间。

下表单位 ms。“之前”为 `133ae15` 的历史缓存，不是同步重测；小幅变化不能解释为稳定收益。

| 任务 / case | 之前 | 现在 | 最终 reference |
|---|---:|---:|---|
| 48 TF32 decode | 0.3907 | 0.0914 | 缓存 cuBLASLt TF32 |
| 49 FP16 MLP | 8.6449 | 6.2252 | 缓存 cuBLASLt |
| 50 INT8 MLP | 9.8811 | 3.0579 | cuBLASLt，INT32 精确累加/输出 |
| 54 FP8 MLP | 4.2270 | 3.0082 | cuBLASLt，FP32 输出 |
| 55 W8A16 prefill | 2.9058 | 1.8639 | 大 M 宽 tile CUTLASS |
| 56 pointer-batched decode | 0.3877 | 0.2537 | cuBLAS batched / CUTLASS 选择 |
| 63 prefetch 大 decode | 0.1752 | 0.0937 | N tile=128，完整 SiLU→GEMM |
| 69 grouped W8A16 大 case | 2.4343 | 0.3837 | N tile 从 16 改为 128 |
| TK FP8 decode | 0.1702 | 0.0312 | cuBLASLt，BF16 输出 |
| TK BF16 decode | 0.0608 | 0.0295 | cuBLASLt；大矩阵保留 TK |
| FA3 fwd 128 FP16 非 causal | 12.8656 | 3.2304 | 上游 production release 参数 |
| FA3 bwd 128 FP16 非 causal | 1.3943 | 0.6414 | 同上 |
| FA3 split 128 FP16 非 causal | 0.4368 | 0.1472 | release 参数，重新校准 splits |

## 全任务处理结果

| 范围 | 检查及处理 |
|---|---|
| 48 | 枚举 vendor/CUTLASS，采用缓存 cuBLASLt FAST_TF32；用独立 FP32/RN/RZ oracle 判断精度。 |
| 49 / 50 / 54 | 采用 cuBLASLt FP16 / INT8 / FP8，保持各任务的布局、类型、累加精度和输出。 |
| 51 GETT | 比较 cluster 和宽 tile；release 复测没有稳定大收益，保留原配置和真实 B 排列。 |
| 52 gather/scatter | 比较原融合与 gather→cuBLASLt→scatter；完整三阶段计时，完整输出交叉检查。 |
| 55 mixed W8A16 | 小 M 原 tile，大 M 宽 tile；另比较每次反量化→cuBLASLt，反量化计入每次调用。 |
| 56 pointer-batched | 比较 cuBLAS batched 和 CUTLASS，保留 device pointer-array 接口，完整输出交叉检查。 |
| 57 grouped FP8 | cooperative/pingpong 对比，后者更慢，保留原 cooperative release kernel。 |
| 61 router | 比较 CUTLASS 融合与 cuBLASLt FP32 logits→vLLM top-k→scatter；修复离散数据、并列值及默认 alpha。 |
| 62 sparse | 对比 cluster/宽 tile；宽 tile 导致 decode 明显退化，保留原稀疏配置。 |
| 63 prefetch | 调整 N tile；SiLU predecessor 仍在两边每次完整调用中计时。 |
| 67 block-scaled FP8 | 保留 128×128 block scales、FP32 输出；release 比较不同 clusters，没有稳定收益。 |
| 68 grouped block-scaled FP8 | 比较 clusters/宽 tile，宽 tile 更慢；保留原融合 scales 和不均匀 groups。 |
| 69 grouped W8A16 | 修复 N tile=16 的明显限制，改为 128；INT8 字节解释、FP16 输出及不均匀 groups 不变。 |
| 88 attention | 比较 production release CUTLASS/FA3，按 shape 缓存选择，完整 O 交叉检查；保持 MHA、非 causal、FP16。 |
| DeepGEMM MQA logits | 与当前上游核对配置；KV=128/math=256 更慢、5 KV stages 收益不足，保留 3/3 stages、KV=256、512 math threads。 |
| TK BF16 | decode 用 cuBLASLt，大矩阵保留原 TK 编译单元，小矩阵 backend 单独链接。原版 release 大矩阵为 3.3277 ms，新版 3.3336 ms。 |
| TK FP8 / scaled FP8 | 普通 FP8 改 cuBLASLt；scaled 比较完整 vendor GEMM+scale epilogue 与 TK 融合路径。 |
| TK rotary | 显存带宽表现良好，保留 TK，移出重复的 kernel attribute 设置。 |
| FA3 20 fwd / 20 bwd / 20 split | 全部补齐 release 参数；20 个 split 重测 1/2/4/8/16/32 中正确的候选，最终原分割数仍合适。 |
| Class2 TK Based / Hedgehog / Linear / Mamba2 | 保留已验证的生产 kernel；它们原已使用 release 构建，独立数学 oracle 和源码指纹仍匹配。 |

55、63、69 的配置也在恢复的 `133ae15` 源码上以 release 模式比较过。
GETT 宽 tile 的早期收益在 release 复测中不再稳定，已撤回；未采用早期 debug 构建的探索结论。

## 选择与数值规则

- cuBLASLt descriptor、256 MiB workspace、算法枚举/选择在准备阶段完成。选择用 scratch D，
  不污染 beta 输入 C，正式调用执行缓存配置。
- 原生与组合 vendor 实现采用五次交替 cold-L2 试验，中位数优势超过 5% 才切换。
- gather、scatter、反量化、token/channel scaling、router top-k/清零均计入完整操作；
  保留 block-scaled、稀疏、GETT、prefetch 的特殊语义。
- 48 接受准确 FP32 或标准 TF32 RN/RZ；独立 oracle 约束累加误差及对真实 FP32 的 NRMS，
  避免用一份 TF32 近似结果错误拒绝更准确的 FP32 解。
- 61 使用完整随机小数、默认 `alpha=1/sqrt(K)`，按原始 FP32 logits 选恰好八个 expert，
  并列按较小索引排序，再归一化、舍入。其新旧时间不是严格的同输入算法比较。
- router 来自
  [vLLM cbe948fe 的 topk_softmax_kernels.cu](https://github.com/vllm-project/vllm/blob/cbe948fe0c7f3db65b9f5bc9ec50754c706d0cd9/csrc/moe/topk_softmax_kernels.cu)，
  保留 Apache-2.0 标识及版权；去掉 Torch/ROCm dispatch，并按原始 logits 排序，
  避免 softmax 舍入制造并列。DeepGEMM/FlashInfer 的 vendor backend 与配置缓存来源见 task 48 报告。

结论限于已测候选，不是全部库/配置的 SOTA 认证。GPU 时钟没有锁定，完整逐 case 记录、
库版本和依赖源码指纹随缓存发布。

## 验证及版本

- 25/25 Make：62 个默认 case 的 reference 通过；每任务诊断尺寸正控制与默认首 case 的空实现负控制通过。
- 60/60 FA3：每任务 seed 42/43/44，独立 FP32 oracle 和全部负控制通过。
- 20/20 split 校准通过，冻结配置与新构建匹配。
- 15 项 CPU 回归通过；GPU 验证 FP32/RN/RZ 均接受，零值/NaN/BF16 降精度均拒绝，
  beta 输入不受准备阶段改变。
- 新正式 harness 下，原 KDA 48 kernel 仍 PASS，decode/prefill/MLP 为
  **0.0925 / 2.9457 / 11.3564 ms**，几何平均分数 **1.010645×**。
  原 **2.125234×** 属于旧 reference 版本，保留原记录；KDA agent 没有恢复。

证据：`docs/validation/hopper_h200.json`、`docs/validation/hopper_reference_audit.json`；
正式缓存：`reference_benchmarks/NVIDIA_H200_NVL.json`。
复现工具为 `scripts/tune_hopper_native_references.py`、`scripts/audit_hopper_dense_references.cu`、
`scripts/tune_hopper_tf32_reference.cu`，以及已有的三个 Hopper 验证脚本。

修改 reference、构建参数、输入契约或依赖库后，应重新验证并执行
`scripts/record_hopper_baselines.py`。KDA 恢复或重新评测时应明确 benchmark 版本，
保留暂停任务已经使用的时间/evaluation 预算。
