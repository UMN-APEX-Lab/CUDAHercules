# Hopper benchmark 修复与 H200 验证

后续性能 reference 检查见 [HOPPER_REFERENCE_AUDIT.md](HOPPER_REFERENCE_AUDIT.md)：
替换弱 GEMM baseline、补齐全部 FA3 的生产 release 参数，并重新验证和发布 85 项缓存。
该报告和 `docs/validation/hopper_reference_audit.json` 保留新旧版本来源；历史 KDA 分数没有改写。

本次修复对应 `HOPPER_BLACKWELL_AUDIT.md` 中的 Hopper 部分：21 个 Class 1、64 个 Class 2，共 85 项。验证机器实际为 **H200 NVL / SM90**，并非审查时假设的 H200 SXM。Blackwell 任务文件未修改，也未在 H200 上验证。

## 判分与计时

- 浮点输出在调用 candidate 前填充 NaN；NaN、正无穷、遗漏输出和未写区域均不能通过。合法 mask 使用的负无穷必须逐位置匹配。
- 输出按名字比较，支持每个输出独立的误差界；FA3 的 LSE 单独收紧，梯度同时限制归一化 RMS，拒绝 `LSE+0.3`、`dV*0.9`。
- 所有 Hopper 配置明确要求 `sm_90a`，补齐精度规则和禁止调用 reference/harness 的规则。静态规则是约束的一部分，不是恶意代码的完整隔离沙箱。
- Make 任务逐个解析 `=== case ===`，支持 Ref/Kernel 两种打印顺序；任何缺少计时的 case 都不得分。各 size 加速比取几何平均。
- Reference 与 candidate 使用相同的冷缓存条件和 **minimum/minimum**：Make 任务清理 128 MiB，FA3 清理 256 MiB L2。CUTLASS 初始化、workspace 分配和 TMA descriptor 构造移出稳态计时；算法必须执行的 workspace 清零、输出归约及 task 63 的前置激活仍计时。
- 新增 `reference_benchmarks/NVIDIA_H200_NVL.json`。缓存需要任务源码指纹匹配，修改 shape、精度、描述、Makefile 或共享校验头会使旧缓存失效。CUDA/Torch 版本、硬件和计时方法记录在缓存 metadata 中；更换依赖版本后应重新生成缓存。

## Class 1

| 任务 | 修复后的关键行为 |
|---|---|
| 48 TF32 | 非周期 FP32 数据，加逐 K 互逆的 `2^-20..2^20` 缩放，覆盖 TF32 指数范围；禁止 BF16/FP16 降精度；独立 FP64 采样 oracle。 |
| 49 FP16 | 随机小数、合理的逐元素与 RMS 误差界；GPU 采样代替大规模朴素全量 reference。 |
| 50 INT8 | 全范围随机 INT8，INT32 输出精确比较；提示签名与代码一致。保留整数 GEMM 的原始任务契约。 |
| 51 GETT | 实际收缩两个 K mode，B 使用 `[d,o,h]` 排列，不能作为普通列主序 GEMM 读取；FP32 输出。 |
| 52 gather/scatter | 增加同一运算的 CUTLASS baseline 和完整计时；通过转置视图表达 MoE token 路由；选中输出必须写，未选中输出保持零。 |
| 54 FP8 | 统一为 E4M3→FP32 GEMM，去除只有 reference 执行的 ReLU/aux/amax；解码后的 cuBLAS pedantic FP32 独立 oracle 取代 CPU 三重循环。 |
| 55 W8A16 | B 改为真正 INT8；明确 `half(scale*B+zero)` 和 MN-major scale/zero 索引；覆盖 decode/prefill。 |
| 56 batched | 128 个专家的矩形 GEMM，batch 间独立随机输入；FP16 不能通过 FP8 降精度。 |
| 57 grouped | 非均匀专家及空组，B 明确 E5M2、D 列主序；使用周期性 FP32 promotion，增加 FP64 采样 oracle。 |
| 61 router | N=128、top-8；先用 FP32 logits 选择并归一化，再舍入到 FP16；随机小数避免整数退化。 |
| 62 sparse | 从真实 2:4 剪枝矩阵转换 canonical compressed/metadata 接口；校正转置和 FP16 输出舍入。 |
| 63 prefetch | BF16 输出避免 FP8 饱和；reference/candidate 均执行 SiLU→E4M3 前置 kernel，实际形成 PDL 流水线。 |
| 67 blockwise | 128×128 正数非恒定 scales，移除额外 epilogue，明确 scale 布局；输出填毒并收紧误差。 |
| 68 grouped blockwise | 非均匀及空组，A 使用 1×128 scales、B 使用 128×128 scales；BF16 输出，修复两个 starter 路径的 scale 索引。 |
| 69 mixed grouped | 真正 FP16×INT8 ConvertOnly；校验与计时使用同一输入和运算，错误退出码传递，不再无条件 PASS。 |
| 88 FMHA | D=128、8K/16K 非 causal MHA，NaN 安全校验及更紧误差。此 ABI 仍是 MHA；GQA/causal 由 FA3 覆盖。 |
| DeepGEMM logits | 4K/32K、8K/32K、8K/64K；全部 size 计分；修正末尾 KV tile 的精确 mask，增加独立采样 oracle，缓存 TMA。 |
| TK BF16 GEMM | 矩形 decode/prefill、独立 cuBLAS oracle、考虑 BF16 舍入的 max/RMS 校验。 |
| TK FP8 GEMM | BF16 输出；逐 K tile promotion，避免低精度累加漂移；独立 oracle。 |
| TK scaled FP8 GEMM | oracle 解码实际量化输入并应用 token/channel scales，校验与被测运算一致。 |
| TK rotary | B1/H72/N16384/D128，各 head 输入独立，position 从 8192 开始；修复第 16 行之后错误的 sin/cos 行索引，加入独立 oracle。 |

GEMM 默认矩形尺寸包含 decode 和 prefill；具体尺寸与 API 由 `scripts/generate_hopper_descriptions.py` 从实现提取，生成全部 25 个 Make 任务的说明。任务中的简单 starter 用来验证可实现性，不承诺达到生产性能。

## Class 2

60 个 FA3 任务由 `scripts/generate_fa3_tasks.py` 和 `scripts/templates/fa3/` 生成，`--check` 检查产物一致性。

- Forward：8K prefill，D128 使用 64 query heads / 8 KV heads，其余 head dimension 使用 32/8。D64/96/192 是通用算法覆盖，不声称对应某个完整模型；D192 的 V 维度仍为 192，不等同于 MLA 的异维 V。
- Backward：B1/H16/S4096 的 MHA 训练片段，独立分块 FP32 analytic oracle，避免完整 attention score 的显存开销。
- Split decode：Sq=4，KV 长度约 16K–128K，均包含非整齐尾块；B 与 head 数按长度调整。20 项分别在 H200 上测试 splits=1/2/4/8/16/32，冻结最佳配置为 D64→4、D96→8、D128→4、D192→16、D256→32。校准表为 `scripts/templates/fa3/h200_splits.json`。
- Causal 使用 **右下对齐** `k <= q + Sk - Sq`。Split causal 输入含有可观测的尾部相关 key，防止长 KV 下忽略少量 forbidden keys 被平均掉。
- 缓存设备属性；修复 D128 non-causal backward 的 **80 行** padding，并同步修复 scratch、LSE 和 semaphore 的尺寸。额外测试 B2/H4/Sq257/Sk385，所有缓冲区附带 canary。
- LSE：`atol=2e-4, rtol=2e-5`，normalized RMS≤2e-5。主要输出同时检查逐元素误差和 normalized RMS；FP16 RMS≤0.002、BF16≤0.01。实际每个输出的完整阈值以生成的 `def.py` 为准。

四个 TK Class 2 任务改为原生随机输入 harness，candidate 在独立 translation unit 中编译，不再能直接调用 reference kernel。默认 Based/Hedgehog 为 B8/H32/N8192，linear attention 为 B4/H32/N16384，Mamba2 为 B8/H80/N8192、**state=128、value=64**。初始化 feature maps、slopes、decays、alpha/beta，独立输入覆盖 batch/head 索引。修复 Based 的完整二阶 Taylor starter、Hedgehog 混合窗口及特征映射、Mamba2 跨 chunk 状态；linear reference 的 store-add 输出清零计入自身计时。

## 验证与复现

验证证据保存在 [`validation/hopper_h200.json`](validation/hopper_h200.json)，包含任务源码指纹、各 shape 基线、FA3 三个随机种子的最差误差、负向控制及 split 调参数据。框架、memcheck 和实际 runner 检查记录在 [`validation/hopper_checks.json`](validation/hopper_checks.json)。原始构建和运行日志位于被 git 忽略的 `results/`。

| 检查 | 结果 / 范围 |
|---|---|
| Make reference | 25/25，62 个默认大尺寸编译、执行、reference 检查及计时成功 |
| Make 正负控制 | 25/25：简单正确实现通过诊断尺寸，真实 no-op candidate 被拒绝；rotary 使用完整默认尺寸 |
| FA3 独立 FP32 oracle | 60/60，seeds=42/43/44；拒绝空/零/NaN 输出、LSE 偏置或 dV 缩小、FP8 operands，causal 项拒绝忽略 mask |
| TK 独立 FP32 oracle | 4/4，B1/H2/N256，多 chunk；normalized RMS 0.21%–0.28%；FP8 operands 的误差为 2.59%–4.04%，均超过 harness 的 1% RMS 界；Based 的纯 cumsum 路径也被拒绝 |
| Backward memcheck | hdim128 non-causal、非整齐多 head 形状、连续五次调用，0 个 memory errors，canary 未损坏 |
| 框架回归 | 14 个 CPU tests；两套生成器 `--check` |
| 真实 runner/eval no-op | FA3 fwd 的实际 CUDA no-op 编译成功，输出填毒后被判 FAIL |
| 真实 CLI | `cuda-hercules run` + rotary starter，完整构建、静态检查、判分、计时 PASS |

环境：CUDA 13.0.88、driver 580.178.04、PyTorch 2.14.1+cu130，Python 3.10。需要 ninja、Python 开发头文件和已准备的 `reference_sources/`。使用项目 Python 环境运行：

```bash
python scripts/generate_fa3_tasks.py --check
python scripts/generate_hopper_descriptions.py --check
python -m pytest -q tests/test_hopper_regressions.py
python scripts/test_hopper_references.py --jobs 4 --positive --resume --output results/hopper_make_verified.json
python scripts/test_fa3_references.py --jobs 4 --seeds 42,43,44 --resume --output results/hopper_fa3_verified.json
# 先在四个 tasks/class2/hopper/tk_* 目录分别 make -B ref
python scripts/test_tk_hopper_oracles.py
python tests/cuda/hopper_eval_poison.py
compute-sanitizer --tool memcheck --error-exitcode 1 python tests/cuda/hopper_bwd_workspace.py
python scripts/record_hopper_baselines.py
```

两个批量验证脚本允许 CPU 编译并行，GPU 正确性与计时共用文件锁串行运行。`--resume` 只复用源码指纹匹配的成功结果。`record_hopper_baselines.py` 拒绝缺失、失败或过时记录。

本次改变了输入规模、精度契约及部分函数签名，包括 FA3 的 Hkv、task 51 的 FP32 输出、55/69 的 INT8 输入、57 的 E5M2 B、63/68/TK FP8 的 BF16 输出、Mamba2 state-128。**旧 solution 和旧分数需要重新生成或适配并重新测量，不能与新基线直接比较。** 当前验证针对 85 个活动任务；未编译的历史变体没有变成新增 benchmark 任务，也未声称通过测试。
