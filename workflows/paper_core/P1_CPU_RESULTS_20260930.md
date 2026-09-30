# 核心五模型原生 VCD 补充结果（2026-09-30）

Food-101 eval 的五个原生 VCD 条件均完成 2,424 条，主正确性、完整名称评分、弃权和参考连接全部已决。每个条件覆盖 101 类，每类 24 张；原生 VCD 与无引导 Direct 按模型、样本、类别、稳定 seed 和冻结参考逐项配对。原生条件为 `guided=false`、`reference_guided=false`、`marker=NONE`、`reference_marker=NONE`、`replicate=0`，两个分支均采用原登记的无引导问题。

中央产物位于 `outputs/paper_core_20260930/run_20260930_core_p1_cpu/`，当前评分为 `scores/v6_all5_literal_root_closed`，验收与可复用分析为 `analysis/v6_literal_complete_reuse`。主统计和图表源为完整的 `analysis/v5_all5_root_closed`、`figures/v5_all5_root_closed`，v6 保存对应字节副本及来源回执。本地小型表格、统计和图表位于 `E:/OneDrive/文档/Playground/kdm_remaining11_20260930_140337/core/p1/`。

无引导 Direct 精确复用 P0 的 `native_direct_root6_20260930`，共 12,120 条。P1 合并表包含冻结的 352 条件、五个无引导 Direct 条件和五个原生 VCD 条件，共 362 条件、877,488 条评分。冻结部分的分母、主正确数、完整名称正确数、弃权数、参考正例、TP/FP/FN、Acc/P/R/F1 与独立重算的 P0 表逐项比较，差异均为零。

## 主正确性与配对变化

Acc 为百分数，变化为百分点。95% CI 使用注册的 2,000 次共享类别抽样、seed 20260929；抽样单元为 101 个 Food-101 类别。

| 模型 | Direct 正确数 / Acc | 原生 VCD 正确数 / Acc | Acc 变化 [95% CI] |
|---|---:|---:|---:|
| Qwen2.5-VL | 737 / 30.4043% | 687 / 28.3416% | -2.0627 [-4.1254, -0.0815] |
| Qwen3.5-4B | 1,244 / 51.3201% | 1,258 / 51.8977% | +0.5776 [-1.9802, +3.0538] |
| LLaVA1.6-Mistral | 821 / 33.8696% | 927 / 38.2426% | +4.3729 [+2.1452, +6.7244] |
| MiniCPM2.6 | 1,043 / 43.0281% | 1,057 / 43.6056% | +0.5776 [-2.2690, +3.2591] |
| Gemma3-4B | 1,328 / 54.7855% | 1,302 / 53.7129% | -1.0726 [-2.3102, +0.1650] |

这五组配对结果显示模型差异：LLaVA1.6-Mistral 的准确率变化区间全为正值，Qwen2.5-VL 的区间全为负值，另外三组区间包含零。该结果对应本次五个已登记的原生 VCD 条件。

## 弃权与固定集合保留

冻结统一参考沿用 `outputs/paper_20260929/references.parquet`，eval 每模型 2,424 条，所有连接完整。P/R/F1 均采用其参考正例集合；以下为百分数。没有预测弃权时 precision 保留空值。

| 模型 | 参考正例 | 原生 VCD 弃权 / TP / FP | Precision | Recall | F1 | Direct 合理弃权保留数 / 固定集合大小 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL | 383 | 0 / 0 / 0 | 空值 | 0% | 0% | 0 / 0 |
| Qwen3.5-4B | 486 | 0 / 0 / 0 | 空值 | 0% | 0% | 0 / 0 |
| LLaVA1.6-Mistral | 949 | 0 / 0 / 0 | 空值 | 0% | 0% | 0 / 7 |
| MiniCPM2.6 | 597 | 1 / 1 / 0 | 100% | 0.1675% | 0.3344% | 1 / 3 |
| Gemma3-4B | 422 | 1 / 1 / 0 | 100% | 0.2370% | 0.4728% | 0 / 0 |

固定集合由同一模型的 Direct 实际弃权与冻结参考正例交集构成。零分母的保留率和 CI 保留空值。LLaVA1.6-Mistral 保留率为 0/7；MiniCPM2.6 为 1/3，其类别 bootstrap 95% CI 为 [0%, 100%]。原生 VCD 的弃权非常稀少，MiniCPM2.6 和 Gemma3-4B 的 precision 均仅来自一个预测弃权。

## 纠正与转换

| 模型 | 两者均正确 | Direct 正确转为错误 | Direct 错误转为正确 |
|---|---:|---:|---:|
| Qwen2.5-VL | 561 | 176 | 126 |
| Qwen3.5-4B | 1,085 | 159 | 173 |
| LLaVA1.6-Mistral | 716 | 105 | 211 |
| MiniCPM2.6 | 873 | 170 | 184 |
| Gemma3-4B | 1,216 | 112 | 86 |

`analysis/v5_all5_root_closed/native_transitions_reference_stratified.csv` 保存五模型、两个冻结参考分层、九种 C/E/A 转换，共 90 行；每模型转换合计 2,424。`native_paired_statistics.json` 保存五组准确率变化、参考召回变化、固定合理弃权保留率及其 bootstrap CI。

`comparison_identity_map.csv` 包含 417 项比较身份，其中 332 项为冻结条件与同主问题 Direct 的对应关系，80 项为 instruction-preserving 条件与其已登记 guided/reference-off 基线的对应关系，五项为本次原生 VCD 与无引导 Direct 的配对。新计算的原生方法效果为这五组配对；冻结同措辞与交叉措辞条件在独立 CSV 中保持原条件主键。

## 来源与判定

五个原生 raw 均在对应完成回执出现后接收。接收记录保存原主机、原相对和绝对路径、source identity、generation identity、SHA256、配置、seed、完整回答及行号。CPU 核验覆盖 2,424 个唯一键、101×24 配额、两个无引导 prompt、登记参数、token 预算、有限概率与 gate/admission。Gemma3-4B 另保留完成时的 replay audit。

| 模型 | 原生成位置 | 原 raw SHA256 |
|---|---|---|
| Qwen2.5-VL | 6403 / `run_20260930_core_p1_a100` | `cfcb48e81d3fcc23f89eab0785445ce2d7cbdbff813041133c03d62fc2deaa81` |
| Qwen3.5-4B | K100 独占 runtime / `run_20260930_core_p1` | `13fd6ed861af9d8c799ff43cd3a541865d6ef9d43c72b0fa13a0391c249286ca` |
| LLaVA1.6-Mistral | 6403 / `run_20260930_core_p1` | `c542fb024e58e1e2c10f3ebd788e4124d82a93edbfd8dbcc41d10db18c36c923` |
| MiniCPM2.6 | 6403 / `run_20260930_core_p1_a100` | `fd497d8e3be37b8dd72a88913780cda370a4076c12faf716e6ea3e5b35edda22` |
| Gemma3-4B | 6403 / `run_20260930_core_p1` | `6ddf59a13aecce4b2a7b455d9d5cd13fecd1419df27512af7b793100497cdaa0` |

行为先精确复用完整问题与回答相同的最终普查标签，再使用冻结规则和实际标注裁定。新增边界共 175 个独占 QA：四个先完成模型 100 项，对应 104 条源回答；Gemma3-4B 75 项，对应 75 条源回答。实际 GPT-5.6 Luna medium 判定及 root 逐项复核保存于两个 annotation 目录；作者、model/effort、原文 span 与源成员核验保留。v5 从既有评分增量合并，接收 raw 未重复扫描。

完整名称评分使用完整主菜名称，保留份量、品牌、风味及烹饪修饰。五项实际 root 裁定保存于 `annotation/root_literal5_20260930/root_reviewed.jsonl`，作者为 root / gpt-6.1-sol / max，文件准备者为 assets。实际核验六个原 raw 成员的 key、样本、完整 QA、identity、config、seed 和行 SHA 后，仅更新六条完整名称评分；历史名称字段与既有 score 来源保留。主正确性、弃权、参考及所有主统计逐项保持相同。

| 模型 | 完整名称正确数 | 完整名称错误数 | 完整名称 Acc |
|---|---:|---:|---:|
| Qwen2.5-VL | 678 | 1,746 | 27.9703% |
| Qwen3.5-4B | 1,190 | 1,234 | 49.0924% |
| LLaVA1.6-Mistral | 828 | 1,596 | 34.1584% |
| MiniCPM2.6 | 883 | 1,541 | 36.4274% |
| Gemma3-4B | 769 | 1,655 | 31.7244% |

每个完整名称条件分母均为 2,424，未决均为零。362 条件合并表同时保留 `literal_correct`、`literal_incorrect`、`literal_pending`、`literal_complete` 和 `literal_accuracy`。

## 实际验证与复用入口

接收入口为 `workflows/paper_core/receive_native.py`，评分入口为 `score_native.py`。v5 使用 v4 四模型评分与 v3 Gemma 单模型评分，并依次读取各批 Luna 与 root 文件；v6 使用 v5 评分和五项实际 root 完整名称裁定，仅更新六个未决成员。最终 `score_rows.jsonl.gz` 共 12,120 行、5,355 个独占 QA。主正确性未决、完整名称未决、弃权未决、boundary queue、参考连接缺项均为零。

CPU 解释器为中央 `/home/g203-4028/projects/knowledge-deficit-mitigation/venv/bin/python`。实际运行入口与验收文件：

```bash
export TMPDIR="$PWD/.cache/tmp"
venv/bin/python workflows/paper_core/score_native.py \
  --reuse-score-rows outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v4_core100_root_closed/score_rows.jsonl.gz \
  --reuse-score-rows outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v3_rules_gemma_only/score_rows.jsonl.gz \
  --decision-file outputs/paper_core_20260930/run_20260930_core_p1_cpu/annotation/core_p1_native_luna_medium_20260930/decisions.jsonl \
  --decision-file outputs/paper_core_20260930/run_20260930_core_p1_cpu/annotation/core_p1_native_luna_medium_20260930/root_reviewed.jsonl \
  --decision-file outputs/paper_core_20260930/run_20260930_core_p1_cpu/annotation/gemma_native_luna_medium_20260930/decisions.jsonl \
  --decision-file outputs/paper_core_20260930/run_20260930_core_p1_cpu/annotation/gemma_native_luna_medium_20260930/root_reviewed.jsonl \
  --output outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v5_all5_root_closed
venv/bin/python workflows/paper_core/analyze_native.py --score-dir outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v5_all5_root_closed --output outputs/paper_core_20260930/run_20260930_core_p1_cpu/analysis/v5_all5_root_closed
venv/bin/python workflows/paper_core/render_native.py --score-dir outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v5_all5_root_closed --analysis-dir outputs/paper_core_20260930/run_20260930_core_p1_cpu/analysis/v5_all5_root_closed --output outputs/paper_core_20260930/run_20260930_core_p1_cpu/figures/v5_all5_root_closed
venv/bin/python workflows/paper_core/close_native_literal.py prepare --pending outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v5_all5_root_closed/literal_sensitivity_pending6.jsonl --output outputs/paper_core_20260930/run_20260930_core_p1_cpu/annotation/root_literal5_20260930
venv/bin/python workflows/paper_core/score_native.py --reuse-score-rows outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v5_all5_root_closed/score_rows.jsonl.gz --decision-file outputs/paper_core_20260930/run_20260930_core_p1_cpu/annotation/root_literal5_20260930/root_reviewed.jsonl --literal-only --output outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed
venv/bin/python workflows/paper_core/verify_native_score.py --score-dir outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed --output outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed/verification.json
venv/bin/python workflows/paper_core/close_native_literal.py verify-reuse --previous-score outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v5_all5_root_closed --current-score outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed --analysis outputs/paper_core_20260930/run_20260930_core_p1_cpu/analysis/v5_all5_root_closed --figures outputs/paper_core_20260930/run_20260930_core_p1_cpu/figures/v5_all5_root_closed --output outputs/paper_core_20260930/run_20260930_core_p1_cpu/analysis/v6_literal_complete_reuse
```

上述评分、分析和图表目录均为独占新版本。v6 `verification.json` 实际通过，冻结完整名称正确数也为零差异。`analysis/v6_literal_complete_reuse/receipt.json` 验证 12,120 条主值和源身份、362 条件主指标均与 v5 相同，再复制已完成五组配对、90 条转换、417 条比较身份和两张 PNG/PDF；来源与副本 SHA 全部一致，新增 bootstrap 计算为零。PNG 实际视觉核验通过，独立回执为 `figures/v5_all5_root_closed/visual_acceptance.json`。CPU 处理期间初始化 GPU 为零、新增生成与 API 调用均为零。

六个新增 Python 入口的 `venv/bin/python -m py_compile` 实际通过，七个源码和文档路径的 `git diff --check` 实际通过。

当前小型提交文件为六个独立入口 `receive_native.py`、`score_native.py`、`analyze_native.py`、`verify_native_score.py`、`render_native.py`、`close_native_literal.py` 及本文件。数据、raw、评分、标注和图表保留上述数据路径。
