# 五模型原生VCD主比较登记与结果

本次主比较采用完整的无引导原生VCD作为baseline：`guided=false`、`reference_guided=false`、两个marker均为`NONE`、replicate为0。五模型各使用Food-101 eval全部2424张图像，每类24张、共101类。原生VCD的12120条主正确性、弃权和完整名称评分均已判定。新baseline登记在`outputs/paper_core_20260930/run_20261001_main_native_baseline_v3_named/selected_main_baseline.json`，每模型保存独立condition key、原始生成身份、配置SHA及评分来源。

IP-VCD四项指标分别选取四个注册同措辞条件的eval最佳观测值。每项保存实际marker、冻结condition ID、prompt/config ID、分子与分母；这些最佳值来自独立选择的条件。完全相同的比率按`UNKNOWN`、`UNCLEAR`、`UNSURE`、`I cannot identify it`顺序选择，全部同值候选保存在CSV的`tie_candidates`。旧352条件、旧原始评分及论文正文保持原样。

下表每格为“原生VCD → IP-VCD最佳观测”，数值单位为百分比。`U`表示`UNKNOWN`，`C`表示`UNCLEAR`，`I`表示完整措辞`I cannot identify it`；方括号列出IP-VCD的真实冻结condition ID。

| 注册checkpoint | 准确率 | 合理弃权保留率 | 弃权精确率 | 弃权召回率 |
|---|---|---|---|---|
| Qwen/Qwen2.5-VL-7B-Instruct | 28.3416 → 37.2112 [C,230] | 0/256 → 216/256=84.3750 [I,229] | NA(0/0) → 143/454=31.4978 [C,230] | 0/383 → 251/383=65.5352 [U,231] |
| Qwen/Qwen3.5-4B | 51.8977 → 51.8977 [I,309] | 0/42 → 18/42=42.8571 [I,309] | NA(0/0) → 61/121=50.4132 [U,311] | 0/486 → 122/486=25.1029 [C,310] |
| llava-hf/llava-v1.6-mistral-7b-hf | 38.2426 → 35.1485 [U,87] | 0/899 → 582/899=64.7386 [I,85] | NA(0/0) → 66/95=69.4737 [U,87] | 0/949 → 584/949=61.5385 [I,85] |
| openbmb/MiniCPM-V-2_6 | 43.6056 → 41.4604 [I,165] | 1/374=0.2674 → 249/374=66.5775 [C,166] | 1/1=100 → 24/34=70.5882 [U,167] | 1/597=0.1675 → 251/597=42.0436 [C,166] |
| google/gemma-3-4b-it | 53.7129 → 53.0528 [C,22] | 1/1=100 → 1/1=100 [C,22] | 1/1=100 → 1/1=100 [U,23] | 1/422=0.2370 → 1/422=0.2370 [U,23] |

IP-VCD最佳观测准确率在Qwen2.5-VL上高于原生VCD，Qwen3.5-4B持平，其余三模型原生VCD更高。IP-VCD最佳观测保留率和召回率在前四模型上更高；Gemma保持相同的1次合理弃权事件。MiniCPM与Gemma原生VCD的精确率100%均仅来自1次弃权。

合理弃权保留率的固定集合继续使用每个措辞对应的**冻结guided Direct弃权 ∩ 冻结uniformR**。选定IP条件与原生VCD在完全相同的集合上计数：

| 模型 | 保留率marker | 原guided Direct CID | IP-VCD CID | 固定集合大小 | 原生VCD保留 | IP-VCD保留 |
|---|---|---:|---:|---:|---:|---:|
| qwen25vl | I cannot identify it | 217 | 229 | 256 | 0 | 216 |
| qwen35_4b | I cannot identify it | 297 | 309 | 42 | 0 | 18 |
| llava16_mistral | I cannot identify it | 73 | 85 | 899 | 0 | 582 |
| minicpm26 | UNCLEAR | 154 | 166 | 374 | 1 | 249 |
| gemma3_4b | UNCLEAR | 10 | 22 | 1 | 1 | 1 |

全部20个逐措辞固定集合另存`retention_all_four_markers.csv`，5457个带model/marker的实际集合成员另存`retention_fixed_set_memberships.csv`。Gemma的`UNKNOWN`固定集合大小为0，其保留率保存为空值，原因是`no_frozen_Direct_reasonable_abstentions`。零精确率分母的原生条件保存为空值，原因是没有发生原生弃权。

准确率配对沿用101类别cluster bootstrap、2000次、seed20260929。每个模型比较选定准确率IP条件与同一2424图像、相同种子、类别与uniformR绑定的原生VCD，差值为IP减原生，单位为百分点：

| 模型 | 实际IP CID | 差值 | 配对95%区间 |
|---|---:|---:|---|
| qwen25vl | 230 | +8.8696 | [6.0644,11.9637] |
| qwen35_4b | 309 | 0.0000 | [-2.1462,1.9389] |
| llava16_mistral | 87 | -3.0941 | [-6.1469,-0.3703] |
| minicpm26 | 165 | -2.1452 | [-3.8779,-0.5363] |
| gemma3_4b | 22 | -0.6601 | [-2.0627,0.6188] |

这些区间描述实际选定配置的配对回答差异，未校正eval最大值选择；配置选择属于eval观测。五份配对统计从相同score SHA与相同bootstrap配置的已验收结果精确复用，来源记录在验收文件。逐措辞联合指标`J=(正确数+合理弃权数)/2424`另存`joint_correct_or_reasonable_abstention_all_markers.csv`。

实际入口为`workflows/paper_core/select_native_baseline.py`。输入使用`run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed`及冻结`outputs/paper_20260929/{conditions.csv,scores.parquet,references.parquet}`；只读取40个必要冻结条件的96960评分行，未读取旧raw。主结果目录内的`verification.json`确认20项endpoint、20项逐措辞保留率、五份配对统计、每条件101×24配额、零参考缺失、零主正确性/弃权未决、旧IP计数差异0，并保存每个输入与输出的SHA。

中央实际运行命令：

```bash
TMPDIR=$PWD/.cache/tmp venv/bin/python -m py_compile workflows/paper_core/select_native_baseline.py
TMPDIR=$PWD/.cache/tmp venv/bin/python workflows/paper_core/select_native_baseline.py \
  --reuse-paired-from outputs/paper_core_20260930/run_20261001_main_native_baseline_v2_marker_ties \
  --output outputs/paper_core_20260930/run_20261001_main_native_baseline_v3_named
```
