# 四模型原生基线 CPU 检查点

当前检查点：`outputs/supplemental/remaining4/native_cpu_20261001_0226/checkpoints/v4_19392_all_native_closed`。

全部 19392 条唯一任务已经接收并评分，生成缺口为 0，重复任务为 0。每个模型的 VCD/M3ID 条件均使用同一 Food-101 eval 2424 张图像，101 类各 24 张；`guided=false`、`reference_guided=false`、两个 marker 均为 `NONE`、`replicate=0`。算法、原注册精度、参数、提示词、seed 和身份均由不可变来源的 CPU 验收证明连接。

| 模型 checkpoint | VCD 主正确数 / 2424 | M3ID 主正确数 / 2424 |
| --- | ---: | ---: |
| OpenGVLab/InternVL3_5-8B | 462 / 2424 | 426 / 2424 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | 849 / 2424 | 777 / 2424 |
| microsoft/Phi-3.5-vision-instruct | 916 / 2424 | 862 / 2424 |
| Qwen/Qwen3-VL-8B-Instruct | 1380 / 2424 | 1302 / 2424 |

主正确性、完整抽取名称正确性和弃权状态全部已决，三类未决均为 0。上述数值为实际单条件计数；方法配对效应尚未计算。

## 有限来源与复核

- 旧 3632 条：`native_cpu_20260930_2338/scores/v2_root24_cone1_closed`。
- 追加 10928 条：`native_cpu_20261001_0115/scores/v2_root40_closed`。
- Phi 追加 3960 条：`native_cpu_20261001_0226/scores/v2_root69_closed`。实际 Luna medium 标注 69 个完整问答，root 实际复核并追加 69 个裁定，涉及 74 个来源成员；24 个多主答案、45 个唯一答案。`canonical_override` 全部为空，评分由原 101 类词边界函数计算。其余 3886 个评分对象逐键保持一致。
- Phi 最后 872 条：`native_cpu_20261001_0226/scores/v4_root_tail8_closed`。仅接收最新快照中的两个新增不可变分片，排除此前 18520 个任务键；规则及精确 QA 复用解决 864 条。其余 8 个完整问答经实际 Luna medium 标注和 root 逐项复核，追加于 `annotation/root_tail8_20261001/root_reviewed.jsonl`；3 个多主答案、5 个唯一答案。仅更新 8 个评分对象，其余 864 个对象逐键相同。原规则评分目录 `scores/v3_rules_phi_tail872` 和其 8 问答包保留。

上述路径前缀均为 `outputs/supplemental/remaining4/`。原 69 问答包与所有 Luna 文件保留；root 作者记录为 `/root`、`gpt-6.1-sol`、`max`，脚本写入者为 `/root/assets`，未知 call ID 为空。复核保存完整来源、实际行 SHA、模型身份、seed 和完整问答成员。

## 实际验收

`review_phi_boundary69.py` 已完成 74 个来源成员核对与有限评分更新。`review_phi_tail8.py` 已完成最后 8 个来源成员核对与有限评分更新。`verify_native_score.py` 已验收 3960 条与最后 872 条。`merge_native_scores.py` 已合并 19392 条；`verify_native_checkpoint.py` 对全部来源评分对象逐键比较，确认合并对象与来源相同，并核对八个条件的 101×24 配额和缺口。

当前 checkpoint 的 `verification.json` 为实际验收凭据；19392 个来源对象全部相同，8 个条件全部输入完整、评分已决，缺失任务和重复任务均为 0。最终 `score_rows.jsonl.gz` 的 SHA256 为 `6eba9425c29ccf4f3c36271c88481bb299dbf8ac4b46ce025bc9b4c1fe7adfd2`。

旧 raw 未重新读取，本 CPU 子任务的 GPU 初始化和新增 API 调用均为 0；实际 Luna 调用由单独标注线程完成并保留执行回执。原 v4 检查点只保存完整原生评分；后续参考连接另存独占目录。原五模型资产及正文保持冻结。

## OneVision/Qwen3-VL 已有完整参考连接

`native_cpu_20261001_0226/reference_join/one_qwen3_v1_20261001/receipt.json` 已实际核对原生评分键和 `selected4_v10_root46/reference_G.jsonl` 来源，两个模型各 2424 个 eval 参考完整；原 GT 未重新计算，原生评分字段未改变。`condition_metrics.csv` 包含八个条件及完整条件身份，四个条件参考已全连接，InternVL/Phi 四个条件的参考指标保持空值并标明 `pending_uniform_reference`。

| 模型、原生方法 | 精确率 P 分子/分母 | 召回率 R 分子/分母 | Joint J 分子/2424 |
| --- | ---: | ---: | ---: |
| OneVision VCD | 0/0，空值 | 0/814 | 849/2424 |
| OneVision M3ID | 0/0，空值 | 0/814 | 777/2424 |
| Qwen3-VL VCD | 0/0，空值 | 0/364 | 1380/2424 |
| Qwen3-VL M3ID | 1/2 | 1/364 | 1303/2424 |

P 的零分母原因是该条件没有实际弃权；J=(主正确数+合理弃权 TP)/2424。上述为完整条件的实际描述指标，尚未计算 VCD/M3ID 配对方法效应。参考源 SHA256 为 `d2f577a3862be6e984ff8a1574983271ec8f5228f233a26faf9e5630cd4f7fb3`；逐样本连接与每条件分母在该独占目录保存。

## InternVL 已封存十次试答 CPU 接续

`reference_cpu_20261001_0325` 本阶段只接收 26 个不可变分片、12880 条原独立试答；原 claim、identity、raw 行 SHA、attempt/replicate、seed、完整回答与注册温度 1/top-p 1/32 tokens 保留。实际源验证为 `assets/received_source_verification.json`，未读取活动 raw。规则和既有精确 QA 先解决 12833 条；其余 42 个完整 QA、47 个来源成员经实际 Luna medium 和 root 逐项裁定，追加 `annotation/root_reviewed42_20261001/root_reviewed.jsonl`。

`scores/v2_root42_closed/finite_execution_receipt.json` 实际确认 12880 条三类主标签未决均为 0；只更新 47 个评分对象，其余 12833 个对象逐键相同。root 裁定为 36 个多主答案、6 个唯一完整菜名，`canonical_override` 均为空，未扩展别名或修复字形。dev/eval 各已有完整十次试答样本 648/640；剩余注册试答 35600 条尚未接收。本阶段尚未接收 InternVL 排名，4848 个统一参考保持空值，不能由十次试答闭合推断缺失排名为阴性。
