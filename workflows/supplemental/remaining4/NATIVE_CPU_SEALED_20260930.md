# 四模型原生基线封存分片 CPU 接续

本次有限快照实际接收 13 个不可变 VCD 分片，共 3,632 条、3,632 个唯一任务键。源快照为中央 `outputs/supplemental/remaining4/receiving_manifest_20260930_2329/native_parts_snapshot_d4030_20260930_2327.json` 和 `native_parts_snapshot_6403_20260930_2327.json`，采集时间为 2026-09-30 23:28:58 附近。仅复制快照白名单中的 raw、identity、complete receipt、admission、operator audit、owner 和 claim keys。

当前接收与评分目录为 `outputs/supplemental/remaining4/native_cpu_20260930_2338/`，评分版本为 `scores/v1_rules_exactQA`。本地镜像为 `E:/OneDrive/文档/Playground/kdm_remaining11_20260930_140337/remaining4/native_cpu_20260930_2338/`。

| 模型 / 方法 | 已接收 | 尚未接收 | 已决正确 | 已决错误 | 主正确性未决 | 弃权未决 |
|---|---:|---:|---:|---:|---:|---:|
| InternVL3.5-8B / VCD | 2,064 | 360 | 395 | 1,653 | 16 | 1 |
| OneVision / VCD | 528 | 1,896 | 121 | 402 | 5 | 0 |
| Qwen3-VL / VCD | 1,040 | 1,384 | 533 | 504 | 3 | 0 |
| Phi3.5 / VCD | 0 | 2,424 | 0 | 0 | 0 | 0 |
| 四模型各自 M3ID | 各 0 | 各 2,424 | 各 0 | 各 0 | 各 0 | 各 0 |

表格的已决数量仅对应本次接收成员。每个完整条件的注册分母为 2,424；本次没有计算完整条件效应、Acc/P/R/F1 或配对统计。独立参考连接在该有限检查点标为 `not_connected_in_this_finite_sealed_partial_score`，每条 `reference_G=null`，完整参考由独立接续流程连接。

所有条件仍采用 `guided=false`、`reference_guided=false`、`marker=NONE`、`reference_marker=NONE`、`replicate=0`。CPU 来源验收逐项核对冻结模型登记、输入 SHA、算法 Git blob、同一个 native task_id、owner keys、稳定 seed、两路无引导 prompt、有限概率和 token 预算。参数沿用登记值：VCD alpha=1.0、beta=0.1；M3ID lambda=0.02、threshold=0.3；max_tokens=32、temperature=0.0、top_p=1.0。M3ID 的 offset 保留原算法依据实际 prompt tokens 计算的规则。

原 runtime、checkpoint、dtype、device_map 和主机 admission 保存于原 identity/admission。每个 claim 既有真实首八条算子 proof 已核验 `passed=true`，随后封存分片复用原 proof。接收保持远端原件和 claim 所有权，活动 raw 文件读取为零。

其中 27 条回答达到登记的输出 token 预算，原 `terminated=false` 状态及全部已生成内容均保留。每个分片的完成和截断数量保存于原 complete receipt 及 `received_source_verification.json`。

评分复用冻结的 `workflows/main_results/score.py` canonical 函数及已有 `infer_qa`、`score_target`。先复用完整问题和回答精确相同的最终普查行为标签、已审核 QA 与现有规则。实际行为来源为最终普查精确复用 3,147 条、已裁定精确 QA 82 条、规则 402 条、行为边界 1 条。独占 QA 为 974 个。

`boundary_queue.jsonl` 保存影响主正确性或弃权的 24 个独占 QA、25 个实际源成员，包含完整问题、全部已生成回答、原 task key、identity、source line、line SHA 与具体未决字段。主正确性未决为 24 条，弃权未决为 1 条；完整名称评分未决为 27 条。未决字段保持空值。该边界包交给真实 Luna medium 和 root 后续复核。

实际 CPU 命令：

```bash
TMPDIR="$PWD/.cache/tmp" venv/bin/python workflows/supplemental/remaining4/score_native.py --received-manifest outputs/supplemental/remaining4/native_cpu_20260930_2338/received/received_manifest.json --output outputs/supplemental/remaining4/native_cpu_20260930_2338/scores/v1_rules_exactQA
TMPDIR="$PWD/.cache/tmp" venv/bin/python workflows/supplemental/remaining4/verify_native_score.py --score-dir outputs/supplemental/remaining4/native_cpu_20260930_2338/scores/v1_rules_exactQA --output outputs/supplemental/remaining4/native_cpu_20260930_2338/scores/v1_rules_exactQA/verification.json
```

`received_source_verification.json` 与 `verification.json` 实际通过。独立验收验证唯一任务键、13 个源分片、八条件分母、实际已决/未决数量、类别部分覆盖和边界包源成员一致性；未知项转成已决项的数量为零。源码语法检查通过。旧 raw 扫描、旧评分重算、新 GPU 初始化、新生成、新 API 调用均为零。

评分文件 SHA256 为 `4ba45cfd2198378ca2d50296b1065eaea28e618d3f74bc7710828927eb61a232`，边界包 SHA256 为 `069e7db8ffc52a68e98c8caf72d34bc0fc167c0c35ae71dc50abbb5fc913f86c`。代码改动仅新增 `workflows/supplemental/remaining4/receive_native.py`、`score_native.py`、`verify_native_score.py` 和本文件。
