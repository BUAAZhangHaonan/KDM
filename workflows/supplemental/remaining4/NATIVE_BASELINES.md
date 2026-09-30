# 四模型无引导补充入口

`native.py` 的范围为 InternVL3.5-8B、OneVision、Phi3.5、Qwen3-VL，各模型运行 VCD/M3ID 的 Food-101 eval 单条件。主路与参考路均使用原始无引导问题，标记字段为 `NONE`，`kind=native_unguided`，replicate 为零。完整计划为八个条件、19,392 条，每个条件 2,424 张图像、101 类各 24 张。

参数读取冻结的 `configs/kdm/study.json` 与 `DecodeConfig`：VCD alpha=1、beta=0.1，M3ID lambda=0.02、threshold=0.3，greedy、32 tokens。M3ID 每条记录的 offset 沿用 `pipeline.run_tasks` 对原始问题的实际 `backend.encode` 长度。原 dtype、图像处理、tokenizer、chat 模板、factory 和 device_map 使用冻结模型登记。Direct 使用已有无引导 census 来源及其原始身份。

## 2026-09-30 历史 CPU 验证

中央 `/home/g203-4028/projects/knowledge-deficit-mitigation/venv/bin/python` 已完成语法检查、原输入/原算法/原 native 与 method proof 核对以及八个实际缺口计划。结果位于 `outputs/supplemental/remaining4/plan_20260930_native/manifest.json` 与 `checks_cpu/verification.json`：19,392 个唯一缺口键；每个条件两个样本分片分别为 1,232、1,192，交集为空，合并覆盖全部 2,424；CUDA initialized 为 false。

d4030 独立目录 `/home/d4030/projects/knowledge-deficit-mitigation/supplemental/remaining4/runtime_20260930` 中，原字节迁移的 `venv/bin/python` 已完成 InternVL VCD/M3ID 两份 CPU 计划，各 2,424 条。`d4030_internvl_vcd_plan.json` 与 `d4030_internvl_m3id_plan.json` 保留在其独立计划目录。历史 CPU 迁移回执为 `receipts/cpu_admission_projecttmp.json`。

## 2026-10-01 已完成原生生成与评分

八个原生条件现已全部生成并完成标签闭合：`outputs/supplemental/remaining4/native_cpu_20261001_0226/checkpoints/v4_19392_all_native_closed/verification.json` 实际验收 19,392 个唯一键、八个完整条件、101×24 配额，主正确性、literal 和弃权未决均为 0。原精度、参数、算法和 claim 所有权保持原登记。

16 个 claim 的实际首八条 operator gate 均为 `passed=true`；GPU 准入、模型身份和首八条原算子证明来自原运行回执。本次文档更新只读取已接收的小型证明。下表每个 claim 基名分别连接 `_first8` 与 `_remaining2416` 两个 claim；其 `operator_audit.json`、`admission.json` 与原 SHA 保存在对应接收目录。

| 模型、方法 | claim 基名 | 中央接收目录前缀 |
| --- | --- | --- |
| InternVL VCD | `d4030_internvl35_8b_vcd` | `native_cpu_20260930_2338/received/` |
| InternVL M3ID | `d4030_internvl35_8b_m3id` | `native_cpu_20261001_0115/received_delta/` |
| OneVision VCD | `6403_onevision_vcd` | `native_cpu_20260930_2338/received/` |
| OneVision M3ID | `6403_onevision_m3id` | `native_cpu_20261001_0115/received_delta/` |
| Qwen3-VL VCD | `6403_qwen3vl_vcd` | `native_cpu_20260930_2338/received/` |
| Qwen3-VL M3ID | `6403_qwen3vl_m3id` | `native_cpu_20261001_0115/received_delta/` |
| Phi VCD | `6403_phi35_vcd` | `native_cpu_20261001_0115/received_delta/` |
| Phi M3ID | `6403_phi35_m3id` | `native_cpu_20261001_0226/received_delta/` |

上述目录前缀均位于 `outputs/supplemental/remaining4/`。逐 claim 证明由各来源评分目录的 `received_source_verification.json` 连接，合并回执为最终检查点的 `coverage_receipt.json` 与 `verification.json`。OneVision/Qwen3-VL 已有完整参考通过独立 `reference_join/one_qwen3_v1_20261001/receipt.json` 连接；InternVL/Phi 的十次试答、排名与 uniform reference 仍有实际缺口，原生条件评分完成不等同于完整参考完成。

## CPU 命令

从项目目录执行，TMPDIR 指向已有项目临时目录：

```bash
venv/bin/python -m py_compile workflows/supplemental/remaining4/native.py workflows/supplemental/remaining4/native_audit.py workflows/supplemental/remaining4/verify_native_plan.py
venv/bin/python workflows/supplemental/remaining4/native.py --prepare-manifest --manifest-dir outputs/supplemental/remaining4/PLAN_NAME
venv/bin/python workflows/supplemental/remaining4/verify_native_plan.py --manifest outputs/supplemental/remaining4/PLAN_NAME/manifest.json --output-dir outputs/supplemental/remaining4/PLAN_NAME/checks_cpu
venv/bin/python workflows/supplemental/remaining4/native.py --check-plan --model internvl35_8b --method vcd --missing-keys outputs/supplemental/remaining4/PLAN_NAME/internvl35_8b_vcd_missing_keys.jsonl --key-start 0 --key-stop 8 --plan-output outputs/supplemental/remaining4/PLAN_NAME/internvl_vcd_probe_plan.json
```

计划与缺口文件使用独占新目录。完成片段可以通过重复的 `--completed-raw` 参数提供给 `--prepare-manifest`；入口核对实际 key、raw identity、原配置、稳定 seed、完整记录与原始来源后扣除完成键。已经完成的八条算子检查回答可直接进入后续评分。

## GPU 执行接口

root/generation 使用原注册解释器、物理 GPU、GPU worker 锁与主线程的跨机所有权登记执行：

```bash
REGISTERED_PYTHON workflows/supplemental/remaining4/native.py --execute --model MODEL --method METHOD --missing-keys GAP_FILE --key-start START --key-stop STOP --shard SHARD --n-shards N_SHARDS --run-name RUN_NAME --claim-id CLAIM_ID --owner OWNER --host-registry REGISTRY_FILE
```

`--host-registry` 默认使用独立 remaining11 主机登记；也可显式传入已验证的新登记文件。入口复用 `remaining11.execution.validate_supplemental_runtime` 的 registered runtime、GPU UUID、继承锁、环境版本和 checkpoint 检查。d4030 仅登记物理 GPU0/1，InternVL 原 36 层双卡 factory、18/18 device_map 与每卡 22GiB max_memory 沿用原值。

同一输出根目录中的原子 claim 检查完整 task keys 并拒绝交集；跨机区间与分片由 root/generation 的统一队列分配。每个 claim 首个片段最多八条，观察同次真实生成的原处理输入、匹配前缀与完整词表 logits。VCD 核对 step500 processed noise 及支持集合；M3ID 核对原文本参考输入、offset、阈值、activation 与动态权重。实际 token 必须等于真实 pipeline distribution argmax，官方算子分数差采用既有 1e-8 数值容差。审计通过后保存其实际输入、prefix、trace、token 与 EOS 证据，再继续剩余片段。

raw、identity、逐片完成回执、operator_audit、claim 与 progress 分别保存在新的 `outputs/supplemental/remaining4/RUN_NAME`。真实失败保存来源与受影响键，并直接终止本次 claim；恢复需要独立的新缺口计划和主线程的明确所有权安排。
