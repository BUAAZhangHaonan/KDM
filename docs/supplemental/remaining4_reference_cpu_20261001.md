# InternVL 封存试答与排名参考接续

本次有限 CPU 接续已完成 12880 条真实独立试答评分和 408 条封存排名验收。来源分别为原 d4030 两个十答 claim 的 26 个不可变分片，以及 6403 的 400 条排名加 d4030 原首 8 条排名。未读取活动 raw，未初始化 GPU，没有新增生成或收费 API。

## 试答评分与有限实际复核

独占目录为 `outputs/supplemental/remaining4/reference_cpu_20261001_0325`。`assets/received_source_verification.json` 逐源核对注册模型、原参数、完整 prompt、原 input/algorithm provenance、task key、claim、identity、raw 行 SHA、seed 与 replicate 0–9，显式 attempt ordinal=replicate+1。

原 `scores/v1_rules_intern_sealed12880` 保存规则及精确 QA 复用结果：12833 条已决，47 条对应 42 个完整 QA 边界。实际 Luna medium 批次 `intern_sealed12880_boundary42_luna_medium_20261001` 已成功写入；root 完整阅读该批问题、回答和 Luna 决定后逐项裁定，并另存 `annotation/root_reviewed42_20261001/root_reviewed.jsonl`。作者为 `/root`、`gpt-6.1-sol`、`max`，写入者为 `/root/assets`，未知 call ID 为空。原规则结果与 Luna 文件保留。

`scores/v2_root42_closed/finite_execution_receipt.json` 验收 12880 个唯一试答键、三类主标签未决均为 0，仅 47 个评分对象更新，其余 12833 个对象逐键相同。42 个 root 裁定中有 36 个多主答案、6 个唯一完整菜名；`canonical_override` 均为空。完整名称修饰、词序、字形与同菜同位语保留，由原 101 类规范词边界函数计算每个目标类别的正确性，不扩展别名。

## 排名与统一参考的实际交集

`rank408_received/received_manifest.json` 保存 13 个来源分片的实际接收证明。其中 12 个 6403 分片共 400 条；d4030 原 first8 独立按其原 part0 receipt 验收，不将 400 条宣称为 408。`connect_reference_rank.py` 验收每条 101 个候选的有限 `sum_logp`、`mean_logp`、`n_tokens`、原 gold rank、精确 closed prompt、sample、key、identity 与稳定 seed。全部原候选分数和来源版本保存于 `reference_join/v3_rank408_attempt12880/rank_rows_source_bound.jsonl.gz`。

6403 已注册生产来源包括 clone-prefill 和 readonly-prefill 两个版本；分别连接其原 pinned gate/SHA。两个历史 gate 都使用 8 个真实输入、808 个候选，sum/mean 分数最大差为 0、rank 差为 0；readonly 原 gate 的 tensor integrity 也通过。本 CPU 接续只验收这些已完成原证明，未重新运行 GPU 算子。初次薄适配未识别 readonly 字段版本及后续相对路径输出错误均保留于 `metadata/connect_rank_first_cpu_failure.json`、`connect_rank_second_cpu_failure.json` 及对应实际执行源码快照；最终 CPU 连接实际 exit 0。

| split | 已有完整十次试答样本 | 排名已验收 | 统一参考完整 | 完整参考阳性 | 仍待连接 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 648 | 216 | 216 | 123 | 2208 |
| eval | 640 | 192 | 192 | 85 | 2232 |
| 合计 | 1288 | 408 | 408 | 208 | 4440 |

当前 `reference_G.jsonl` 保存全部 4848 个注册样本。参考沿用 `gold_rank>1` 且十次当前主正确次数为 0 的原公式；只有实际 rank 与完整十次已决试答相交时才生成布尔 GT。缺 rank、少于十次或未决试答保持空值，分列 missing reason；每次 attempt、correct_count、split、原身份和来源继续保存。已有接受的 v10 元数据中，该模型 historical accepted 字段本次可连接非空数为 0，仍另列其来源与空值，未改写历史裁定。

最终连接回执：`reference_join/v3_rank408_attempt12880/receipt.json`，SHA256 `d19a4ed3288a9d03d45e9a8e9a267c8501c6b0c7cd748e731b310b28ab4de858`。参考表 SHA256 `13c1b9b7b574f1fae0469b9e7ef897938022eca71e71c2928436165899a58e7e`；connector 实际执行源码 SHA256 `b1904ae3ded300e83f952631ef397af43956e32c686de84203f429dd197d590b`。

本阶段尚余 35600 条注册十答未接收、4440 条排名未接收，InternVL eval 的完整 2424 分母尚未具备参考连接，不能由已有 192 个参考子样本替代整条件 P/R/J 分母。未开展是否应该弃权的交叉验证标注。OneVision/Qwen3-VL 的原生条件完整参考薄连接由独立 `native_cpu_20261001_0226/reference_join/one_qwen3_v1_20261001/receipt.json` 保存；其紧凑表为 `compact_20261001/compact_native_reference_metrics.csv`。
