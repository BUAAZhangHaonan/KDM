# 已保留历史 Food 来源的有限 CPU 接续

2026-10-01 22:04（北京时间）实际运行完成。范围仅显式 accepted-key 文件中的 51,569 个 Food 合法任务键；没有追加生成、API、GPU 工作或新人工标注。

| 模型登记键 | 接续行数 | 原评分复用 | 新封存回复评分 | C / literal / A 未决 | config hold | 参考已连接 | 完整条件 |
|---|---:|---:|---:|---:|---:|---:|---:|
| internvl35_8b | 2,450 | 2,450 | 0 | 0 / 0 / 0 | 0 | 2,450 | 1 |
| onevision | 6,823 | 6,823 | 0 | 0 / 0 / 0 | 0 | 6,823 | 1 |
| phi35 | 16,998 | 16,554 | 444 | 0 / 0 / 0 | 0 | 16,998 | 4 |
| qwen3vl | 25,298 | 25,298 | 0 | 0 / 0 / 0 | 0 | 25,298 | 7 |
| 合计 | 51,569 | 51,125 | 444 | 0 / 0 / 0 | 0 | 51,569 | 13 |

51,125 条完整历史对象另存 `historical_score_rows.jsonl.gz`，原 canonical/A 字段逐对象差异为 0。仅四个精确 QA 成员依据已存在且真实 root 裁定的 `root_literal_cone1.jsonl`，将 literal 从 null 增量改为 0，主正确性与弃权不变。

旧 record 没有 config 字典。本次使用原 DecodeConfig/run_tasks 注册计算，或实际已封存同模型配置，重建 config 后逐行匹配原 `config_sha256`；51,569 行全部相等。OneVision、Qwen3-VL 的 instruction-M3ID offset 使用原登记环境中的 tokenizer，编码实际无引导 task_prompt（包括 `Give a concise answer.`）得到 14；原 tokenizer 文件 SHA 与版本已校验，未加载模型权重。Phi 使用实际 sealed row 的 offset/config。无证据匹配时入口会保存 hold，本次 hold 为 0。

444 条新增评分来自两个已 administratively stopped、独立 sealed 的 Phi partial（315 与 498 原物理行中，精确选取 189 与 255 合法键）。stop、seal、complete、identity、raw SHA、原输入、任务键、种子、有限数值与原 config 均实际验证；原 claim 完成状态保持 false。只读这两个已封存成员，没有打开旧大 raw 或 active raw。采用现行 canonical/abstention 函数、最终 census 行为与现有精确 QA authority，边界队列实际为空。

参考由 `reference_full_20261001_1600/checkpoints/v2_reviewed_role_reference/reference_G.jsonl` 按 model/sample/split/class 连接；51,569 行全部参考完整。参考公式与旧五模型资产没有修改。

13 个完整条件分别为 Intern Direct/UNKNOWN；OneVision Direct/UNKNOWN；Phi Direct、DoLa、DeCo、SID/UNKNOWN；Qwen3-VL Direct、DoLa、DeCo、SID、IP-VCD、IP-M3ID、CDA/UNKNOWN。每个实际包含同一 eval 2,424 键，101 类各 24；其余条件保留部分覆盖，不作为完整分母。完整 condition ID、全部 marker、参考引导开关与 config SHA 见 `condition_coverage.jsonl`。

## 输入与输出

中央输出独占目录：

`outputs/supplemental/remaining4/historical_retained_cpu_20261001_2145_v2_registered_prompt/`

- `score_rows.jsonl.gz`：51,569 已决评分及历史、配置、参考来源。
- `partial444_score_rows.jsonl.gz`：本次两 sealed partial 的 444 实际新评分。
- `historical_score_rows.jsonl.gz`：51,125 原评分对象副本。
- `literal_delta.jsonl`：四个 source-bound literal-only 修正。
- `condition_coverage.jsonl`、`receipt.json`、`verification.json`：覆盖与逐对象验收。
- `boundary_queue.jsonl`、`config_holds.jsonl`：实际均空。

主要源：

- accepted keys：`audit_20261001_1600/key_cache_v2_phi_sealed/accepted_formal_keys.jsonl`，SHA `b2a5b78126423eff2702094ab290cc11fbac1a4da0915c7824245f3297880aee`。
- 原评分：`outputs/supplemental/remaining11/run_20260930_140337/scores/selected4_v10_root46/score_rows.jsonl.gz`，SHA `f96f04a2cad796a8c2ae7a1dbc39c8e28951553e1ea9d24b6fa57816284a745a`。
- 停点来源：`dispatch_20261001_1600/formal_source_audit_phi/receipt.json`；两个源原 SHA、identity、stop/seal 证据完整保存在本次 receipt。
- 参考 v2 SHA：`d3feac8fef2ade5967c0ff17762e15a17aac92e244f980c01d6ed360ff8d36a0`。
- literal root authority SHA：`78cb0bb7a8545db04490a1118d0750bc6bc3b15217f126dfa70ccf3b84636c8f`。
- 新评分 SHA：`c2e0f502ae3e3fc1e08e91adb602df60193c245fe10f0a9a1ade0737a09ef45b`。

新增 source：`workflows/supplemental/remaining4/recover_retained_historical.py`（SHA `b2b5b0a51c326f661d0708561a755a80ede1bfc402d746546ff8913e88e0731a`）与 `historical_tokenizer_config.py`（SHA `ea451de6607b5b521c21dffa25845a43dc1689d7f282b503acba57fadfbbcd06`）。实际中央 CPU Python 为 `/home/g203-4028/projects/knowledge-deficit-mitigation/venv/bin/python`。

## 实际运行与验收

在中央项目根运行以下命令（实际 SSH 调用使用完整绝对 source 路径）：

```sh
venv/bin/python workflows/supplemental/remaining4/recover_retained_historical.py \
  --output outputs/supplemental/remaining4/historical_retained_cpu_20261001_2145_v2_registered_prompt \
  --tokenizer-proof outputs/supplemental/remaining4/historical_retained_preparation_20261001_2145/onevision_tokenizer_registered_prompt.json \
  --tokenizer-proof outputs/supplemental/remaining4/historical_retained_preparation_20261001_2145/qwen3vl_tokenizer_registered_prompt.json
venv/bin/python -m py_compile workflows/supplemental/remaining4/recover_retained_historical.py workflows/supplemental/remaining4/historical_tokenizer_config.py
```

两命令 exit 0。主入口实际写入后自行执行输出验收：51,569 唯一键；51,125 历史对象原字段逐个保留；四个 literal-only delta；51,569 config SHA 相等；参考完整；13 个全量条件。旧裸 question token proof 与首次空输出目录作为失败准备历史保留，未用于 accepted 结果。

本地小型回执镜像：`E:/OneDrive/文档/Playground/kdm_remaining11_20260930_140337/remaining4/historical_retained_cpu_20261001_2145_v2_registered_prompt/`。大评分文件留中央数据路径；原五模型、论文正文、旧 panel/manifest 与 CURRENT_STATE 未修改。
