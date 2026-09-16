# 第九阶段 · 数据源差异核对（SOURCE_DIFFERENCES）

核对日期：2026-09-17。核对执行：ZCode（服务器端，直接读取仓库真实表格）。

## 1. 提交版本核对

交付包源提交：`e5eda213dd3f51c08b43c1616305c93b84011dff`。
服务器主分支当前提交：`e5eda213dd3f51c08b43c1616305c93b84011dff`（本地与 `origin/master` 一致，工作树干净）。

**服务器主分支没有比交付包源提交更新的提交**，因此不存在"包内数字落后于服务器"的情形；
以下核对均为交付包数据与服务器同一提交真实表格的逐项复核。

## 2. 三处事实的确证结果（对照 `00_SUPPLEMENT_FOR_ZCODE.md` §1）

| 项目 | 补充说明取值 | 服务器真实表复核值 | 结论 |
|---|---|---|---|
| 高准确率子集净准确率下降组合数 | 14/16 | `outputs/tables/stage6_core.csv` 高准确率 16 组：14 组 `acc_method − acc_direct < 0`；2 组为正 | 一致，采用 14/16 |
| 两个净上升组合 | Qwen3.5-4B + DoLa +0.17 个百分点；InternVL3.5-4B + M3ID +0.83 个百分点 | q4b-dola 高组 +0.001667（+0.17 ppt）；internvl4b-m3id 高组 +0.008333（+0.83 ppt） | 一致 |
| 名称层增量下限 | 0.056937 | `outputs/tables/stage6_dname.csv` `d_name_level` 最小值 0.056937169854650964 | 一致，采用 0.056937 |
| Qwen3.5-4B + M3ID 低组端点 | 0.160 → 0.186667 | `stage6_core.csv` q4b-m3id 低组 acc_direct 0.16、acc_method 0.18666666666666668 | 一致，不采用 0.21 |
| InternVL3.5-4B + DoLa 高组下降幅度 | 98/600 = 16.33 个百分点 | acc_direct − acc_method = 0.163333（600 张分母） | 一致 |

包内 `data_sample/stage6_core_selected.csv`（GitHub 连接器转录）与服务器
`stage6_core.csv` 在上述各项上无差异。包内 13/16 的外部概述值不采用
（见 `10_DATA_NOTES.md` 与补充说明 §1.1）。

## 3. 命名修正记录（补充说明 §1.2）

`docs/stage8/CLAIMS_STAGE8.md` 与 `docs/stage8/MAINLINE_STAGE8.md` 中的
"LLaVA-1.5-16B" / "LLaVA-1.5" 已改为 "LLaVA-v1.6-7B（Mistral 骨干）"。
文件内部键 `llava16` 保留。第六阶段文档中"官方 SID 仅支持 llava-1.5 /
instructblip / shikra"指的是官方 SID 仓库自身支持的模型列表，不是本项目的
模型命名，不属本次修正范围，保持原样。纯命名修正，无数值变化。

## 4. 交付包完整性

`deliverables/KDM_next_stage/` 全部文件通过 `MANIFEST.sha256` 校验
（sha256sum -c，0 个失败）。包内真实样例表为固定提交的选择列转录
（`data_sample/PROVENANCE.json`），逐样本分析仍以服务器
`outputs/raw/*_s6_eval_naming.jsonl` 为准。
