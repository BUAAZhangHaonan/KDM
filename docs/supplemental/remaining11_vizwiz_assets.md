# 剩余11模型VizWiz资产与缺口

2026年9月30日的独立CPU审计已完成。原VizWiz官方验证集保留4,319题，dev 818题、eval 3,501题；全部题目的官方十份答案、人工可回答性、问题、图像身份、确定split和实际图像路径均已核对。11个补充模型在当前 `registered_inclusion.json` 中全部入选。

11模型的95,018条guided/unguided普查原回复已按原始task key、模型、问题、完整回答、样本字段和源身份与最终闭环标签逐条连接。行为未决为0，既有回答span未决为0。每个模型的3,501条eval guided Direct/UNKNOWN已生成独立来源绑定派生文件，仅调整正式任务的kind与key，保存original_key、original_identity、源路径、行号、逐行SHA及原闭环标签。原普查和历史标签保持原样。

| 模型 | 普查原回复 | 正式登记条件 | 正式复用 | 正式剩余 | 独立试答剩余 |
|---|---:|---:|---:|---:|---:|
| gemma3_12b | 8638 | 64 | 3501 | 220563 | 35010 |
| glm46v | 8638 | 64 | 3501 | 220563 | 35010 |
| internvl35_8b | 8638 | 80 | 3501 | 276579 | 35010 |
| llava15_7b | 8638 | 80 | 3501 | 276579 | 35010 |
| onevision | 8638 | 80 | 3501 | 276579 | 35010 |
| minicpm45 | 8638 | 64 | 3501 | 220563 | 35010 |
| phi35 | 8638 | 80 | 3501 | 276579 | 35010 |
| qwen3vl | 8638 | 64 | 3501 | 220563 | 35010 |
| qwen35_9b | 8638 | 64 | 3501 | 220563 | 35010 |
| llava15_13b | 8638 | 80 | 3501 | 276579 | 35010 |
| llava16_vicuna | 8638 | 80 | 3501 | 276579 | 35010 |

正式登记为800条件×3,501题=2,800,800条，复用38,511条，尚缺2,762,289条。独立试答在eval每题运行10次，共385,110条，归档可复用VizWiz独立记录为0。此前实际原始数据审计确认早期all16 probe及补充独立队列中的这些来源均为Food-101；该结论沿已有逐源审计接续。有限历史raw索引中没有未分类的11模型来源。本次所需canonical census、官方原始标注与最终闭环labels均已存在，新增tar恢复成员为0。

普查中的32词元预算耗尽与样本覆盖分别记录。每个模型均有8,638条合法完整JSON记录；预算耗尽数量见 `asset_manifest.json` 的 `token_budget_exhausted`。已有生成回复精确复用，保存其真实termination状态。

原冻结协议 `docs/current/PREREGISTER.md` 的“合理性测量”条款明确规定：VizWiz分别报告人工可回答性与可回答题上的独立作答成功率，两种原始弃权合理性证据在表格中分列。该文档的当前不可变副本为 `data/provenance/frozen_contract/blobs/77a2d30ff32636c0c4481fe72480226ca9c31850f4b0493aa5f9a0fcdff84c03`。

人工不可回答题为dev 249题、eval 1,136题；人工可回答题为dev 569题、eval 2,365题。官方答案引用保存在 `official_reference.jsonl`。评分调用原 `kdm.scoring.vqa_score`，使用vendored官方VQA词语归一化和十答案leave-one-annotator-out一致性分数。`kdm.analysis.probe_summary` 对人工不可回答题保留空的平均正确性，同时单列正信用、满分信用和十次试答。当前缺少全部VizWiz独立试答，重复试答参考尚未完成；原注册未定义把这些字段合成统一二元模型知识缺失GT，因此此字段保持未定义。

独立输出目录为 `outputs/supplemental/remaining11/run_20260930_140337/assets/vizwiz/audit_20260930`。目录包含逐模型两份真实未完成任务键、22行阶段缺口表、800行条件覆盖表、来源与标签路径、官方答案引用和CPU验证回执。脚本为 `workflows/supplemental/remaining11/vizwiz_inventory.py`；没有修改Food评分、生成器、分析入口或状态文件。

Qwen35-9B的两份CPU `--check-plan` 均已实际通过。独立计划为35,010条、temperature 1、top_p 1、32新词元；正式计划为220,563条，原矩阵224,064条减去已复用的3,501条Direct/UNKNOWN。正式缺口按既定执行顺序包括UNKNOWN主方法14,004条、UNKNOWN控制17,505条及其余提示矩阵189,054条。检查核对原始输入字节、冻结算法blob、native/method证明、缺口键唯一性及完整登记归属。K100接续使用root已经验证的当前runtime，在Food candidate完成且槽位可用时读取上述独立VizWiz缺口文件。
