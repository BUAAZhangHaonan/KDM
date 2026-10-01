# 四扩展模型完成分片的评分接续

范围为 InternVL3.5-8B、LLaVA-OneVision、Phi-3.5-Vision、Qwen3-VL 的独立补充目录。五模型已发布评分、参考、配置和正文保持冻结。

`score_received.py` 只读取显式接回、SHA 验收且已完成的不可变分片；检查样本、身份、键、配置字段及有限值。新增普通带引导 VCD/M3ID 不准进入该入口。Food 复用当前主评分的完整回答提取、canonical/literal 及角色判定。

VizWiz 先复用最终 census 的完全相同问题与完整回答。其余只接受实际语义标注和已有注册的完整弃权标记；未决行为保持空值。主回答质量使用已决、原文连续的主答案 span 和官方连续共识分数；弃权/invalid 为零。完整 raw 的官方分数另存，不替代短答案分数，`unanswerable` 的官方原始计分保留。

`apply_food_reviews.py` 只对有限、来源绑定的真实复核 QA 追加裁定，其他评分对象保持一致。`prepare_viz_label_queue.py` 仅按完整 QA 去重和划分互斥所有权，不创建语义标签。

`join_received_references.py` 将已验收的 19,392 条 Food GT 连接到新增已决评分；核对模型、样本、split、类别和唯一键。连接不修改正确性、弃权或 Viz 对象。方法覆盖保留完整 condition 和解码配置；分片输入不足 2,424 时，不提升为完整方法比较分母。

实际验收目录 `outputs/supplemental/remaining4/received_reference_20261001_1940/`：81,692 条唯一新评分，34,536 条 Food 已连接 GT，47,156 条 Viz 对象完全不变。本批完成分片尚无单独覆盖全部 101×24 输入的方法条件，后续按不可变增量合并。对应六个原评分 cohort 及 SHA 写入 receipt。

实现已运行真实不可变分片评分、有限 Food 裁定、跨 cohort 去重与 GT 连接，并通过相关 Python 编译检查。大型 raw、评分、标注和来源回执留数据路径；这里仅提交执行入口与小型说明。
