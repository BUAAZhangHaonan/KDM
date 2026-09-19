# 必须产生的执行记录

`outputs/records/environment.json`：实际环境与模型版本。
`outputs/records/migration.json`：原仓库提交、归档路径和提取源指纹。
`outputs/records/model_census.json`：每个候选模型、全部题数、完整弃权数、入选条件。
`outputs/records/backend_equivalence.json`：各模型原生generate与新后端的逐词元检查。
`outputs/raw/current/`：普查、独立试答、基线、替代方法、共享前缀、闭集排序。
`outputs/annotations/current/`：完整回答语义标签、证据与简短答案原文。
`outputs/tables/current/`：所有方法、模型和提示条件的完整汇总。
`outputs/figures/current/`：正式真实图及绘图输入。
`docs/current/RESULTS.md`：每条主张、支持结果、范围、对应图表。
`docs/current/DEVIATIONS.md`：工程变化、科学设置变化及原因。

完成状态以数据清单中的每一个任务ID为准。异常、空输出和解析失败不能计作成功完成，也不能静默从分母排除。所有未终止回复保留实际token预算和完整文本。

VQA官方源码保持下载原文与blob指纹。供当前评分调用的文件只提取两个规范化方法，以及它们在构造函数中使用的六个字段。数据集API对象与完整评价器状态无需初始化，因此不会通过空VQA对象调用问题索引。规范化规则保持官方方法原文的运算。

DoLa与DeCo的共享前缀重放记录实际选中层、系数、候选支持和组概率分解。SID使用它实际构造的参考会话。M3ID的四条件恒等式在相同原始权重下计算；新方法自己的无引导门控权重另列，避免把两种权重混为一项观察。
