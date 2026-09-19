# KDM：对比解码中的弃权保持

KDM研究带有弃权引导的视觉问答中，对比解码如何改变模型原有的不确定表达，以及如何同时保持回答质量与合理弃权。

当前研究以四个相连的问题组织：原有弃权是否对应较低的独立作答成功率；压低一种弃权表达是否影响其他同义表达；不同参考构造产生的影响如何由分数关系解释；将弃权引导与视觉对比计算分开能否改善回答与弃权的共同表现。

## 当前执行状态

已迁移活动代码并保留历史。全量研究清单为9,167条；正式普查、干预、选择与论文结果尚未完成。缺失模型及CDA定义冲突见 `docs/current/EXECUTION_STATUS.md`、`docs/current/DEVIATIONS.md`，`docs/current/PREREGISTER.md`明确标记尚未冻结。

## 历史动机证据

用户交接记录报告：LLaVA-v1.6-7B在带弃权引导的1200张Food-101评测图像中产生72次原始弃权，VCD保留6次、替换66次；替代结果包含35个错误菜名、21个上位描述、9个其他物体名称、1个正确名称。原始完全匹配正确数量由280变为289。文件 `data_example/reported_counts.json` 标明这些数字来自交接记录。这些数字保留其历史身份，不能用作当前全量研究的正式结果。

代码基于远程提交 `eceed2246515cb938ad478ee23c94590c990eb5d`。正式执行时保留原始输出与提交历史，把阶段脚本移入归档目录。新的活动代码位于 `src/kdm/`，按任务与计算功能组织。

## 读取入口

- `docs/current/PAPER_STORY.md`：论文主线与章节安排。
- `docs/current/THEORY.md`：语义分组、参考偏好和指令保持的完整推导。
- `docs/current/STUDY_SPEC.md`：冻结实验、数据与报告规则。
- `docs/current/CODEX_PROMPT.md`：可直接交给CodeX执行的任务书。
- `docs/current/PAPERS_AND_STRUCTURE.md`：全文阅读与结构分析。
- `docs/current/MATH_PIPELINE_REVIEW.md`、`docs/current/MODEL_ADAPTER_REVIEW.md`、`docs/current/DATA_ASSET_REVIEW.md`：本轮实际检查。`verification/VALIDATION_REPORT.md`保留开发包交付时的软件验证身份。

迁移已经执行。原开发包保存在 `deliverables/KDM_research_refactor/`，旧代码与README保存在 `archive/pre_refactor_41388226b4fb/`。正式工程目录固定为 `/home/g203-4028/projects/knowledge-deficit-mitigation/`。

## 活动目录

`src/kdm/` 提供解码、测量、数据、标注、统计与模型接口。
`configs/kdm/` 保存固定模型清单、数据来源、提示条件和研究参数；`configs/runtime/`保存实际模型/环境身份。
`tests/` 检验数学计算、数据身份、语义标签与运行流程。
`scripts/` 提供安装、资源下载、模型发现、执行和制图入口。
`paper/` 保存中文论文结构和图表安排。
`source_materials/` 保存用户原始写作要求。

## 研究状态

现有数字、数学推导、已运行的软件测试、待执行的真实推理分别记录。新方法使用三个条件分布，增加一条无弃权指令的清晰图像计算；无需建立弃权词典。词元级与完整回答级的测量分开保存。方法的正确回答与弃权表现由正式实验给出。
