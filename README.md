# KDM：视觉问答中的弃权保持

KDM研究对比解码如何改变视觉问答中的作答与弃权，并评估同时保持回答正确率和合理弃权的方法。

当前主结果覆盖Food-101上的五个固定模型：Qwen2.5-VL、Qwen3.5-4B、LLaVA-v1.6-Mistral、MiniCPM-V2.6与Gemma3-4B。实验包含直接解码、注册的VCD/M3ID/CDA/DoLa/DeCo/SID、保留指令分量的方法及对应控制组，按完整提示条件比较。

## 数据与进度

五模型正式方法输出共853248条，分为352个完整条件，每条件2424个eval样本。另有24240条候选记录和242400条独立回答。主指标标注与复核已完成，352个条件的指标、372组主配对比较和40组简单弃权对照均已计算。结果入口为 [五模型结论](docs/RESULTS.md)，英文论文为 [main.md](paper/main.md)。

审阅材料的阅读顺序与核查问题见 [REVIEW](docs/REVIEW.md)。

## 主指标

从完整回答中提取明确主菜，再匹配Food-101原类名或下划线转空格形式。配菜、修饰、竞争候选和明确弃权分别保存在抽取字段。具体评分规则见 [SCORING](docs/SCORING.md)。方法效果按相同样本、相同主marker和参考marker比较；配对区间使用Food-101类别聚类bootstrap。

## 目录

- `src/kdm/`：模型接口、冻结解码算法及实验计算。
- `workflows/main_results/`：最终评分、参考GT及统计入口。
- `configs/`：方法、数据与模型运行参数。
- `data/`：数据集清单、规范原始回答及参考GT来源。
- `outputs/annotations/main_results/`：最终问答抽取与逐样本评分。
- `outputs/analysis/main_results/`：条件表、配对统计、图表与结论。
- `docs/`：当前研究与复现说明。
- `paper/main.md`：论文稿。

历史运行记录及被替代代码保存在项目外的SHA校验档案；有效数据、原始回答、模型权重与来源清单保留可恢复关系。

## 数据来源

Food-101的101类及固定图像标签用于类别正确率计算，见 [Bossard等，ECCV 2014](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf)。本项目保存自由文本回答的明确名称抽取和弃权判断，供主结果与敏感性分析复现。
