# DoLa / SID 有限补测

用户于 2026-10-04 授权更新实现，并先以小批比较决定是否需要全量。当前授权范围固定为每类一题的 Food 101 题代表清单：DoLa 九模型、SID 六模型，共 1,515 条。每条件前八题已计入 101，接续入口精确复用这八题。没有启动新的 2,424 题全量条件。

## 实现

- DoLa 使用作者固定 commit `805230e57e63ca561cb759994681b122ff6f81f0` 的反向 KL 组合选择提前层。提前层候选范围和 repetition penalty=1 保持本研究登记设置。
- SID greedy 使用作者固定 commit `127dd412fa6b61ab1c9babf6979ec4da98002438` 的未屏蔽对比分数，补测显式设置 alpha=0.5。实际注意力选择与固定保留 100 个视觉 token 不变。非 greedy 路径仍保留原有 beta 支持；VCD alpha 与支持不变。
- 这次对齐作者核心算子，不宣称逐项复刻作者 benchmark 的全部模型、层桶、精度与生成参数。每个模型继续使用其真实登记精度；OneVision 为 float16。
- 活动源码已更新；旧算子仅在有限同前缀审计函数中保留，来源为 `ede009c7`。没有生产旧/新实现开关。历史正式预测和评分保持原来源，作为配对对照。

`tests/test_author_baseline_operators.py` 对 96 组构造分布与作者 Torch 表达做 CPU 数值对照：层选择、支持、argmax 与归一化输出通过；最大 DoLa log-probability 误差 7.105e-15。该测试不作为真实效果实验。

## 入口和数据

- 生成：`workflows/paper_core/author_baseline_pilot.py`。实际生成后，在每条历史回答前缀的前三个位置计算旧/新算子；它们共享本次真实前向，不冒充完整历史运行重放。
- 评分：`workflows/paper_core/score_author_baseline_pilot.py`。旧标签直接读取冻结完整主表；新回答优先复用相同完整 QA 和现有规则，边界留给人工语义裁定。未决行不记错，未闭合条件的 J 为空。
- 代表清单：`workflows/paper_core/attribution_representative101.csv`。采用既有每类一题，不按补测结果挑样本。
- 活动输出：各执行主机 `outputs/paper_core_20261002_dev_viz/closeout_20261004/baseline/<model>_<method>_101/`。
- 每条件包括实际配置、身份、claim/PID/starttick、原始 token/终止状态、原回答路径与行号、最多 303 个同前缀位置、完成回执及 SHA。
- 对应 `<model>_101.launch.json` 和 `launch_<model>_101.sh` 保存实际 Python、registry、物理 GPU、命令和锁。使用原 `outputs/locks/gpu_N.lock`，没有新锁体系。
- `history/` 保留从已完成/已封存原 raw 精确取出的 15×101 条历史原文，来源含原主机、路径、行号、原任务键与完成回执 SHA。

实际执行必须先读取已通过的项目清理回执，核对冻结输入、模型证明、共享算法源码，以及本次明确授权的新 decoder SHA。不得把历史全量条件重命名为新实现全量。

## 判断范围

有限表同时保留旧/新 C、W、A、TP、J 和九格转换，分母为真实 101。后续与 IP、原生 VCD 或 CDA 比较，应从冻结结果选择相同 101 输入；不混用 101 与 2,424 的分母。仅凭八题或文本变化率，不能声称全量结果等价或不等价，也不据此自动全量重跑。

旧 2,424 题结果没有更新生成时，仍标注其原登记算子身份。完整输出覆盖和语义评分闭合分别验收。该文档登记执行范围，不以计划数冒充完成数。
