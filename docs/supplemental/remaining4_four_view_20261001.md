# 四个扩展模型的有限四路测量

2026-10-01，独立补充路径。原五模型数据、评分和正文保持冻结。

## 实际完成与核验

复用原 `attribution_representative101.csv` 的 101 个 Food eval 输入，每类一图。四个扩展模型均已完成 101 个回复首位的四路前向，以及同输入 101 条无引导扰动图像 Direct 自回归回复。合计 **404 个首位测量、404 条真实新回复**。各模型先测固定 8 题，验收后只接其余 93 题。

| 模型键 | 首位测量 | 扰动图回复 | 累计 GPU 秒 | 扰动图正确数 / 101 | 语义弃权数 |
|---|---:|---:|---:|---:|---:|
| internvl35_8b | 101 | 101 | 424.7764 | 13 | 0 |
| onevision | 101 | 101 | 232.5183 | 11 | 0 |
| phi35 | 101 | 101 | 117.5035 | 13 | 0 |
| qwen3vl | 101 | 101 | 120.6499 | 19 | 0 |

累计 **895.4480 GPU 秒，约 0.24874 GPU 小时**，含载入、小批准入、四路前向和实际扰动图生成；InternVL 的双卡时间已计入。原 checkpoint、精度、双卡映射、处理器、模板、扰动 seed 与算法复用原登记。没有新增带引导 VCD 的自由生成网格。

中央 CPU 核对了原样本、唯一键、101 类配额、提示、共享 token 前缀、噪声和 seed、有限候选分布以及原公式。404 个位置、404 个候选对均通过；float64 公式闭合残差最大 **1.4211e-14**。一个 `2.2204e-16` 的 argmax 数值并列保留在核验收据中。

自然参考回复按已接受的主菜、101 规范名和语义弃权流程评分，复用精确 QA；唯一新边界经过真实 Luna medium 标注和 root 阅读完整原文后的独立追加复核。最终 404 行的 canonical、literal、弃权、边界未决均为零。模型—样本的统一参考连接另行完成，连接前 P/R 保留空值。

## 解释范围

四路使用正常/扰动图像与有/无 UNKNOWN 指令，同一实际生成 token 前缀。候选对采用原生 VCD 回复首 token 与字面 UNKNOWN 首 token；后者是机制测量的词元代理，不能当作完整语义弃权标签。共同支持与实际支持字段分别保留。

这批代表输入上的无引导扰动图回复全部为正常作答，准确率为 10.89%–18.81%。它直接描述这批回复的行为，尚不能替代自由生成的原生 VCD/IP-VCD 方法比较。

**诊断分岔位置与完整真实案例重放尚未完成。** 需要从已决的实际方法比较选择有限输入，保存各类实际覆盖与缺项，再接续到既定归因预算。不得将首位面板写成全部归因任务完成。

## 可复算来源

- 准备：`outputs/supplemental/remaining4/four_view_sources_20261001_1621/source_receipt.json`。
- 已封存 GPU 来源：`outputs/supplemental/remaining4/four_view_received_20261001_1621/`，四模型分别有 `gate8` 与 `tail93`。
- CPU 核验及稀疏数据：`outputs/supplemental/remaining4/four_view_verified_20261001_1621/all4_full101/`。
- 独立最终评分：`outputs/supplemental/remaining4/natural_reference_scoring_20261001_1700/rules_v2_rootreviewed/`。
- 实际 Luna 与 root 文件保存在上一路径的同级 `annotation/`，旧裁定不覆盖，未知 call ID 留空。

新增薄入口为 `prepare_four_views.py`、`measure_four_views.py`、`verify_four_views.py`、`score_natural_reference.py`。原 `src/kdm` 算法、原五模型入口和其输出均未修改。
