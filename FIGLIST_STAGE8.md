# 第八阶段 · 图表定稿清单（FIGLIST_STAGE8，替换第七阶段版本）

论文使用的图，按正文出现顺序。规则（任务书 §5.3）：第一张图必须是"纠错变化与
置信度变化并排"的配对图；第二图按第八阶段主检验结果改为残差回归呈现，原分格
一致性散点移入附录。

| 序 | 图 | 内容 | 数据来源 | 结论指向 |
|---|---|---|---|---|
| 1 | `outputs/figures/stage6_fig1_core.pdf` + `stage6_fig1b_correction.pdf` | 纠错变化（救回/破坏率）与置信度变化（答案未变样本增量 + CI）并排，四模型×两组×四方法 | `outputs/tables/stage6_core.csv` | 第一层诊断：抬升覆盖全部样本、纠错只覆盖少数 |
| 2 | `outputs/figures/stage8_fig_residual.pdf` | 残差回归呈现：(a) 逐格残差差 γ_m（答错−答对）对人群干预前准确率，32 格 + 三条元回归线（FE/RE/未加权，权重敏感性即结论）；(b) 格内去均值残差按答对/答错的分布，标注汇总 γ = +0.0097 与六口径 CI | `outputs/tables/stage8_residual_regression.csv`、`stage8_metareg.csv`、`stage8_meta.csv` | 第二层机制 + 第八阶段分支判定：汇总选择性未确立（CI 含零）、格级两向并存 |
| 3 | `outputs/figures/stage6_fig2_fourcell.pdf` | 四格分解（一直对/一直错/由错变对/由对变错）的平均增量 + CI | `outputs/tables/stage6_fourcell.csv` | 残余成分的形态：被纠正格与一直错格的增量偏大 |
| 4 | `outputs/figures/stage6_fig3_did.pdf` | 分层差中差森林图（16 格全正、CI 不含零） | `outputs/tables/stage6_did.csv` | 校准后果的人群符号结构 |
| 5 | `outputs/figures/stage6_fig4_dose.pdf` | 剂量-响应：增量与 ΔECE 随 α 单调；变号点随 α 上移 | `outputs/tables/stage6_dose.csv` | 支撑一（幅度与强度的经验关系） |

不进入正文、留作附录/补充：`stage7_fig_cellwise.pdf`（原第二图：分格一致性
散点，被图 2 取代）、`stage6_variant` 对照表、`stage7_seedrobust`、
`stage7_dolasubset`、`stage7_dogspredict`、`stage8_dogs_transfer`、`stage7_sid`、
`stage7_sidunit`、`stage7_m3idgate_*`、`stage8_m3id_gate_recheck`、
`stage7_poscontrol`、`stage7_drift`、`stage7_distlevel`、`stage8_equivalence`
（审稿追问时指向）。

无新增测量；本清单即封版清单（项目封版，不再有实验或再分析）。
