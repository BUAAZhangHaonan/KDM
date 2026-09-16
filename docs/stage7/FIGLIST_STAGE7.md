# 第七阶段 · 图表定稿清单（任务书 §4.4）

论文使用的图，按正文出现顺序。规则：第一张图必须是"纠错变化与置信度变化并排"
的配对图；拟合曲线与变号点不得置于首位；四格分解图与剂量-响应图各一张。

| 序 | 图 | 内容 | 数据来源 | 结论指向 |
|---|---|---|---|---|
| 1 | `outputs/figures/stage6_fig1_core.pdf` + `outputs/figures/stage6_fig1b_correction.pdf` | 纠错变化（救回/破坏率）与置信度变化（答案未变样本增量 + CI）并排，四模型×两组×四方法 | `outputs/tables/stage6_core.csv` | 第一层诊断：抬升覆盖全部样本、纠错只覆盖少数 |
| 2 | `outputs/figures/stage7_fig_cellwise.pdf` | 分格一致性散点：方法的"一直错−一直对"增量差 vs 温度对照的同差，23 个可判定格 | `outputs/tables/stage7_cellconsistency.csv` | 第二层机制：多数格落在对角线上方之外——错误侧增量更大是尖化解释不了的残余结构 |
| 3 | `outputs/figures/stage6_fig2_fourcell.pdf` | 四格分解（一直对/一直错/由错变对/由对变错）的平均增量 + CI | `outputs/tables/stage6_fourcell.csv` | 残余结构的形态：被纠正格与一直错格的增量偏大 |
| 4 | `outputs/figures/stage6_fig3_did.pdf` | 分层差中差森林图（16 格全正、CI 不含零） | `outputs/tables/stage6_did.csv` | 校准后果的人群符号结构 |
| 5 | `outputs/figures/stage6_fig4_dose.pdf` | 剂量-响应：增量与 ΔECE 随 α 单调；变号点随 α 上移 | `outputs/tables/stage6_dose.csv` | 支撑一（幅度与强度的经验关系） |

不进入正文、留作附录/补充：`stage6_variant` 对照表、`stage7_seedrobust`、
`stage7_dolasubset`、`stage7_dogspredict`、`stage7_sid`、`stage7_m3idgate_*`、
`stage7_poscontrol`、`stage7_drift`、`stage7_distlevel`（审稿追问时指向）。

无新增测量、无新增图型；本清单即封版清单。
