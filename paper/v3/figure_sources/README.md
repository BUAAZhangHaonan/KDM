# v3 图件与复算入口

正文保留五张图。图1与图3采用本轮内置ImageGen设计；图1的实验照片以原始JPEG嵌入PDF，边际标尺按精确数据绘制。图2、4、5从真实CSV导出。五图均提供PDF/PNG，数据图同时提供矢量SVG。

| 图 | 文件前缀 | 内容及来源 |
|---|---|---|
| 1 | fig01_authentic_case_teaser | 原始生牛肉片与苹果派照片、真实回复、边际−0.1875→+0.421875 |
| 2 | fig02_mechanism_four_panel | 287例流向、4×4弃权率、四路log-odds、参考交互解析插值 |
| 3 | fig03_three_condition_duty_schematic | g提供基础与Sg，c−r提供视觉差分；苹果为概念示意 |
| 4 | fig04_food_nine_model_J_heatmap | Food九模型、70个工作点，包含Gemma原生SID |
| 5 | fig05_vizwiz_paired_J_forest | 512题配对ΔJ、18个真实95% bootstrap区间 |

数据图嵌入Arial字形，SVG转路径避免跨机器字体替换。统计值、模型顺序、零轴与区间端点均保留，跨栏图按174毫米、单栏图按83毫米宽设计。

## 重绘

- `python figure_sources/build_figures.py`：numpy、pandas、matplotlib和EasyPlot；环境变量`EASYPLOT_ROOT`可指定技能目录。
- `python figure_sources/compose_imagegen_figures.py`：reportlab、Pillow、Windows Arial；组合随包设计、原照片与精确标尺。生成提示词见`imagegen_*_prompt.txt`，无需再次调用图像服务。PNG可用Poppler从PDF渲染。
- `python figure_sources/compute_vizwiz_ci.py --source-dir SOURCE_DIR`：源目录含`merged31_new_scores.csv`和`finalrows/*.jsonl.gz`；原路径与SHA见`vizwiz_ci_receipt.json`。2,000次按输入配对重采样、seed20260929、95%百分位区间。
- `python figure_sources/merge_gemma_sid.py --source GEMMA_EXPORT_DIR`：核验2,424唯一键及评分闭合后加入一个工作点，保留69条已有值。

设计参考阅读：[VCD](https://arxiv.org/pdf/2311.16922)、[Seeing the Image](https://proceedings.neurips.cc/paper_files/paper/2024/file/37294f033582ac0064bf90fa557c2573-Paper-Conference.pdf)。本版重新制作布局，数值均来自项目记录。

`figure_manifest.json`与`imagegen_provenance.json`保存数据、生成素材和原照片SHA。图3的苹果为示意素材；图1保留真实评估照片。

附录图 `figA01_vizwiz_all_baselines_J` 汇总全部68个VizWiz工作点。原生DoLa/DeCo各九模型，SID五模型；缺项按原算法适用范围标注。
