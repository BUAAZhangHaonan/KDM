# KDM 论文 v3（2026-10-04 更新）

**Lost in Contrast: Preserving Abstention in Multimodal Contrastive Decoding**

- `paper_v3_zh.pdf`：唯一当前PDF；正文8页，参考文献从第9页开始，随后为完整附录。
- `paper_v3_zh.tex`：LaTeX主文件；`paper_v3_zh.md`为同步的正文Markdown。
- `appendix_v3_zh.*`、`appendix_supplement_v3.*`：九模型完整结果、推导和补充分析。
- `figures/`、`figure_sources/`：五张当前主图及一张完整VizWiz附录图、重绘脚本、ImageGen提示词及来源。
- `statistics/`：70个Food主工作点、68个VizWiz工作点、控制、开发选择和机制表。
- `change_summary.md`：逐项修改说明。

Gemma原生SID补全2,424题；MiniCPM与Qwen3.5按原算法适用性标注。Food以联合效用J为主，IP四表达取同一最高J配置；开发集所选配置另表。原生VCD为主基线，带引导VCD/M3ID用于机制分析。

运行`sh build_pdf.sh`编译。依赖XeLaTeX、BibTeX、Noto Serif CJK SC/Noto Sans CJK SC、Latin Modern Roman及常用TeX Live宏包。图表和引用均为包内相对路径。

配套`KDM_Nine_Current_Data_20261004.zip`含逐样本结果、参考、配置与来源索引。Food共484条件、1,173,216条；VizWiz512题主比较已闭合，全量独立试答的扩展语义缺口单列。

本次补入 VizWiz 原生 DoLa、DeCo 各4,608条及原SID 2,560条。23条件均512题，生成、语义评分、官方参考连接已闭合。所有当前表和图直接读取这次接受的汇总。
