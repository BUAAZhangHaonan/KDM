# 主要来源与核查位置

查询日期：2026-09-16。下列为实际读取的原论文、官方/作者仓库与会议页面。表格中的理论和新实验判据是本包设计，不冒充文献结论。

**[S1] KDM 固定证据与实现。** 提交`e5eda213dd3f51c08b43c1616305c93b84011dff`。主要读取`outputs/tables/stage6_core.csv`、`stage6_fourcell.csv`、`stage6_dname.csv`、`stage6_correction.csv`；`code/stage6_run.py`、`stage6_engine.py`、`stage6_analysis.py`；`docs/stage8/CLAIMS_STAGE8.md`。入口：https://github.com/BUAAZhangHaonan/KDM/tree/e5eda213dd3f51c08b43c1616305c93b84011dff 。后者仍有失效表述，数字优先于旧总览。源blob指纹见数据样例的PROVENANCE。

**[S2] Mirage。** Hao Yin等，The Mirage of Performance Gains: Why Contrastive Decoding Fails to Mitigate Object Hallucinations in MLLMs? 论文4—5节与附录：https://arxiv.org/html/2504.10020v4 。作者仓库README及脚本索引：https://github.com/ustc-hyin/cd_rethink 。NeurIPS2025正式条目：https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html 。本包不把作者对整个方法类别的评价直接采纳为结论。

**[S3] VCD_Analysis。** Yi-Lun Lee、Yi-Hsuan Tsai、Wei-Chen Chiu，Delve into Visual Contrastive Decoding for Hallucination Mitigation of Large Vision-Language Models。参考图像操作见3.2节，纠正/致错及回答倾向见4.4节，融合方法见5节：https://arxiv.org/html/2412.06775v1 。代码入口：https://github.com/YiLunLee/VCD_Analysis 。正文比较依据论文内容，不根据仓库标题推断证据范围。

**[S4] 测试时适应校准。** Respect Your Zero-Shot Uncertainty: Conservative Calibration for Test-Time-Adapted Vision-Language Models。2026年8月预印本：https://arxiv.org/html/2608.05945v1 。近邻内容包括未变预测、概率变化与保守校准。本次未确认正式录用状态，材料统一称预印本。

**[S5] 任务关系文献。** ActLCD：https://arxiv.org/abs/2505.23657 ；VISOR：https://arxiv.org/abs/2608.11024 ；CHASD：https://arxiv.org/abs/2605.23344 ；PhantomBench：https://arxiv.org/abs/2606.11105 。这些只用于任务及观察位置定位，不声称KDM直接验证或否定它们。

**[S6] 选择性回答与POPE。** Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly，ECCV2022：https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2576_ECCV_2022_paper.php 。官方POPE数据：https://github.com/RUCAIBox/POPE/tree/main/output/coco 。本包在下载脚本中校验random/adversarial标注的Git blob SHA，不从第三方表格重建问题。

**[S7] TMM官方范围。** https://signalprocessingsociety.org/publications-resources/ieee-transactions-multimedia 。用于确认多媒体研究范围，不据此保证审稿结果、分区或周期。

**[S8] NAACL2027官方征稿。** https://2027.naacl.org/calls/main_conference_papers/ 。ARR截稿2026-10-12 AoE。这是本次查到的可落在一个月窗口的会议出口，不代替TMM首选。

**[S9] 原方法实现来源。** VCD：https://github.com/DAMO-NLP-SG/VCD （项目记录d6568ff）；DoLa：https://github.com/voidism/DoLa （805230e）；DeCo：https://github.com/zjunlp/DeCo （c1a9129）；SID：https://github.com/huofushuo/SID （127dd41）；M3ID按论文算法重建：https://arxiv.org/html/2403.14003 。本轮复用KDM已核对的实现，不把适配到VLM的DoLa说成原始多模态方法。

**写作依据。** 用户附带《我对科研和论文写作的思考与要求》和`Research_Writing_Clear_Prose_Guide.md`：研究先回答问题与新认识、准确表述证据、正文不回放过程、代码范围与失败必须明确。本包执行说明与论文骨架分开，不复制整份写作指南增加重复。
