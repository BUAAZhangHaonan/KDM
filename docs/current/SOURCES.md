# 论文、数据与代码来源

核查日期：2026-09-18。技术论断使用作者论文、官方会议页面、官方仓库与官方数据页。下载文件实际SHA256由 `scripts/fetch_assets.py` 写入 `outputs/records/assets_*.json`。

## 与研究主线直接相关的论文

[S1] Hyuhng Joon Kim, Youna Kim, Sang-goo Lee, Taeuk Kim. When to Speak, When to Abstain: Contrastive Decoding with Abstention. ACL 2025, pp.9710–9730. DOI:10.18653/v1/2025.acl-long.479。
正式页面：https://aclanthology.org/2025.acl-long.479/
全文：https://aclanthology.org/2025.acl-long.479.pdf
用途：知识可用性操作化、三路解码、空输入熵校准与回答/弃权联合评价。包中的 `cda_visual` 明确标注为视觉任务迁移，作者原始实验使用文本输入。未提供已经核实的官方代码地址。

[S2] Hao Yin, Guangzong Si, Zilei Wang. The Mirage of Performance Gains: Why Contrastive Decoding Fails to Mitigate Object Hallucinations in MLLMs? NeurIPS 2025；本次读取arXiv v4（2026-06-04）。
全文：https://arxiv.org/pdf/2504.10020
作者仓库：https://github.com/ustc-hyin/cd_rethink
用途：分布偏移与候选约束的独立解释，以及分析论文的结构。

[S3] Alessandro Favero等. Multi-Modal Hallucination Control by Visual Information Grounding. CVPR 2024, pp.14303–14312。
全文：https://arxiv.org/pdf/2403.14003
正式页面：https://openaccess.thecvf.com/content/CVPR2024/html/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.html
用途：M3ID准确公式、门控、问题长度偏移及不加区别减去先验可能影响正确词元的讨论。

[S4] Spencer Whitehead等. Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly. ECCV 2022。
全文：https://arxiv.org/pdf/2204.13631
作者仓库：https://github.com/facebookresearch/reliable_vqa
用途：模型能力、输入可回答性与选择性预测的关系。

[S5] Alisa Liu等. DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts. ACL-IJCNLP 2021, pp.6691–6706. DOI:10.18653/v1/2021.acl-long.522。
正式页面：https://aclanthology.org/2021.acl-long.522/
全文：https://aclanthology.org/2021.acl-long.522.pdf
用途：基础分数加差分分数的已知方法家族；不将该代数形式宣称为本文首次提出。

[S6] Sicong Leng等. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding. CVPR 2024。
全文：https://arxiv.org/pdf/2311.16922
官方仓库：https://github.com/DAMO-NLP-SG/VCD
既有KDM参考提交：d6568ff。核查加噪作用位置、sigmoid调度、alpha与beta、候选限制及采样方式。

[S7] Fushuo Huo等. Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models. 论文全文：https://arxiv.org/pdf/2408.02032
官方仓库：https://github.com/huofushuo/SID
既有参考提交：127dd41。用途：通过视觉词元处理构造参考，检验参考构造边界。本文执行版本修正既有图像区间长度解释，保留因果注意力掩码；真实模型需与官方片段重新核验。

[S8] VCD_Analysis对应全文：https://arxiv.org/html/2412.06775v1
用途：视觉扰动、置信度、熵、纠正与致错的既有分析。正文不将其缩写成只研究准确率。

[S9] Xintong Wang等. Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding. Findings of ACL 2024, pp.15840–15853. DOI:10.18653/v1/2024.findings-acl.937。
全文：https://aclanthology.org/2024.findings-acl.937.pdf
仓库：https://github.com/PostMindLab/ICD
用途：说明通过指令构造参考也属于相关家族。ICD具有模型相关的指令注入位置，本包不将任意负向提示套用到全部模型并冠以官方ICD名称。

[S10] Tim van Erven, Peter Harremoës. Rényi Divergence and Kullback–Leibler Divergence. IEEE Transactions on Information Theory, 2014. DOI:10.1109/TIT.2014.2320500。
全文：https://arxiv.org/pdf/1206.2459
用途：Rényi散度定义；本文的语义分组分解由直接求和推导。

[S11] MM-SAP: A Comprehensive Benchmark for Assessing Self-Awareness of Multimodal Large Language Models in Perception. ACL 2024。
正式页面：https://aclanthology.org/2024.acl-long.498/
仓库：https://github.com/YHWmz/MM-SAP
用途：感知自省的操作性评价依据。它的选择题协议与本研究的引导开放命名分别说明。

## 数据

[D1] Food-101官方页面：https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
正式执行复用项目 `data/samples_manifest.jsonl` 的全部图像和既有分割。此清单的全量与官方101000张全集区分清楚。

[D2] VizWiz-VQA官方页面：https://vizwiz.org/tasks-and-datasets/vqa/
标注：https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip
验证图像：https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip
完整官方验证集4319题，每题10份答案。使用当前官方标注记录，不将人工不可回答标签直接当作模型参数知识缺失的标签。

[D3] VQA官方归一化与评分源码：https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
源码blob：e4ff7887d53195f12856ab1e9087e69abe2e75c8。下载器提取原始归一化函数，评分实现保留逐标注者排除自身的共识计算。

## 模型与项目

[C1] KDM：https://github.com/BUAAZhangHaonan/KDM/tree/eceed2246515cb938ad478ee23c94590c990eb5d
[C2] mprisk模型登记：https://github.com/BUAAZhangHaonan/mprisk/blob/master/configs/assets/model_assets.yaml
[C3] vLLM支持模型：https://docs.vllm.ai/en/latest/models/supported_models/
[C4] DoLa：https://github.com/voidism/DoLa ，既有提交805230e。
[C5] DeCo：https://github.com/zjunlp/DeCo ，既有提交c1a9129。

候选模型的正式HF页面已经逐项保存在 `configs/models.json`。模型文件、分词器和远程代码按服务器实际文件指纹记录；已有模型不重复下载。

## 投稿规则

NAACL 2027正式征稿：https://2027.naacl.org/calls/main_conference_papers/
ARR提交日期2026-10-12 AoE，NAACL承诺提交日期2026-12-23，长文正文8页。
TMM作者页面：https://signalprocessingsociety.org/publications-resources/information-authors
同一稿件不得在两个渠道同时处于评审。
