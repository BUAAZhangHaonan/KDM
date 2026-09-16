# 第六阶段实现对照表（IMPLEMENTATION_MAPPING_STAGE6）

逐方法对照官方定义、本阶段实现、与先前实现的差异、依据位置、第三方一致性。
本文档同时回答仓库代码审阅中悬置的三个问题（见文末）。
官方仓库克隆于 `reference_repos/`（不入库，gitignore），版本以提交哈希为准：

| 仓库 | 提交 | 用途 |
|---|---|---|
| DAMO-NLP-SG/VCD | d6568ff | VCD 官方实现 |
| huofushuo/SID | 127dd41 | SID 官方实现（含其内嵌 VCD） |
| voidism/DoLa | 805230e | DoLa 官方实现 |
| zjunlp/DeCo | c1a9129 | DeCo 官方实现 |
| BillChan226/HALC | 主干 | 第三方一致性对照 |

---

## VCD（Visual Contrastive Decoding，arXiv:2311.16922，CVPR 2024）

| 官方关键部件 | 本阶段实现 | 与先前实现（v2–v5）的差异 | 依据 | 第三方一致性 |
|---|---|---|---|---|
| 加噪：`betas = sigmoid(linspace(-6,6,1000))·(0.5e-2−1e-5)+1e-5`，`x = √ᾱ·x+√(1−ᾱ)·ε`，noise_step=500，施加在处理器输出的像素张量上，不截断 | `stage6_engine.add_diffusion_noise_official`，逐点同式，作用于 `pixel_values` | 先前用线性 β 调度（1e-4→0.02）且截断回 [0,1]，加噪发生在 PIL 图上（预处理前）——三处均不一致，本阶段全部改齐 | `vcd_utils/vcd_add_noise.py`（L3-28）；施加位置 `experiments/eval/object_hallucination_vqa_llava.py` L58-60 | HALC 内嵌的即官方 VCD 仓库，同一代码；SID 仓库的 `vcd_sample.py` 同公式 |
| 对比：`diffs = (1+α)·z_原图 − α·z_加噪`，cd_alpha=1 | 同式 | 无差异（此前已一致） | `vcd_utils/vcd_sample.py` L150-153 | 同上 |
| 截断：`cutoff = log(β)+max(z_clean)`，`z_clean<cutoff` 置 −inf，β=0.1 | 同式（logit 空间，作用在对比后的 logits 上，掩码条件取自原图分支） | 无实质差异（此前用概率比 ≥β·pmax，与 log 空间版本数学等价——官方仓库注释中的 version 1 与 version 2 等价） | 同上 L150-153 | 同上 |
| 采样：`do_sample=True`，T=1，top_p=1，top_k=None，seed=42 | 取同一被截断分布的**贪心 argmax** | 偏差（有意）：全项目配对测量要求确定性；官方默认为多项采样 | 同上 L160-166 及脚本 L75-121 | SID 仓库提供 `--sample-greedy` 贪心变体，公式一致 |
| 噪声随机性：全局 set_seed(42) 顺序抽样 | 逐样本确定性种子 hash(文件名,方法) | 偏差（有意）：分布相同、逐样本可复现 | — | — |

## SID（Self-Introspective Decoding，arXiv:2408.02032）——受阻，未实现

| 官方关键部件 | 状况 | 依据 |
|---|---|---|
| 依赖：内嵌修改版 transformers，`pip install -e transformers` | **4.29.2**（2023-05）与项目 5.17.0（Qwen3.5 模型类型所需）不兼容；无法安装并存 | `sid/transformers/setup.py` L433、`src/transformers/__init__.py` L21 |
| 支持模型 | 仅 llava-1.5 / instructblip / shikra；不含本项目四模型 | `sid/pope_eval.py` L223/292-296 |
| 机制：CT2S 在第 2 层用第 1 层注意力保留注意力最低的 100 个视觉 token，对比式同 VCD（α=1，β=0.1，可贪心） | Qwen3.5 层 0–2 均为线性注意力（config.layer_types），无注意力矩阵可排序——机制在该架构上无定义 | `sid/transformers/.../modeling_llama.py` L721-800（`USE_FAST_V`、`topk(largest=False)`）；Qwen3.5-4B config.json layer_types |
| 处置 | 按任务书 §3：不兼容即停止上报，不静默改写。实验 A 以 4 方法 + direct 运行；机制通用性由 DeCo/DoLa（层轴）与 M3ID（同视觉轴）承担，报告中如实说明 | — |

## M3ID（Multi-Modal Mutual-Information Decoding，arXiv:2403.14003，CVPR 2024；无官方代码）

| 论文关键部件 | 本阶段实现 | 与先前实现（本地历史名 MIB）的差异 | 依据（论文位置） |
|---|---|---|---|
| 双分支：条件 `l_c=log p(y|提示,前缀,图像)` vs 无图先验 `l_u=log p(y|提示,前缀)`（同一模型去掉视觉输入） | 条件分支=常规输入；先验分支=同一提示的文本-only 输入（Qwen 文本模板 / LLaVA 去图像 token / InternVL 去 `<img>` 块） | MIB 用"高斯模糊图分支"——模糊图 ≠ 无图先验，性质不同（模糊图仍含视觉内容） | 论文 §3.2、式 (3)；"as long as it is possible to drop the visual conditioning" |
| 合成式 `l* = l_c + 𝟙[max_k(l_c)_k < log α]·(1−γ_t)/γ_t·(l_c−l_u)`，贪心 argmax | 逐 token 同式；λ=0.02，α=0.3（论文默认） | MIB 用线性衰减权重 `a_t = a_max(1−t/T)`、α=1、无置信度门控——三处均与论文不同 | 论文式 (4) 与 Algorithm 1；默认值见论文表 3 与实现细节（λ∈{0.001..0.03} 选 0.02，α∈{0..1} 选 0.3） |
| 调度 γ_t=exp(−λ(t+t₀))；t₀：描述任务取 0，POPE 取 ≈问题长度 | t₀ = 命名问题文本的 token 数（本项目任务为短答案问答，采 POPE 规则） | MIB 无此调度（其线性衰减是自造） | 论文实现细节："In all our captioning experiments, we set t₀=0; for POPE, t₀=10 (≈ question length)" |
| 无截断、无采样 | 同 | MIB 亦无截断（一致） | 论文 Algorithm 1 |

M3ID 无官方代码（facebookresearch/M3ID 404，已确认未开源）；本实现按论文逐项重建，上表每行给论文依据。SID 仓库不含 M3ID，无第三方对照可用。

## DoLa（Decoding by Contrasting Layers，arXiv:2309.03883，ICLR 2024）

| 官方关键部件 | 本阶段实现 | 与先前实现（本地历史名 LCD）的差异 | 依据 | 第三方一致性 |
|---|---|---|---|---|
| 成熟层=最终层；候选早熟层=hidden_states 下标集合（脚本传入，README 示例 0,2,4,…,32 或上半区连续层） | 候选=下标 0..L−1 全体（动态选层的完整候选集；脚本子集为算力取舍） | LCD 固定用 `int(0.5·L)` 一层或相邻层——不同 | `dola_greedy_decode` 签名与 `tfqa_mc_eval.py` L270-289（`--early-exit-layers` 多值=候选集，末值=成熟层） | HALC 无 DoLa 实现（仅残留旗标） |
| 每步动态选层：取与**最终层**分布 JSD 最大的候选层 | 同（JSD(final, candidate) 逐候选计算） | LCD 取**相邻早熟层之间** JSD 最大处——选层对象错误，本阶段改齐 | `transformers-4.28.1/src/transformers/generation/utils.py` dola_greedy_decode（stacked_premature_layers ↔ softmax_mature_layer 的 js_divs） | — |
| 早熟层 logits = `lm_head(hidden_states[i])`，**不施加 final norm** | 同 | LCD 对早熟 hidden 施加 final norm——与官方代码不同（注意：论文正文叙述含 norm，官方代码未施加；以仓库为准，记为论文-代码差异） | 同文件 modeling_llama.py L699-704 | — |
| 对比：`z* = relative_top_filter(z_final, 0.1) − log_softmax(z_prem)`，掩码位 base 截 −1e3 | 同式同参数 | LCD 无 relative_top 掩码、用 `2·logp_final − logp_early`（等价性：在未掩码位置与官方式相差一个 token 无关常数，掩码为新增） | 同文件 L2456-2462（relative_top_filter）、dola_greedy_decode 主体 | — |
| 选层每步进行 | 同（每步重算） | 一致 | 同上（premature_layer_dist 逐步累计） | — |
| 贪心模式为官方支持（dola_greedy_decode） | 贪心 | 一致 | — | — |
| 纯文本方法用于多模态模型=迁移使用 | 报告中明确声明 | — | — | — |

## DeCo（Dynamic Correction Decoding，arXiv:2410.11779，ICLR 2025）

| 官方关键部件 | 本阶段实现 | 与先前实现（无；DeCo 为本阶段新增）的差异 | 依据 | 第三方一致性 |
|---|---|---|---|---|
| 候选层：llama-7b 默认 `range(20,29)`（hidden 下标） | 按层占比迁移：`[ceil(0.625L), floor(0.875L)]`（L=32 时恰为 20..28=官方；q4b/q9b/llava16 L=32→20..28；internvl4b L=36→23..31） | 新增方法 | `pope_llava.py` L123,164-165 | — |
| 每步：最终层 top-k（20）候选 token，按累计概率截到 top-p（0.9） | 同（top-k 与 top-p 交集；预登记表述只写了 top-k，此处按官方代码精确化为交集，见 STAGE6 偏差说明） | 新增 | `transformers/generation/utils.py` deco_greedy_search L2660-2668 | — |
| 选层：候选层在候选 token 上的概率最大者（网格 argmax），取其概率为 premature_max_probs | 同 | 新增 | 同上 L2669-2676 | — |
| 校正：`z* = z_final + alpha·maxprob·z_selected`，`alpha=0.6`，随后仅保留候选 token | 同 | 新增 | 同上 L2677-2682 | — |
| 早熟层 logits = `lm_head(model.norm(h_l))`（**施加** final norm，与 DoLa 不同） | 同 | 新增 | `transformers/models/llama/modeling_llama.py` L819-824 | — |
| 采样：官方 eval 默认 `do_sample=True`(T>0)；greedy 分支为官方支持 | 贪心（deco_greedy_search） | 记为偏差（同 VCD，配对测量确定性） | pope_llava.py L114；generation/utils.py L1541 | — |

---

## 悬置问题三则

**1. 置信度记录时机（掩码重归一化之前还是之后）**：之后。`answer_maxp`/`step_probs` 取自各配置**实际用于选 token 的最终分布**——对比合成、截断掩码、softmax 重归一化全部完成之后（掩码位置概率为 0，其余位置重归一化）。这一口径自第二阶段沿用至今，未把未归一化 logits 当概率。直接解码与 M3ID 无掩码，取其（对比后）softmax；VCD/DoLa/DeCo 取掩码后重归一化分布。

**2. 分层是否泄漏**：不泄漏。类别分组由**分组半区**上的实测命名准确率排序（种子 42 的对半切分，先于一切干预实验），评测只用**评测半区**样本；两半区样本不相交（同一类的 24+24 张互斥划分）。本阶段的开发子集（B 温度匹配、F 偏移估计）同样只取自分组半区，与评测半区不相交；评测半区的数据从未参与任何超参或对照强度选择。

**3. 是否忠实原文**：逐方法见上表。共同记录的两类有意偏差（不改变方法定义，只改变解码协议/随机性管理）：(a) 全方法贪心 argmax（官方 VCD/DeCo 默认采样、DoLa/M3ID 官方即支持贪心）；(b) VCD 噪声逐样本确定性种子（官方全局种子）。其余部件均按官方代码/论文逐条对齐。
