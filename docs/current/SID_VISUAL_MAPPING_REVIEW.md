# SID 原生视觉位置映射调查

只运行 native processor，不加载权重或改 SID 源。每个模型使用自己的 spec 环境，固定16张真实图与核验脚本第一条prompt；记录为 `outputs/verification/{minicpm26,minicpm45,phi35}_sid_visual_mapping.json` 及同名log。处理器构造、image crop参数沿用现有remote adapter，未为凑100修改尺寸或切片。

## MiniCPM2.6 / 4.5

实际 checkpoint `modeling_minicpmv.py` 的 `get_vllm_embedding` 先创建等长token embedding，再将视觉向量 scatter 到 `image_bound` 的每个半开 `[start,end)`。2.6第160–167行用stack后flatten，4.5第175–189行用cat后scatter；两者位置集合就是这些区间的并集，decoder在 `llm.model.layers`。

两个模型固定16图均观察到64和192两种视觉总数：64为1段，192为3段，每段64；3段间共5个非视觉token。JSON逐图保留全部真实位置、半开区间、gap token id和解码文本。绝不能用min/max包住这些间隙，把image/slice分隔符当成视觉embedding。先前SID resolver缺少llm路径是适配缺口；但64<100是独立的确定问题，即使正确补映射，固定topk100仍未定义。不能clamp100、改选层、扩大image切片或把文字塞进候选来“支持”。因此两个固定候选目前不能按同一全量rank100协议通过，不能把192图的可计算性推广到全部样本。

若将来获准实现纯位置集合接口，严格等价形式为：I是按原序排列的视觉位置并集；a=mean_heads(A_second[-1,I])；J=I[topk(a,100,largest=False)]；对所有后续层只屏蔽I\J，保留其余原始mask元素。I连续时与当前start/length逐元素相同；I离散时是对视觉集合的索引表示推广，不能直接称固定旧源码已验证。应在独立可用样本证明scatter位置与native视觉embedding写入一一对应，并用独立oracle的gather/选择/scatter映射核查；不足100仍必须拒绝。当前尚未实施这一推广。

## Phi3.5-Vision

实际 checkpoint `modeling_phi3_v.py` 第66行 `MAX_INPUT_ID=int(1e9)`，`Phi3ImageEmbedding.forward` 第227–242行以 `(input_ids<0)&(input_ids>-MAX_INPUT_ID)` 取得positions，然后`index_put(positions,image_features_proj,accumulate=False)`。这正是native embedding替换的位置，输入序列长度保持不变，不是单placeholder在decoder内部再展开。

固定16图的处理器输出全部是一段757个负token位置，无间隙。可严格补充：只对实际Phi3V配置使用上述原生谓词，要求batch1、位置连续、>=100，确认Q/K长度仍等于expanded input_ids+prefix；同一映射直接还原start/length，无需改变选择和mask核心，也不更换层。HD视觉特征里的newline/sub-global separator也是native image_features_proj写入该位置的向量，应按native视觉序列原样保留其身份，不能额外“净化”位置。

仅此处理器证据不证明Phi的原生attention/KV与SID接口兼容。需要审查 `self.model.layers[1].self_attn` 输出API、后续attention实现能否接收相同4Dmask，并在稳定修复后用它自己的phi443环境跑真实原片段数值核验。当前未改源、未跑Phi SID前向、未标passed。

## 运行失败的额外结构诊断

InternVL真实SID尝试在FA2 `_upad_input/_index_first_axis`触发OOM（额外21.12GiB），原始proof保留。当前端口只强制第二层eager，后层恢复FA2却收到4D mask，后层mask契约需另核查，不能用降规模掩盖问题。

Gemma3-4B真实前向触发浮点4D full-KV mask契约错误。安装TF5.5.3 `models/gemma3/modeling_gemma3.py`第735行起 `create_causal_mask_mapping`明确图像token为bidirectional，并按full/sliding层生成mask；原始SID oracle从全1重建纯causal。即使解决bool/dtype/shape，保留Gemma原生mask与严格官方pure-causal oracle不是同一个定义。不可静默改变Gemma image attention。此处不把一次接口异常本身写成架构不支持。

## 后续授权执行状态

以上“未改源/未跑Phi”描述的是映射调查阶段。主线程随后明确授权后，Phi专支已按原生负token谓词补入SID，并在原phi443环境真实核验通过：`outputs/verification/phi35_sid_reference_v2_summary.json`，完整raw保持在同名无summary文件。层2、rank100、757原生视觉位置、输入和processor保持不变；普通native/clean logits在SID前后均零差。MiniCPM的64<100仍不改参数，Qwen3.5也不改选层。InternVL后层FA2/4D软件契约修复后仍真实eager OOM，单卡不通过，双卡仅有只读方案。
