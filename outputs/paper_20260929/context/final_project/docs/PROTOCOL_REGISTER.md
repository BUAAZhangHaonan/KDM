# 主结果协议登记

## 研究范围

当前主结果比较Food-101上的五个固定模型：Qwen2.5-VL、Qwen3.5-4B、LLaVA-v1.6-Mistral、MiniCPM-V2.6和Gemma3-4B。完整回答共853,248条，覆盖352个模型、方法与提示条件；每个条件使用相同的2,424个eval样本。原始回答及其来源身份由`data/responses/formal/manifest.json`绑定。

方法、提示和控制条件按正式固定面板逐项执行，包含direct基线、注册的视觉/解码干预、保留指令条件及相应控制。各模型只运行其登记支持的方法。正式比较按同一样本与主提示条件配对；参考提示条件作为干预变量单列。完整任务矩阵见`workflows/formal/panel.json`。 已登记的`copy_original_abstention`控制按样本选择完整回答：direct弃权时保留direct原回答，否则采用对应VCD或M3ID回答；控制选择只读取direct弃权标记。其CPU派生实现与200条件结果、40组同marker指令配对比较见`workflows/main_results/preserve_abstention_control.py`和`outputs/analysis/main_results/controls/`。

## 主评分

从回答中提取明确的主菜名称，再匹配Food-101的101个规范类别名及其下划线转空格形式。按完整类别词边界匹配，保留原文拼写和烹饪、风味等限定。完整句子可以表达规范菜名，例如“The dish is fried rice.”映射到`fried_rice`。配菜、原料、否定提及和解释文字不单独构成主菜；多个同级主菜或互相竞争的候选在单标签任务中计0。明确弃权按回答行为标签记录。

`canonical_name_in_primary`是主准确率字段；`literal_extracted_name`记录完整抽取名称的严格字面敏感性结果。单复数、同义词、错拼和连字符变体不扩入主类别集合。标签抽取先于读取目标类别并与Food-101图像标签比较。

## 参考真值与审阅

登记并保留两套参考真值：已接受的Qwen2.5-VL、MiniCPM-V2.6和LLaVA三模型操作性参考真值，以及使用统一主菜抽取规则建立的五模型参考真值。三模型真值和已接受冲突记录位于`data/reference_gt/`；五模型逐问题来源、抽取结果和审阅记录与主结果一同保存。两套真值分别报告，保持各自来源和适用范围。

回答行为标签由独立Luna自动标注。Root对主菜边界和疑难样本所作的裁定按精确问答绑定并保存；自动判断与root复核在记录中区分。

## 方法参数与来源

解码算法及数学实现沿用冻结研究协议；方法参数、模型运行环境、权重身份、样本清单和原始证明由`data/provenance/frozen_contract/`中的原字节副本及SHA清单绑定。当前活动运行配置位于`configs/kdm/`、`configs/runtime/`。复现入口为`workflows/formal/generate.py`，主评分、参考真值和条件分析入口位于`workflows/main_results/`。

## 分析顺序

先完成主评分、352条件方法比较、配对统计、图表和论文结论，再开展补充机制测量。机制工作复用同一固定样本与方法身份，并与正式文本生成结果分开报告。
