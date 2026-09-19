# SID 固定候选能力核验

范围为官方 commit `127dd412fa6b61ab1c9babf6979ec4da98002438` 的 agg_layer=2/rank100/mask-mode 参考构造。native16通过不等于SID通过。passed只表示固定官方选择与掩码核心在同一native backbone前向的数值比对通过；oracle共享hook transport，不是完整旧官方fork复现，不证明研究收益。

单卡任务使用每个spec自己的environment_python、物理GPU4或5单卡、双卡使用物理4,5、`bash scripts/worker.sh`锁与项目内缓存。不改输入、生成配置或权重映射，不在模型失败后重试/降规模。LLaVA1.5-7B沿用主线程已跑证据、不重复。Gemma12B和LLaVA13B原为两卡调度待办，未执行不能算完成。无权重preflight有实际config和源码行证据，区分固定参考所需结构不成立与当前适配映射缺口。

|候选|状态|实际数值/具体原因|证据|
|---|---|---|---|
|gemma3_12b|runtime_mask_contract_error|ValueError: SID requires an additive floating 4D full-KV causal mask|`outputs/verification/gemma3_12b_sid_reference.json` / 同名 `.log`|
|gemma3_4b|runtime_mask_contract_error|ValueError: SID requires an additive floating 4D full-KV causal mask|`outputs/verification/gemma3_4b_sid_reference.json` / 同名 `.log`|
|glm46v|pending|尚未执行|—|
|internvl35_8b|oom_v2_eager|OutOfMemoryError: CUDA out of memory. Tried to allocate 1.37 GiB. GPU 0 has a total capacity of 23.56 GiB of which 1.02 GiB is free. Including non-PyTorch memory, this process has 22.53 GiB memory in use. Of the allocated memory 20.56 GiB is allocated by PyTorch, and 1.67 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)|`outputs/verification/internvl35_8b_sid_reference_v2.json` / 同名 `.log`|
|llava15_13b|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/llava15_13b_sid_reference_v2.json` / 同名 `.log`|
|llava15_7b|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/llava15_7b_sid_reference_v2.json` / 同名 `.log`|
|llava16_mistral|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/llava16_mistral_sid_reference_v2.json` / 同名 `.log`|
|llava16_vicuna|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/llava16_vicuna_sid_reference_v2.json` / 同名 `.log`|
|minicpm26|fixed_rank_undefined_on_observed_inputs|Native processor produces 64 visual embedding positions for some fixed16 inputs, fewer than required rank100. Existing llm path/span adapter gap is separate; fixing it cannot make topk(100) defined on 64 tokens without changing fixed protocol.|`outputs/verification/minicpm26_sid_reference.json` / 同名 `.log`|
|minicpm45|fixed_rank_undefined_on_observed_inputs|Native processor produces 64 visual embedding positions for some fixed16 inputs, fewer than required rank100. Existing llm path/span adapter gap is separate; fixing it cannot make topk(100) defined on 64 tokens without changing fixed protocol.|`outputs/verification/minicpm45_sid_reference.json` / 同名 `.log`|
|onevision|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/onevision_sid_reference_v2.json` / 同名 `.log`|
|phi35|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/phi35_sid_reference_v2.json` / 同名 `.log`|
|qwen25vl|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/qwen25vl_sid_reference_v2.json` / 同名 `.log`|
|qwen35_4b|fixed_reference_structure_incompatible|Fixed SID reads second decoder self_attn weights, but registered layer_types[1]=linear_attention constructs Qwen3_5GatedDeltaNet as linear_attn (no self_attn). Selecting another full-attention layer would change agg_layer=2 protocol.|`outputs/verification/qwen35_4b_sid_reference.json` / 同名 `.log`|
|qwen35_9b|fixed_reference_structure_incompatible|Fixed SID reads second decoder self_attn weights, but registered layer_types[1]=linear_attention constructs Qwen3_5GatedDeltaNet as linear_attn (no self_attn). Selecting another full-attention layer would change agg_layer=2 protocol.|`outputs/verification/qwen35_9b_sid_reference.json` / 同名 `.log`|
|qwen3vl|passed|14 visits; max logit error=0.0; fresh error=0.0|`outputs/verification/qwen3vl_sid_reference_v2.json` / 同名 `.log`|

映射调查详见 `docs/current/SID_VISUAL_MAPPING_REVIEW.md` 和 `outputs/verification/{minicpm26,minicpm45,phi35}_sid_visual_mapping.json`。Mini两版均有64<100的原生输入，属于固定rank在实际样本未定义；Phi757连续位置的补充方案已报告但尚未改源/前向验收。

分类解释：`fixed_reference_structure_incompatible` 表示实际结构不具备固定第二层attention定义；`adapter_structural_mapping_gap` 表示当前适配器未实现实际结构/视觉区间映射，不能推断该模型本质上不能定义SID；`oom` 是资源失败；`runtime_error` 是软件/运行错误；`numerical_mismatch` 是实际对照未通过。后三类不能写成架构不支持。

启动异常保留：LLaVA-v1.6-Vicuna第一次只在shell层因worker.sh无执行位失败（模型未加载），日志为 `outputs/verification/llava16_vicuna_sid_reference.launch_error.log`；随后用bash运行同一脚本，未改变权限或模型参数。
