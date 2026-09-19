# InternVL3.5-8B SID 双卡只读方案

这是未执行、未修改spec的方案。当前单卡v2在原图像尺寸/切片/输入下发生真实eager softmax OOM：还需1.37GiB、仅1.02GiB可用；保留 `outputs/verification/internvl35_8b_sid_reference_v2.json/log`。没有重试、改尺度、换精度或减少核验分支。

原 checkpoint config 的 language model 是 Qwen3ForCausalLM：36层，hidden4096，32 attention heads，8 KV heads，head_dim128；vision为24层、hidden1024、448/14。`modeling_internvl_chat.py`实际创建 `vision_model`、`language_model`、`mlp1`，Qwen3实际decoder路径为 `language_model.model.layers`。这些是源码/config读取结果，不是本次重新加载的运行时named_modules快照。

可审阅的无重叠显式映射在 `verification/internvl_sid_dual_gpu_proposal.json`：physical4->logical0，physical5->logical1；vision_model、mlp1、text embedding、rotary、decoder0–17放logical0，decoder18–35及final norm/lm_head放logical1。每层恰一次；不使用auto、不落CPU/disk；max_memory每卡22GiB是限额而不是实测峰值。保持bf16、现有processor的max_num12/thumbnail、所有实际输入、第二层选择与最低100不变。必须同时用worker的4,5两把锁，且无其他占用。

当前adapter无法仅改spec立即执行：`InternVLModel.__init__`仍是from_pretrained后`.to(device)`，get_engine的internvl分支也未传device_map/max_memory。这意味着未来若授权双卡，需明确修改该构造路径为显式dispatch并移除整体`.to`，不改视觉预处理/embedding替换；随后重新验证native16和SID，记录hf_device_map与各卡实际峰值。本文不声称上述映射已经运行或保证不OOM，也没有将其写入spec。

内存保留核对：native `_prefill`没有请求全模型output_attentions；SID只捕获第二层attention到当前forward作用域，trace_collector不会在event记录里保留GPU attention tensor，返回的summary也只有标量/索引。Qwen3 eager核心在matmul后用float32 softmax，当前OOM栈落在该必需计算。控制器目前确实保留第二层整个attention tensor到forward结束，即使选择只读last query；这是一项可见的实现内存占用，不代表本次已获准或已实施新的内存优化。未改变核心算子，也未推定某个优化必能解决资源失败。

用户随后明确回复“允许按上述双卡方案核验”。后续实现仅在项目内按该方案执行，并重新建立原生16图/SID证据；本提案中的单卡失败和提出时未执行的历史身份保留。

## 后续明确授权执行

用户随后明确授权该方案。已新增独立factory `kdm.models.internvl_dual:InternVLDualBackend`，其engine继承既有InternVLModel，所有build/模板/预处理/前向方法直接复用，只更换构造时的显式18/18 dispatch；旧hf/backbone/remote/sid均未改。原单卡spec在outputs/records/spec_history/internvl35_8b_single_gpu_before_dual.json完整保留。新factory7项CPU测试通过，物理4,5双锁下正在重新完整native16，之后才进行SID；当前不将待运行步骤当passed。上文JSON仍作为当时未执行提案保留。

双卡native16现已完整16/16通过，六条件状态一致及36层projection均零差；实际峰值logical0=14,374,173,184 bytes、logical1=10,517,976,576 bytes。记录 `outputs/verification/internvl35_8b_dual_final16.json` 含实际完整hf_device_map与新的source依赖；SID v3正在同一双卡构造下核验，仍待结果。

双卡SID v3现也已通过：14 visits，官方选择/掩码核心logits、fresh参考、交错session、非单调prefix及native/clean前后logits比较均成功且数值差为0。规范摘要为 `outputs/verification/internvl35_8b_sid_reference_v3_summary.json`。当前InternVL spec已登记新的native16与SID摘要；实际`validate_method_runtime(...,['sid'])`通过。物理4/5任务已退出并释放。
