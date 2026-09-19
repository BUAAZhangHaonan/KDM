# InternVL3.5-8B SID 双卡只读方案

这是未执行、未修改spec的方案。当前单卡v2在原图像尺寸/切片/输入下发生真实eager softmax OOM：还需1.37GiB、仅1.02GiB可用；保留 `outputs/verification/internvl35_8b_sid_reference_v2.json/log`。没有重试、改尺度、换精度或减少核验分支。

原 checkpoint config 的 language model 是 Qwen3ForCausalLM：36层，hidden4096，32 attention heads，8 KV heads，head_dim128；vision为24层、hidden1024、448/14。`modeling_internvl_chat.py`实际创建 `vision_model`、`language_model`、`mlp1`，Qwen3实际decoder路径为 `language_model.model.layers`。这些是源码/config读取结果，不是本次重新加载的运行时named_modules快照。

可审阅的无重叠显式映射在 `verification/internvl_sid_dual_gpu_proposal.json`：physical4->logical0，physical5->logical1；vision_model、mlp1、text embedding、rotary、decoder0–17放logical0，decoder18–35及final norm/lm_head放logical1。每层恰一次；不使用auto、不落CPU/disk；max_memory每卡22GiB是限额而不是实测峰值。保持bf16、现有processor的max_num12/thumbnail、所有实际输入、第二层选择与最低100不变。必须同时用worker的4,5两把锁，且无其他占用。

当前adapter无法仅改spec立即执行：`InternVLModel.__init__`仍是from_pretrained后`.to(device)`，get_engine的internvl分支也未传device_map/max_memory。这意味着未来若授权双卡，需明确修改该构造路径为显式dispatch并移除整体`.to`，不改视觉预处理/embedding替换；随后重新验证native16和SID，记录hf_device_map与各卡实际峰值。本文不声称上述映射已经运行或保证不OOM，也没有将其写入spec。

内存保留核对：native `_prefill`没有请求全模型output_attentions；SID只捕获第二层attention到当前forward作用域，trace_collector不会在event记录里保留GPU attention tensor，返回的summary也只有标量/索引。Qwen3 eager核心在matmul后用float32 softmax，当前OOM栈落在该必需计算。控制器目前确实保留第二层整个attention tensor到forward结束，即使选择只读last query；这是一项可见的实现内存占用，不代表本次已获准或已实施新的内存优化。未改变核心算子，也未推定某个优化必能解决资源失败。
