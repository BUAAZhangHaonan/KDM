# SID 官方来源、结构条件与修复核查

本记录只核查 SID 参考分支构造，不证明 KDM 的任务收益。审查发现的多会话覆盖、外部分支 attention 和 rank 静默截断已在 `src/kdm/models/sid.py` 修复；CPU 数值回归通过。真实登记多模态模型的 SID 支持状态仍以独立 GPU 数值记录为准，本文件不替它填入 passed。

## 来源与版本

固定源码为 `reference_repos/sid` commit `127dd412fa6b61ab1c9babf6979ec4da98002438`。主比较对象 `transformers/src/transformers/models/llama/modeling_llama.py` 的 mask-mode `elif USE_FAST_V` 分支（约 764–851 行），`pope_eval.py` 参数默认（约 89–102 行）、`vcd_sample.py` clean/reference forward 与独立 cache 更新（约 108–155、244 行）。只读使用已固定代码，未下载新版本。

已下载论文 `cache/assets/references/papers/SID.pdf` 为 arXiv:2408.02032v3，2025-03-16，ICLR 2025 版本，SHA256 `c7c06d30ac67544075f3876d2c1cd0638d22256a16ff701ca4635ceb048f4576`。实际阅读与此核查相关的方法图和公式（PDF 第 5 页，图 2、式 4/5）、第 6 页示例、第 7 页 §5.1 Implementation Details。式 5 对当前生成位置的视觉 attention 跨 head 平均，然后选低 attention token。图 2 的选取在当前参考路径的浅层之后，后续层才限制视觉可见性；图中省略 KV cache 是讲解方式，不等于算法禁止 cache。

论文此版本第 7 页的实验设置为 Shikra/LLaVA1.5/LLaVA-NeXT 的 i=3、最低 10%，Q-former InstructBLIP 为 i=5、最低 10%；第 6 页图例另用 top-k 50。固定源码默认是 `fast_v_agg_layer=2`、`fast_v_attention_rank=100`、`fast_v_inplace=False`。本次遵循固定源码，不能把新论文参数静默移植进来。源码循环零基 index2 读取前一个 index1 层的 attention，遮罩 index2 及以后的层；这里“第 2 层 attention”指物理第二个 block。

## 逐项比对与确定差异

| 项目 | 固定官方源码 | 旧端口问题与本次处理 |
|---|---|---|
| 视觉区间 | `image_start`、含尾端的 `image_end` 转成长度 `end-start+1`；LLaVA wrapper 在 image placeholder 展开后计算位置 | 端口按 processor 输出重复 image token id 计算连续 start/length；必须实际已经展开且与 decoder 的 Q/K 轴对齐。单 placeholder 不能被当成一个视觉 token。现在少于 100 或非连续、无映射直接拒绝，前向还核对 Q/K 轴 |
| 选取 | 前层 attention 的 `[batch=0, heads, last_query, visual_keys]`，mean heads 后 `topk(100, largest=False)` | 旧版 `min(100,length)` 会把不足 100 的视觉 token 全保留，和官方固定100不等价；已移除 clamp |
| attention 来源 | 同一次参考 forward 的前层 `layer_outputs[1]` | 旧版只在 `active=False` 的外部 clean forward 捕获。不同参考 prompt 或多步 replay 时不是同一路径/长度；现每次参考前向内部捕获自身第二层 |
| 下游层 | index2 构造一次 mask，此后层复用同一选择 | 修复后在 index2+ 施加同一当前 attention 对应的选择。检查不在同一次 forward 中变化 |
| causal | 官方 `_prepare_decoder_attention_mask` 合成 causal 与视觉 key mask，不是任意全连接 | 端口保留已有浮点 additive 4D mask；没有显式 mask 时重建全 KV 的 causal mask。拒绝 bool mask 与隐式 sliding-window。输入限单条无 padding，从而可和官方用全1重建的 mask 直接比较 |
| matched prefix / cache | normal/reference 各有 `model_kwargs` cache，同一生成 token prefix；参考浅层 attention 来自自己的 cache | `HFSession` 各有 output/cache；其同 prefix 复用、非单步扩展时 prefill+逐 token 重放不需修改。SID 现在在每个 `_call` 重新计算本次选择，prefill 与每个 replay step 都有自己的 Q/K 长度 |
| 多 session | 官方生成调用显式切换当前分支，不存在本项目多参考实例生命周期 | 旧 `SIDSession.__init__` 关闭 `backend.sid_control` 导致第二 SID 拆掉第一 SID 的 hooks；mechanism 的 qg/r 交错可使 qg 实际不剪枝。现每个 session 保存自己的 control；仅 `_call` 期间安装 hooks，正常/异常退出均恢复。构造新 session 不改共享模型 |

排序 top-k indices 只改变 mask 赋值次序，不改变保留集合。原始官方 mask 使用 dtype 最小有限数，端口被阻断位置用负无穷；核验同时记录这种数值表示差异和屏蔽位置，并要求 logits **逐元素完全相等**，不靠容差宣称等价。

## 可适用结构，不能凭模型名宣称支持

必须同时满足：batch=1、无 padding；可访问至少三个 decoder block，路径为 `language_model.model.layers`、`model.language_model.layers` 或 `model.layers`；第二层有可返回 `[1,heads,query,key]` 的 eager self-attention；视觉 id 为显式 `img_ctx_id` 或 config `image_token_id`，processor 已展开成至少100个连续视觉位置；前面有可见 BOS/text，避免早期 query 全部 key 被遮罩；全长 KV 轴与展开 input_ids 加当前生成前缀一一对应；后层接受浮点 additive 4D causal mask，或确实是无 sliding-window 的完整 causal attention。

不满足结构就明确报模块路径、视觉映射、attention 输出或 mask 形状原因。Q-former 少于100、LLaVA 未展开 placeholder、多段 image span、batch padding、压缩/裁剪 KV、无法显式确认的 sliding attention 都不能直接标支持。层路径/attention API 的静态匹配只是候选条件，必须跑实际版本的数值核验。共享 backend 前向应串行，代码拒绝嵌套 SID forward；这里未声称任意多线程 clean/SID 并发安全。

## 已运行 CPU 回归

`tests/test_sid_sessions.py` 使用真实随机初始化的四层小 Llama（32 hidden、4 attention heads、2 KV heads），运行 PyTorch attention、真实原生 KV cache 和 `HFSession` 状态流，不用仅返回预设 logits 的 mock。CUDA_VISIBLE_DEVICES 置空，项目 venv 执行，与既有后端测试合并为 6 passed（8.09s）；追加“SID 必须实际改变 clean logits”的非平凡性断言后，4 项专属测试再次通过（6.15s）。模型专属 `.environments/mprisk-tf553` 无 pytest，首次命令明确失败后改用项目既有测试 venv，没有装包。

测试覆盖：两个不同长度 prompt 的参考会话交错；空前缀、单步增长、重复前缀、回退/分叉后 replay；每次与从零构造的官方 mask-core reference 比对；SID 剪枝实际改变参考 logits，而真实 clean logits 不受构造 SID 或 SID 执行影响；故意异常后 hooks 与前向函数恢复；四个 query row 的未来屏蔽和已有屏蔽位置保持；拒绝不足100、非连续视觉span及 bool mask。此 CPU fixture 只证明软件状态和核心构造回归，不代替真实多模态登记模型支持。

## 待主线程安排的最小真实前向核验

脚本 `verification/sid_reference_check.py` 不下载、不改 spec。从上述固定官方源码的 AST 提取并执行 `_make_causal_mask`、`_expand_mask`、`_prepare_decoder_attention_mask`，以及 mask-mode 最低100原始片段（约 785–804 行），同时验证 working blob 未偏离固定 commit。以同一 native checkpoint 分别运行修复实现和原片段 mask-core 移植 oracle，输出原片段全文/行号/源 hash、每次前向每个下游层的 indices、mask 形状、末 query 屏蔽位置、全mask屏蔽位图 hash、causal逐行核对、有限最小数与负无穷计数、logits hash/最大差。

固定 interface16 第一张真实图、两个不同长度真实 prompt；同一固定 tokenizer token 序列包含增长、重复、回退、分叉重放；两参考交错，并每次额外与 fresh oracle 比较，避免双方 cache 同步犯错蒙混通过。需要 actual/oracle logits 与 fresh reference logits **完全相等**，所有检查为 true 才写 passed。失败写明具体 error，并不填登记文件的机制验收字段。

示例（GPU绑定与锁由主线程既有调度提供）：

```bash
PYTHONPATH=src TMPDIR=$PROJECT/cache/tmp "$PROJECT/.environments/mprisk-tf553/bin/python" verification/sid_reference_check.py --spec configs/runtime/llava15_7b.json --output outputs/verification/llava15_7b_sid_reference.json --device cuda:0
```

证据边界：oracle直接执行官方原始选择与mask函数，但为在当前登记 checkpoint 上数值比较，共用 native backbone 和 hook transport。它不是完整旧 transformers 官方 fork 的端到端重新运行；测试不涉及 contrastive decoding operator、解码任务准确率或论文完整复现。只有 CPU 已执行，本记录制作时 GPU 部分未由本代理运行。

## 本次文件身份

- `src/kdm/models/sid.py` SHA256 `a74b3e784a7f81884a0069cd71baeaf11f4e4623203cd87e5e59890562cd3b83`
- `src/kdm/models/hf.py` SHA256 `0b57d836eb5abf2e88d8e9af39f4665255a1b599f4ad65446051fb4acee2e11e`
- `verification/sid_reference_check.py` SHA256 `c7b8fadf89409769bc50b98524a2639948f03a8785806129f50f757061532b92`
- `tests/test_sid_sessions.py` SHA256 `1c06bdf6fce7ae59518b3632f7a03b2ce5f8f65a7bf295f3f27ef5bef4ba21f4`

## Evidence storage

The complete layer-index traces are retained at their original project paths without rewriting. Compact `*_sid_reference_summary.json` receipts preserve method/runtime identity, checks, per-visit errors and each layer event's scalar/mask identity; only repeated selected/blocked index arrays are represented by their lengths. `detail_evidence` records the original full file path, size and SHA256. The deterministic derivation is `verification/summarize_sid_proof.py`. The previously committed LLaVA7 trace remains in Git history; its current tracked replacement is the compact receipt. Other complete traces are excluded from Git. These are interface proofs, not formal research responses or distributions.
