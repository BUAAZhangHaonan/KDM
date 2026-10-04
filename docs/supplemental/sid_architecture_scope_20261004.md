# 原生 SID 的架构适用范围（2026-10-04）

本次继续使用已核对作者实现的第二块注意力聚合、最低注意力的 100 个视觉位置、alpha=0.5、无候选裁剪的贪心组合；模型精度、图像处理和生成预算保持登记值。

## Gemma3-4B

Gemma3 的第二块仍是标准 softmax 自注意力。Food 固定 8 个输入的实际预处理序列均为 282 个词元，其中连续视觉位置为 5–260（256 个），最大 32 个生成词元不会超出其 1,024 词元滑动窗口。原生模型对图像块使用双向掩码，同时保留不同层的滑动/全局掩码。

SID 参考分支在单次 forward 作用域内使用 eager attention，以取得第二块的实际权重和显式原生掩码；仅将未选中的视觉键列设为负无穷。原有掩码的其他值逐项保留，退出作用域后恢复 attention dispatcher。主分支没有设置或掩码更改。

8 个实际 eval 输入通过全生成路径准入：每个生成词元后续 32 层均核对真实 attention 的官方 rank-100 选择、禁止视觉列和原生 mask 保留；未发现 hook 泄漏，首题主分支 SID 前后 logits 完全相等。准入耗时 53.55 秒（其中模型加载 15.56 秒），输出计入 2,424 题正式分母。生产已按剩余 2,416 个输入的稳定 ID 划为互斥 4 片。

## Qwen3.5-4B

当前冻结检查点的 32 个解码块中有 24 个线性注意力块、8 个完整 softmax 注意力块。第二块类型为 `linear_attention`，执行 `Qwen3_5GatedDeltaNet`，保存卷积状态和递归状态，不产生原 SID 所需的逐词元 softmax attention 矩阵和视觉键值缓存。普通完整注意力首次出现在第四块。

把聚合移到第四块，或为递归状态设计视觉影响屏蔽，都会改变原 SID 算法。本轮不实施这些变体。Qwen3.5 的 SID 记为架构不适用，不能把无结果记为 0 分。

已核对的本地权威来源：冻结检查点 `config.json` SHA256 `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`；当前 Transformers `modeling_qwen3_5.py` 的 `Qwen3_5GatedDeltaNet`、`Qwen3_5DecoderLayer` 及模型 mask 分派。结构清单位于 `qwen35_architecture.json`。

建议论文脚注：**Qwen3.5 的第二解码块采用递归线性注意力，不提供原生 SID 所需的 softmax 注意力及逐视觉键屏蔽接口，故不报告该组合。**

English footnote: **SID is not reported for Qwen3.5 because its second decoder block uses recurrent linear attention, without the softmax attention and tokenwise visual-key masking required by the original method.**
