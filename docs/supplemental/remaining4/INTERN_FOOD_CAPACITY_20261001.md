# InternVL Food 双卡共享准入

独立登记仅授权 6403 A100 的 InternVL3_5-8B Food 正式提取：两个物理 GPU 均需在加载前有 32,768 MiB 可用显存，使用既有 slot 0:2/1:2。原 bf16、36 层的 18+18 放置、每卡 22 GiB max_memory、模型权重、提示、种子和方法参数保持。

早期容量不足的加载请求在 claim 与模型初始化前拒绝，原失败记录保留。OneVision 与 Qwen3-VL 保留原 owner、区间和共享负载。新增守卫位于补充入口 `workflows/supplemental/remaining11/execution.py`；核心 `src/kdm/execution.py` 未修改。

真实 8 题准入于 2026-10-01 20:09:11 CST 完成，随后同进程接续原来的 24,046 个未完成键。实际来源：

- claim：`6403_internvl35_8b_food_retained_first8_s0of4_20261001_1848`。
- admission SHA256：`bd79a0e42a8183c0870a09bcc674067ccdda9c6f648e5bfcc664813c98ceba77`。
- `outputs/supplemental/remaining4/dispatch_20261001_1600/event_2038/snapshot_6403_event_2038.json`：8 题已完成及后续不可变 512 条分片。

这证明当前实际共享范围的准入与接续，不证明其余机器、其他数据集或其他模型准入，也不提供尚未实测的全队列 ETA。其余三份 InternVL Food 区间仍按各自主机的真实准入和互斥 owner 派发。
