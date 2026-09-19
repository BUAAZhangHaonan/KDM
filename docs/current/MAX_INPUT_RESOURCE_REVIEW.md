# 最大视觉输入资源边界（当前单卡 v1）

此核验只覆盖全量计数中视觉token数最大的实际VizWiz样本、空/1/2固定前缀和三种真实session组合。不进行答案采样、任务打分或方法效果统计；不声称覆盖每个问题长度或完整32生成步的绝对峰值。各组合独立进程，前一组真正退出后才加载下一组，未将不同方法的session叠加成不存在的负载。

`verification/max_input_resource_check.py`复用pipeline.sessions：layer为一个need_layers图像分支；instruction_vcd依次guided clean、unguided noise、unguided clean三图；CDA依次text prior、image context、guided image abstention、null text prior、同尺寸均匀灰图null context，三图加两text。固定前缀由字面常量编码、没有从logits采样。输出仅记录有限性、形状、session输入形状、实际峰值、错误及身份；保留原始traceback，不按其通用建议变更分配器/输入/参数。

|模型|视觉token|组合|结果|完成forward数|峰值allocated bytes|峰值reserved bytes|
|---|---:|---|---|---:|---:|---:|
|qwen25vl|6417|layer|passed|3|20401233408|21611151360|
|qwen25vl|6417|instruction_vcd|passed|9|24004324864|24354226176|
|qwen25vl|6417|cda|passed|15|24023890432|24496832512|
|qwen3vl|4860|layer|passed|3|21734113792|23563599872|
|qwen3vl|4860|instruction_vcd|OutOfMemoryError at forward/0/neutral|2|23804253696|24222105600|
|qwen3vl|4860|cda|OutOfMemoryError at forward/0/null_context_image|4|23821041152|24110956544|

Qwen3VL两次失败均发生在空前缀第三图像分支；先前成功分支的有限性观察保留。二者为独立组合的首次尝试，不是对同一失败的重试。没有减小输入/改变dtype/关闭分支/自动offload。Qwen25六千余视觉token通过三组不等于给整个研究内存足够的全称保证。

GLM当前等待SID同源oracle进程完成，随后再按原spec独立核验上述三组；不在此伪填结果。Qwen3双卡映射仅为只读提案，见QWEN3VL_DUAL_GPU_RESOURCE_PROPOSAL.md，未改变当前单卡spec。

逐项完整记录与同名log：
- `outputs/verification/qwen25vl_max_input_resource_v1_layer.json` SHA256 `44082a079087d8e0d75b35f1bf2686c6dc8a117a3d9182a9b5631a39c1a36640`
- `outputs/verification/qwen25vl_max_input_resource_v1_instruction_vcd.json` SHA256 `7388f2501ed20d62920d3eb57a8e43baf736454fece4226a33e6a9779afefda3`
- `outputs/verification/qwen25vl_max_input_resource_v1_cda.json` SHA256 `a930be37bb8857ec9b0b7d2cb64b97330d8eb976f037d1a8c23416d7f6438443`
- `outputs/verification/qwen3vl_max_input_resource_v1_layer.json` SHA256 `9800445dade9d70a37de4a59fdfd73f3a9f58e6ff677c193792a77948943faf3`
- `outputs/verification/qwen3vl_max_input_resource_v1_instruction_vcd.json` SHA256 `556dd462f9051347c11d4446ee547e197d4d5bc6fc6d131adc2c26fc5c5c39eb`
- `outputs/verification/qwen3vl_max_input_resource_v1_cda.json` SHA256 `5d5fb9ef59a22b7a3dd85cf9976f8edf5fff34baddd821907b15fe976a30bb1b`
