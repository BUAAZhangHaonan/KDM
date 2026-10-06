# HallusionBench 128-token 评分接续

所有九模型沿 Food 冻结工作点，统一最多生成 128 个新 token。新的生成、验收与评分保存在 `outputs/hallusion_blind128_20261006/`，旧 EOS-only 输出与论文原评分保持原样。

`terminated` 表示真实 EOS；`truncated` 表示达到 128-token 上限；`finish_reason` 为 `eos` 或 `length`。三个字段不决定答案正确性或弃权。正确性根据当前实际生成的全部文本与原问题、官方完整参考判断；弃权独立判断。官方 `unclear=2` 保留，准确率计零；未决保留空值，不默认记错或弃权。没有生成或执行失败的输入保留为缺口，不进入已生成数。

原生 EOS 回复在前 128 个 token 内结束时可精确复用；超过 128-token 的历史完整回复必须按真实 token 前缀和原 tokenizer 解码，并保留来源证据。不同完整文本不能复用语义标签。来源与预算兼容性由 `generate128.validate_rows` 验证，评分只接受本协议的已验收 raw 清单。

```bash
venv/bin/python workflows/hallusion_blind/audit128.py
venv/bin/python workflows/hallusion_blind/score128.py \
  --out outputs/hallusion_blind128_20261006/scoring/snapshot_001
venv/bin/python workflows/hallusion_blind/audit128.py
```

`score128.py` 复用冻结 Direct 评分和旧/新目录里实际完成的 Luna medium 判断，只按完全相同的问题、完整回答与参考连接。每个实际准入条件的最终分母固定为 951，完成前只显示观测指标。逐样本保存终止状态，条件表保存达到上限数和比例。

SID 保留原架构范围：MiniCPM/Qwen3.5 不改造适配；Qwen2.5/Qwen3-VL 的完整面板分别因 30/59 题视觉词元不足 100 而排除。68 个候选面板共 64,668 条；排除原因单列，不能删题改变分母。

2026-10-06验证：新旧评分局部测试16项通过；EOS-only `snapshot_011` 的17,823条实际 raw 重新评分，14,064条已决及全部未决计数一致，逐样本的质量、弃权、QA、token数和来源字段均与原快照一致。旧 `scores.jsonl` SHA256为 `b6fd1aabdcf312fe589c3ed99b9aae1aa2ac2ffc616acf6591ab96b5658ce0c7`。新128-token面板完成量仍由真实回执逐条统计。
