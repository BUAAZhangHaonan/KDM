# HallusionBench 九模型提取完成检查点

2026-10-07，全量生成验收19:26（北京时间）。

- 独立盲测951题，68个适用条件，64,668条全部生成，零重复、零缺失。原始身份、不可变分片和有限概率核验通过。
- 新生成46,477条，精确复用18,191条；自然EOS 52,454条，达到128词元上限12,214条。根据实际文本分别判断质量和弃权，长度终止不自动记错或记为弃权。
- 先将2,452个未完成键分到九卡，再将释放的143个长尾键分成六个互斥分片。移交交集为0，原参数与runtime身份保持；六个分片全部完成。
- 4028 GPU0/1/4/5、4029 GPU1/2、6403 GPU0/1、K100 GPU0全部释放，KDM活动生成worker为0。4030未连接。
- Direct、VCD、DoLa、DeCo、CDA视觉迁移、IP-VCD及IP-M3ID九模型均完成。SID五模型完成原方法；MiniCPM与Qwen3.5架构不适用，Qwen2.5与Qwen3-VL部分输入不足100视觉词元，完整951题SID面板不进入比较。
- snapshot_014读取全部64,668条，正确性和行为均已决29,865条；待决去重组合34,070，最终完整条件0/68。三个实际GPT-5.6 Luna medium槽位与root有限复核继续，暂估6–10小时。
- 新增完整回复规则仅用于128词元补充评分：完整UNKNOWN/UNCLEAR/UNSURE或I cannot identify it按原参考类型评分，完整回答与完整GT精确相等直接判定。原已决记录优先；长回复、修正、竞争答案和未完成推理保留实际语义判断。
- 规则已在951题上通过18,069项有限验证，既有已决记录无冲突；此前704行非语义脚本来源保持隔离，原文件保留。实际Luna新来源经完整绑定、结构校验及root内容复核后单独激活。
- Food、VizWiz及论文冻结资产保持。提取完成与最终指标完成分别记账。

服务器证据（项目根目录相对路径）：

- `outputs/hallusion_blind128_20261006/CURRENT_STATE.compact.json`
- `outputs/hallusion_blind128_20261006/generation_coverage_by_model_20261007.csv`
- `outputs/hallusion_blind128_20261006/resource_inventory/last_dola_generation_complete_20261007.json`
- `outputs/hallusion_blind128_20261006/resource_inventory/generation_complete_gpu_release_20261007.json`
- `outputs/hallusion_blind128_20261006/registration/qwen_last_dola_20261007_1850/transfer_manifest.json`
- `outputs/hallusion_blind128_20261006/scoring/snapshot_014/receipt.json`
- `outputs/hallusion_blind128_20261006/annotations/complete_response_rule_validation_20261007.json`
- `outputs/hallusion_blind128_20261006/annotations/nonsemantic_source_quarantine_20261007.json`
