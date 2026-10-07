# HallusionBench 九模型接续检查点

2026-10-07，生成验收18:08（北京时间）；资源检查18:02。

- 独立盲测951题，68个已准入条件，共64,668条；统一128词元预算。
- 已验收63,831条（98.71%），余837条，仅Qwen3.5-4B、Qwen3-VL的DoLa；七个模型提取齐。
- 6403两张A100与K100空闲后，源队列按当前分片封存，2,452个真实未完成键重新分到九张卡。新旧生成交集与新分片交集均为0，全部目标worker已通过原runtime准入并写入不可变首批。
- 4028 GPU0/1/5与4029 GPU2、6403 GPU0、K100 GPU0承担Qwen3.5；4028 GPU4、4029 GPU1、6403 GPU1承担Qwen3-VL。4030未连接。
- 原精度、batch、方法、提示、seed和Food冻结配置保持。提取预计今晚19–21点，按真实卡释放接续长尾。
- snapshot_013读取63,831条；质量和行为均已决25,405条。待决唯一完整问题/回答/参考组合35,962，最终完整条件0/68。
- 原作者确认11个批次原文件使用了默认/关键词脚本，704行原始来源已隔离；其中此前激活的016已撤回并真实重标。其他原始脚本输出没有进入正式评分。新实际语义结果经根线程有限复核、完整绑定与全量结构验证后使用独立文件接续。
- 语义标注与复核粗估6–12小时，生成完成不计作最终指标完成。Food/VizWiz与论文冻结资产保持。

服务器证据（项目根目录相对路径）：

- `outputs/hallusion_blind128_20261006/CURRENT_STATE.compact.json`
- `outputs/hallusion_blind128_20261006/resource_inventory/qwen_fast_tail_joined_proof_20261007.json`
- `outputs/hallusion_blind128_20261006/registration/qwen_fast_tail_20261007_1610/transfer_manifest.json`
- `outputs/hallusion_blind128_20261006/scoring/snapshot_013/receipt.json`
- `outputs/hallusion_blind128_20261006/annotations/nonsemantic_source_quarantine_20261007.json`

