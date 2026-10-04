# 最终统计摘要

全部百分比在CSV中以0–1小数保存；文中乘100。百分点差值列已按百分点保存。

| 任务 | 方法 | 九模型平均质量 | 九模型平均J |
|---|---|---:|---:|
| Food | vcd | 40.5207 | 40.5345 |
| Food | instruction_vcd | 40.0990 | 46.3284 |
| Food | instruction_m3id | 38.2013 | 44.2428 |
| VizWiz | vcd | 23.2118 | 26.0981 |
| VizWiz | instruction_vcd | 20.1128 | 39.8177 |
| VizWiz | instruction_m3id | 20.1628 | 40.7574 |

Food质量为整体准确率；VizWiz质量为行为调整后的答案文本共识信用。P/R的分母、零弃权未定义值、工作点选择及来源身份见各CSV与SOURCE_MAPPING.md。

Food主比较69行，VizWiz主比较45行；同指令与参考去引导条件216行、配对差值72行；计时14个匹配组共42行。