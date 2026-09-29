# KDM Paper Data Export — 2026-09-29

论文分析入口为 scores.parquet + conditions.csv。所有评分、参考值、控制选择和回答来自当前冻结资产；本次执行CPU格式转换、已有键连接、计数核对及图片复制。

| 内容 | 已导出数量 |
|---|---:|
| 正式逐样本评分 | 853,248 |
| 完整正式条件 | 352，每条件2,424个eval输入 |
| 参考记录 | 24,240，dev/eval各12,120 |
| 复制控制 | 200条件，484,800条既有选择 |
| 去重QA / 评分细节字典 | 44,782 / 8,222 |
| 原始真实案例 / 图片 | 12 / 12 |
| 案例独立试答 | 120 |
| 运行耗时记录 | 1,147组，覆盖853,248条回答 |
| 已有配对比较 / 本包40组点估计核对 | 372 / 40 |

当前数据文件合计约 31.59 MB（未压缩，最终压缩体积见version_receipt/package receipt）。导出检查结论：通过，计数差异0。

正文核心计数：LLaVA/UNKNOWN的Direct、VCD、IP-VCD正确数689、852、852；uniform原有集合287，保留33、61。MiniCPM/UNCLEAR对应733、873、903；uniform原有集合374，保留239、249。MiniCPM的VCD纠正226个，IP-VCD保留212个，复制控制正确722个。accepted历史参考字段分别保留在表中。

已有字段覆盖：全部主正确性、字面正确性和弃权值已决。QA表保存14,134条已有复核全文，其余QA提供直接原始文件/行号；12个案例含完整原文。既有抽取主名称有779,420行null，直接保留。辅助行为表有334行空标签，最终score.abstain字段完整。已有耗时覆盖全体回答，细分prefill/decode、峰值显存和batch size的可用情况逐项列于availability.csv。sources.csv提供当前路径与历史归档映射。

## 读取

```python
import pandas as pd
s = pd.read_parquet('scores.parquet')
c = pd.read_csv('conditions.csv')
r = pd.read_parquet('references.parquet')
wide = s.merge(c, on='condition_id', validate='many_to_one')
# 方法配对使用同一个sample_id以及完整匹配的条件。
sel = pd.read_parquet('control_selections.parquet')
control_rows = sel.merge(s, left_on=['selected_condition_id','sample_id'],
                        right_on=['condition_id','sample_id'], validate='many_to_one')
```

schema.md列出字段类型、原字段映射、联合主键和空值含义。cases.jsonl与cases/images/可直接用于案例排版。正文主线和用户写作要求位于context/paper_start；最终协议与评分说明位于context/final_project。version_receipt.md记录工作目录、真实Git状态和正常推送结果。

## CPU复现入口

在规范项目根目录依次执行scripts/paper_20260929/export_tables.py、export_runtime.py、export_cases.py、export_runtime.py --comparisons-only、finalize_export.py。Python依赖为pyarrow、pandas、Pillow；SQLite来自标准库。export_tables.py针对新的导出目录进行独占创建，重复执行应另设输出目录。已有模型输出与评分保持冻结。
