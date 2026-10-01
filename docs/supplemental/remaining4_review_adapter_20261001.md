# 补充裁定关系字段接入

2026-10-01。有限核对发现，部分新 Luna/root 裁定使用字典形式 `name_relations`，而原共享评分函数仅读取列表形式的关系。明确的 `multiple_primary=true` 也没有被这条补充调用转换为原评分函数支持的覆盖字段。

实际例子为完整问题 `What specific food is shown in this image?`、完整回答 `Cookies and donuts`。原裁定明确两项同级主菜、没有主配关系；旧补充记录却得到 canonical=1、literal=null。按最终 SCORING.md 的单标签多主菜规则，两列应为 0，语义弃权仍为 false。

修复只在 `workflows/supplemental/remaining11/score.py` 的调用适配内进行：复制裁定对象，把关系字典转换为原函数读取的 `{name, role}` 列表；把明确的 `multiple_primary=true` 接入原 `multiple_primary` 覆盖。冲突的唯一类别覆盖直接报告，不静默选择。原裁定文件与原评分版本继续保留。

中央实际有限核验保存于 `outputs/supplemental/remaining4/audit_20261001_1600/adapter_scope_v1/known_QA_check.json`。上述 QA 对全部 101 个目标的 canonical/literal 均为 0；完整复合名 `Fried ice cream` 仍保持一个名称，不拆为同级主菜。Python 编译检查通过。

影响范围以已知裁定 QA 与已评分记录的精确键连接确定；必要的评分与参考变化追加独立版本。该核验不代表所有历史标签已重新审阅，也不把仍未完成的增量合并写成科学验收。原五模型的评分函数、输出、参考与论文正文保持冻结。
