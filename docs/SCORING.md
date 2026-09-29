# 主指标评分

## 数据与分母

Food-101已有图像类别标签。当前五模型每个条件使用2424个eval样本，保存352个完整条件。原始回答、条件、sample ID、模型、方法、主marker、参考marker和数据划分共同标识每条结果。

## 类别识别

先根据回答和已有抽取记录确定明确主菜，再把该主菜映射到101类。推断阶段使用全部规范类别，目标类别仅用于最后比较。接受原类别拼写及下划线转空格；大小写与首尾排版格式统一。烹饪/风味修饰保留在原始名称中，类别匹配使用主菜中完整的规范词边界。

- The dish is fried rice. → fried_rice。
- Chocolate ice cream → ice_cream；chocolate保存为修饰。
- Seared scallops with mashed potatoes → scallops；mashed potatoes保存为配菜。
- Shrimp and grits → shrimp_and_grits，保留复合类别名称。
- Sushi or sashimi → multiple_primary，按单标签题的无唯一预测类别计0。
- Sushi and sashimi两项同级主菜 → multiple_primary，计0。
- 仅明确配菜与解释的类别词保存在角色字段。

单复数、同义词、拼写替换和连字符形式沿用原文，词汇表保持101规范名称及下划线转空格形式。无匹配类别的明确回答计0。抽取缺失或主配菜关系待判断的回答进入最小复核清单。

主列为canonical_name_in_primary。literal_extracted_name记录完整抽取名称的等值匹配敏感性。两个列使用同一原始回答及分母。

## 弃权

任务规定的完整marker及明确的拒答/无法识别表达记为弃权。作答同时带likely、appears等语气词时保留答案及其置信语气。陌生字形、生成残片和错误菜名保留为正常作答输出。逐行保存abstain值及判定来源。

## 参考GT与比较

已有Qwen25、Mini26、LLaVA操作性参考GT及130项普查冲突保存为已接受来源。五模型统一规则的参考GT从候选排名及十次独立回答计算，并输出逐问题引用。主表按完整条件匹配对应baseline，列出分子、分母、效应量和按Food-101类别聚类的2000次配对bootstrap区间。

## 来源

Food-101论文采用测试图像正确分类比例：[Bossard et al., ECCV 2014](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf)。
