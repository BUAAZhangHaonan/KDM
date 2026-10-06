# HallusionBench：统一128生成预算（2026-10-06）

用户在2026-10-06授权将九模型Hallusion盲测由自然EOS无上限改为最多128个生成token。
题目为完整951题，不划开发集，不改变Food选定配置、模型、精度、batch、图像、提示、seed和任何其他解码参数。
新入口为 generate128.py，独立目录为 outputs/hallusion_blind128_20261006。旧generate.py、原13个算法/适配器来源及旧结果保持。

## 终止与评分

EOS与length分别记录：自然EOS可以在128内任何位置；非EOS结果必须真实生成到128，保存terminated=false、truncated=true、finish_reason=length。
不强制添加EOS。按实际可见完整文本的官方Hallusion兼容性规则评分，截断本身不等于答错或弃权。
不能判定的文本保留unclear=2，在准确率中计0；语义弃权另判。待裁定不默认计错。每条件分母951。
截至预算之外的潜在回答不计分。旧Food/VizWiz冻结评分不回写。

## 真实输出复用与分片

reuse128.py只复制真实EOS-before128，或用原tokenizer解码已记录完整greedy轨迹的前128个实际token。
长回复先全原文roundtrip检验，保留原始完整token/text、文件SHA、行号、原owner身份及原时间。
prefix的未知生成时间不伪造；不用于当前ETA。新GPU分片通过 --task-keys 只接续实际缺键。
新来源15项逐机核对，显式assignment身份绑定；旧注册68候选面板64668条与SID架构范围保持。
当前18191条旧轨迹可复用，新GPU缺口46477条，计数由registration/gap_budget128.json生成。
已完成的32条EOS pending在停旧队列时用原sealer保住，未完成输入记录预算替换中止，不评分。
独立恢复namespace保留真实OOM/释放源，排除已生成键；验收再次拒绝任何非精确副本重复。

## 已验证的非思考配置

Qwen3.5原登记thinking_mode=disabled；视觉build与text-only模板均显式enable_thinking=False。
128预算不改变该配置。3,724-token示例实际来自Qwen2.5-VL Direct，内容是No之后大量平方根小数。
