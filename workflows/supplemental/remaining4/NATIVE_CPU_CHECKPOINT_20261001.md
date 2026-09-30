# 四模型原生基线CPU检查点

当前来源为`receiving_manifest_20261001_0115/native_parts_snapshot_d4030_20261001_0115.json`与`native_parts_snapshot_6403_20261001_0115.json`，合计44个不可变分片、14560条唯一任务。接收程序按source host、claim、part与原SHA排除已经接收的13个分片；本轮仅传输并评分31个新增分片的10928条回答。旧3632条只读取已验收score表与manifest，不重新读取raw。

新增分片的当前评分目录为`outputs/supplemental/remaining4/native_cpu_20261001_0115/scores/v2_root40_closed`。CPU实际验收通过原身份、runtime admission、真实算子proof、完整source SHA、行SHA、互斥任务键、稳定种子、两路无引导prompt、注册算法及参数。rule/精确QA解决10888条；其余40个完整QA经过实际Luna medium标注与root逐项复核后，仅对应40条score行发生更新。10928条主正确性、弃权和完整名称全部已决。原Luna文件和原边界包保留，边界包SHA256为`58b6a5045cabb39f745d02e04025e0bc0f68c356a5e50b4aea66477515540b6e`。

累计检查点为`outputs/supplemental/remaining4/native_cpu_20261001_0115/checkpoints/v2_14560_root40_closed`。只合并两个已验收score文件，保持回答、原始身份与全部历史裁定，未读取raw。14560条的主正确性、弃权和完整名称未决数均为0。

| 注册模型 | 方法 | 已接收 | 未接收 | 主正确性未决 | 弃权未决 | 完整名称未决 |
|---|---|---:|---:|---:|---:|---:|
| OpenGVLab/InternVL3_5-8B | VCD | 2424 | 0 | 0 | 0 | 0 |
| OpenGVLab/InternVL3_5-8B | M3ID | 2424 | 0 | 0 | 0 | 0 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | VCD | 2424 | 0 | 0 | 0 | 0 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | M3ID | 2424 | 0 | 0 | 0 | 0 |
| microsoft/Phi-3.5-vision-instruct | VCD | 16 | 2408 | 0 | 0 | 0 |
| microsoft/Phi-3.5-vision-instruct | M3ID | 0 | 2424 | 0 | 0 | 0 |
| Qwen/Qwen3-VL-8B-Instruct | VCD | 2424 | 0 | 0 | 0 | 0 |
| Qwen/Qwen3-VL-8B-Instruct | M3ID | 2424 | 0 | 0 | 0 | 0 |

六个完整输入条件逐一核对101类别、每类24张。八条件总计划19392条，剩4832条任务键仅属于Phi。`missing_keys.jsonl`为来源核对后的缺口计划；已有claim与活动区间继续保持。未决字段保留为空值。该评分检查点的参考字段明确标为尚未连接，参考精确率、召回率及完整方法效果未在本检查点计算。

最初3632条的24项root裁定与独立Ice cream cone完整名称裁定已追加到`native_cpu_20260930_2338/annotation/root_reviewed24_20261001`。24 QA的25个实际源成员与cone QA的3个成员均完成原始完整QA、key、model、sample、identity、seed及行SHA验证。原Luna文件保留；作者为root实际GPT-6.1 Sol/max，prepared_by记录assets，未知call ID为空。封存评分`scores/v2_root24_cone1_closed`的主正确性、弃权和完整名称未决数均为0。

本轮必要源码：`receive_native.py`新增已接收part排除与身份/SHA核对；`score_native.py`新增上一闭合score的任务键互斥核对；`merge_native_scores.py`新增仅score表合并、八条件覆盖与真实缺口键。评分函数继续复用冻结canonical函数及已有`infer_qa/score_target`，VCD与M3ID注册科学参数保持原样。

`review_native_boundary40.py`保存实际root的40项裁定，其中16项同级或竞争主菜、24项唯一主菜。唯一主菜的canonical_override均为空，完整名称保留原连续span；多个主菜使用multiple_primary。原Luna完整判断、Luna行SHA、原成员和既有实际raw CPU验证链保存在独立`annotation/root_reviewed40_20261001`目录。root作者记录为GPT-6.1 Sol/max，prepared_by为assets，未知call ID为空。40个来源逐项核对QA、key、模型身份、原行SHA和稳定seed，更新score时未重新读取raw；其余10888个score对象经过独立比较完全相同，原Luna文件SHA保持相同。

实际CPU使用`/home/g203-4028/projects/knowledge-deficit-mitigation/venv/bin/python`，语法检查、10928条评分、独立验收及14560条合并均已运行成功。验收文件保存脚本、来源和产物SHA；未知项误设已决数为0，新增GPU初始化与新增API调用均为0。本地镜像位于`E:/OneDrive/文档/Playground/kdm_remaining11_20260930_140337/remaining4/native_cpu_20261001_0115`。
