# KDM · 免训练幻觉抑制方法在知识缺陷人群上的行为

**Training-Free Hallucination Mitigation under Knowledge Deficit**

解码期的免训练幻觉抑制方法（VCD / MIB / LCD 类对比解码）只重新分配置信度，不改变模型
知道什么。它们对校准的影响方向由评估人群的基率决定：模型大多答对的人群上校准改善，
大多答错的人群上同样的机制变成自信的错误；且输出侧统计量无法事先区分两侧。第五阶段把
这一主线变成可检验的定量规律：**ΔECE 可由人群干预前的准确率/校准差预测，变号点在
准确率 ≈0.6**（food101，Qwen3.5-4B/9B；LLaVA-v1.6 与 InternVL3.5 分层级方向一致）。

## 五轮迭代（每轮有独立的预登记与结论文件）

| 阶段 | 问题 | 结论文件 |
| --- | --- | --- |
| 一（v1） | LVIS/COCO 频率分层：低频对象上抑制方法是否有害 | `v1-lvis-frequency/CONCLUSIONS.md` |
| 二 | food101 实测分层（低/高准确率组），错误置信度为主结局 | `PILOT.md` `CONCLUSIONS.md` |
| 三 | 开集失败两因分解（可及未输出 vs 不可及），跨模型跨域 | `STAGE3.md` |
| 四 | 知识可及性测量的边界：名称受限 LL 探针门槛全败、粒度错配 | `STAGE4.md` |
| 五 | 收束：基率预测律、变号点、与最接近工作的对照 | `STAGE5.md` |

每个阶段的判定阈值在运行前写入 `PREREGISTER_STAGE3/4/5.md` 并先于数据提交；
v1 目录保留其完整迭代历史（`git log -- v1-lvis-frequency/`）。

## 目录结构

```
code/                    全部实验与分析代码（入口 run_all.py，阶段脚本 stageN_*.py）
data/                    样本清单、分层定义、近邻表、数据报告（图像不入库，见下）
outputs/raw/*.jsonl      每样本原始输出（四配置逐样本配对）
outputs/tables/*.csv     汇总表（main/effects/…/baserate/crossing/related_work 等）
outputs/figures/         论文图 fig4–9 + figure1–3 + 数据拼图
v1-lvis-frequency/       第一阶段项目原样并入（LVIS v1 / COCO 2017）
run_manifest.json        环境、数据指纹、各阶段耗时
```

## 数据与图像

图像不入库：克隆后按以下方式再生（清单 `data/samples_manifest*.jsonl` 固定样本集合与
文件名）：

- food101：`./venv/bin/python code/prepare_data.py`
- stanford-dogs：`./venv/bin/python code/stage3_prepare_dogs.py`
- LVIS/COCO（v1）：见 `v1-lvis-frequency/code`

## 环境

本地 venv（`--system-site-packages`，Python 3.11 + transformers 5.17.0 + torch 2.9.0 +
timm），模型走本机 `/home/g203-4028/Models`（Qwen3.5-4B/9B、Qwen3-VL-4B、LLaVA-v1.6-
mistral-7B、InternVL3.5-4B、GLM-4.6V-Flash）。不调用外部闭源接口，不做任何参数更新；
全部缓存重定向到项目内 `cache/`。

## 主要结果速览

- 基率关系（food101，101 类别点/模型）：VCD r(ΔECE,acc)=−0.83（两模型，CI 排除 0），
  变号点 0.60/0.62 [0.56,0.66]；LCD 同向；MIB 在 4B 上方向不支持（准确率救回主导）。
- 非 Qwen 确认：LLaVA-v1.6 / InternVL3.5 低准确率组 ΔECE 全正、高准确率组全负、DiD
  全显著为正（`outputs/tables/nonqwen_core.csv`）。
- dogs 域全谱 [0,0.55] 落在有害侧：VCD 六格 ΔECE 全正；混合口径无一穿零。
- 输出侧信号（熵/maxp/空图 JSD）区分能力低于可用下限（AUC<0.70），门控路线在对象层面
  不成立。

完整论证与适用条件见 `STAGE5.md`；与最接近工作的逐条差别见
`outputs/tables/related_work.csv`。
