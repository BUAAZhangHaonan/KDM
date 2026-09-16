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
| 一（LVIS/COCO） | 频率分层：低频对象上抑制方法是否有害 | `CONCLUSIONS_STAGE1.md`（附 `RECON/VALIDATION/BLOCKING_NOTE_*_STAGE1.md`） |
| 二 | food101 实测分层（低/高准确率组），错误置信度为主结局 | `PILOT.md` `CONCLUSIONS.md` |
| 三 | 开集失败两因分解（可及未输出 vs 不可及），跨模型跨域 | `STAGE3.md` |
| 四 | 知识可及性测量的边界：名称受限 LL 探针门槛全败、粒度错配 | `STAGE4.md` |
| 五 | 收束：基率预测律、变号点、与最接近工作的对照 | `STAGE5.md` |
| 六 | 基线忠实化：VCD/M3ID/DoLa/DeCo 按官方实现重跑 + 尖化对照/四格分解/常数偏移校正（SID 受阻，见报告 §2.2） | `STAGE6.md` `IMPLEMENTATION_MAPPING_STAGE6.md` |
| 七（封版） | 收尾：分格尖化一致性定措辞分支（强版）、SID 有界复现（llava16）、dogs 忠实重跑 16/16 方向命中、证据—主张映射与主线定稿 | `STAGE7.md` `MAINLINE_STAGE7.md` `EVIDENCE_MAP_STAGE7.md` |
| 八（终局封版） | 分支判定重做：逐样本残差回归 γ CI 含零（六口径）→ 弱版；等价检验 4/23；dogs 改数量级一致性检验（14/16）；M3ID 强化措辞；git 历史核查零残留 | `STAGE8.md` `MAINLINE_STAGE8.md` `CLAIMS_STAGE8.md` `LIMITATIONS_STAGE8.md` `EVIDENCE_MAP_STAGE8.md` `GITCHECK_STAGE8.md` |

每个阶段的判定阈值在运行前写入 `PREREGISTER_STAGE3/4/5/6/7/8.md` 并先于数据提交。
第一阶段（LVIS/COCO）的产物并入主树：文档带 `_STAGE1` 后缀，实验产物带 `lvis_`
前缀（如 `outputs/raw/lvis_q4b_main_naming.jsonl`、`outputs/figures/lvis_figure1.pdf`），
其 LVIS 管线代码为 `code/stage1_*.py`；原始迭代提交历史完整保留
（`git log -- code/stage1_engine.py` 可用 `--follow` 追溯）。

## 目录结构

```
code/                    全部实验与分析代码（v2+ 入口 run_all.py；阶段脚本 stageN_*.py；
                         第一阶段 LVIS 管线 stage1_*.py）
data/                    样本清单、分层定义、近邻表、数据报告（含 lvis_* 清单；图像不入库）
outputs/raw/*.jsonl      每样本原始输出（四配置逐样本配对；lvis_* 为第一阶段）
outputs/tables/*.csv     汇总表（main/effects/…/baserate/crossing/related_work；lvis_* 为第一阶段）
outputs/figures/         论文图 fig4–9 + figure1–3 + lvis_figure1–3 + 数据拼图
run_manifest.json        v2 起环境、数据指纹、各阶段耗时（第一阶段为 run_manifest_stage1.json）
```

## 数据与图像

图像不入库：克隆后按以下方式再生（清单固定样本集合与文件名）：

- food101：`./venv/bin/python code/prepare_data.py`
- stanford-dogs：`./venv/bin/python code/stage3_prepare_dogs.py`
- LVIS/COCO（第一阶段）：`./venv/bin/python code/stage1_build_dataset.py`（清单 `data/lvis_dataset.jsonl`）

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
