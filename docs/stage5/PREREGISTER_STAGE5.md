# 第五阶段预登记（收束轮）

写入时间：2026-09-14，在任何中间准确率类别推理之前。此后不再修改本文件的判定条款；
如需补充，只允许在文末"补记"一节添加运行前的实现说明。

## 0. 本轮问题

主线：解码期免训练幻觉抑制方法只重新分配置信度、不改变模型知道什么，因此其对校准
的影响方向由评估人群的基率决定。可检验形式：

> 人群 P 上方法 m 的校准误差变化 ΔECE(P,m)，可由该人群在干预前（direct 配置）的
> 状态预测：与"干预前校准差"（平均置信度 − 准确率）正相关；与干预前准确率负相关。
> 两者之间存在变号点：人群准确率低于该点时干预损害校准，高于该点时改善校准。

## 1. 数据与人群

### 1.1 记录来源

- 复用：`outputs/raw/{q4b,q9b}_main_naming.jsonl`（food101，deficient/known 两层，
  direct/vcd/mib/lcd，各 600 样本/层）；`outputs/raw/{llava16,internvl4b}_food101_main.jsonl`
  （同结构）；若 dogs 中间类别运行完成，另加 `outputs/raw/{q4b,q9b}_dogs_main.jsonl`
  与新的中间类别记录。
- 新增：`outputs/raw/{q4b,q9b}_s5middle_naming.jsonl`——food101 上既不在 deficient
  也不在 known 的类别（按各模型自己的 `data/strata_{model}.json` 计算，101−50=51 类，
  每类 eval 半 24 张，共 1224 样本 × 4 配置）。dogs 域同理（120−60=60 类，1440 样本，
  文件名 `*_s5middle_dogs.jsonl`）。
- 新增运行的代码路径：`code/stage5_middle.py`，逐字复用 `code/run_experiment.py` 的 naming
  循环与 `code/engine.py`/`code/prompts.py`/`code/scoring.py`，仅把 stratum 记为 `middle` 并替换类别
  集合。超参、提示、判分、随机过程与主实验完全一致，不做任何调整。

### 1.2 人群的两个口径

- **口径一（类别集合自然构成）**：人群 = 单个类别（每类 n=24，food101 全部 101 类；
  dogs 全部 120 类或仅 deficient∪known 的 60 类，取决于中间类别是否运行完成）。
  另报告三个聚合点（deficient / middle / known 三层）与图示分箱（见 §4）。
- **口径二（样本混合构造）**：对每个（模型, 域），以混合比
  α ∈ {0.0, 0.1, …, 1.0} 从 known 层 eval 样本抽 α·600 张、从 deficient 层抽
  (1−α)·600 张构成混合人群；每个 α 抽 200 次，各量取 200 次的均值。此口径检验
  关系是否依赖人群的人为混合。

## 2. 测量定义（在任何新数据产生前固定）

- **干预前准确率** acc(P) = P 内 direct 记录 outcome==correct 的比例。
- **干预前校准差** gap(P) = mean(direct 的 answer_maxp) − acc(P)。
- **校准误差** ECE = 10 个等宽分箱的加权 |mean_conf − mean_acc|。
- **ΔECE(P,m) = ECE_m(P) − ECE_direct(P)**，m ∈ {vcd, mib, lcd}，同一样本集合配对计算。
- 样本集合内各方法使用同一批文件；仅排除无 answer_maxp 的记录（roundA）。
  弃权记录按其 outcome 原样保留（第二阶段已证弃权率处于地板，不构成可用量）。
- answer_maxp 语义沿用第二阶段：生成首 token 在该配置实际使用的分布下的概率。

## 3. 预登记的判定条款

### 3.1 方向（主判定）

对每个（模型, 域, 方法）在口径一的类别点上计算：

- r_gap = Pearson(ΔECE_c, gap_c)（c 为类别），预登记预测 **r_gap > 0**；
- r_acc = Pearson(ΔECE_c, acc_c)，预登记预测 **r_acc < 0**。

95% 置信区间由类别簇自助（B=2000，对类别重采样）得到。方向成立 = 区间排除 0
且符号与预测一致。两个预测变量分别报告相关系数、R² 与区间，并直接比较 |
r| 大小以说明哪一个预测更准；不对其差做显著性检验。

### 3.2 变号点

对每个（模型, 域, 方法）：类别点上 OLS 拟合 ΔECE = a + b·acc，变号点 = −a/b
（仅当 b<0 时报告）。区间由同一类别簇自助（B=2000）取百分位。另以 gap 为自变量
同法计算。主要报告 food101 两模型；dogs 视中间类别完成情况。

### 3.3 跨模型一致

3.1 的符号须在 q4b 与 q9b 上一致；dogs 域同查（若 dogs 中间类别未运行，dogs 用
deficient∪known 的 60 类类别点检验，并如实注明）。不一致的（模型, 域, 方法）逐项列出。

### 3.4 非 Qwen 确认（独立判定，不影响 3.1–3.3）

用既有 `llava16`/`internvl4b` food101 主实验记录计算：低准确率组（原 deficient 层）
各方法 ΔECE 的符号，与差中差 DiD(m) = ΔECE(低准确率组, m) − ΔECE(高准确率组, m)
的符号。与 Qwen 系方向一致 = 通过。两项模型分别报告。

## 4. 图示（fig9）

口径一：类别按 pre-acc 排序后等量分 8 箱（每（模型,域）各自分箱），箱内计算
acc、gap、各方法 ΔECE，绘 ΔECE–acc 与 ΔECE–gap 曲线并叠加 OLS 变号点。
口径二：11 个混合点的 ΔECE–acc 曲线。分箱仅用于展示，推断全部基于类别点。

## 5. 交付物

```
outputs/tables/baserate.csv     口径一/口径二各人群的 acc、gap、ΔECE（含三层聚合点）
outputs/tables/crossing.csv     变号点及自助区间（按自变量 acc 与 gap 两个口径）
outputs/tables/nonqwen_core.csv LLaVA/InternVL 主线量（含 DiD）
outputs/figures/fig9.pdf
```

## 6. 纪律

- 若方向与预登记相反：不得更换人群定义、调整分箱、改用其他校准度量或剔除类别
  以扭转方向。直接报告实际方向，并按实际方向改写主线表述。
- 既有记录不做任何改写；新增记录只来自中间类别。
- 缺模型/数据/依赖时按惯例写明缺什么、尝试过什么、为何无法继续，停止相关部分并上报。
- GPU 仅用 4、5；缓存与输出只写入项目目录。

## 补记（运行前实现说明）

- 中间类别清单在运行前由 `data/strata_{q4b,q9b}.json`（food101）与
  `data/strata_{q4b,q9b}_dogs.json`（dogs）计算：全类集合 − deficient − known。
  q4b 与 q9b 的中间类别集合不同（各自按自己的分组半准确率分层），这是设计使然。
- food101 中间运行由 `code/stage5_middle.py` 驱动（复用第二阶段 engine 路径）；
  dogs 中间运行复用第三阶段 `code/stage3_model.py` 路径，仅替换类别集合为中间类别、
  stratum 记为 middle。两者均不在本轮改变任何超参、提示或判分。
