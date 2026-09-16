# 给 ZCode 的完整执行提示词

以下内容可直接用于`/goal`。目标是运行一次有边界的第九阶段，不在运行过程中增加研究问题。

## 任务与唯一目标

在KDM已有八阶段结果之上完成固定最小集合：四模型缓存上的配对可靠性分析；两模型同图像证据退化；VCD指定名称路径的连续强度判定及真实生成验证；小规模同图像POPE参照。最后给出中文结果总结、真实结果图、更新README，并细粒度提交和推送。

研究问题是概率集中、纠正和致错为何可以不同步。不要把低命名准确率定义为知识缺失，不拟合新方法，不把指定名称路径不可达解释为所有正确答案不可达，不把模拟图作为实验结果。

## 授权范围

工作目录固定为：

```text
/home/g203-4028/projects/knowledge-deficit-mitigation/
```

所有新增文件、缓存、下载、临时目录、日志与结果必须在这里。现有模型权重目录`/home/g203-4028/Models/`只读。物理GPU只允许0、1、4、5；代码把指定物理卡映射到进程内`cuda:0`，不要再把逻辑卡号混为物理卡号。不要终止不属于本任务的进程。

不改`stage6_engine.py`的科学定义。允许修复本包自身的确定性代码问题，但任何影响输入、评分、选择规则、候选约束、停止词或统计口径的变化必须先写清原因并提交。缺少数据或模型时显式停止相关任务，不换模型、跳过类别或生成替代标签。

## 安装与源记录

将ZIP解压到项目内`deliverables/KDM_next_stage/`。先阅读本包00、01、04、09、10文档。检查`git status`，保留已有用户改动；记录当前分支和提交，与本包源提交`e5eda213dd3f51c08b43c1616305c93b84011dff`比较。

将本包七个`code/stage9_*.py`复制到项目`code/`，将`code/viz/stage9_figures.py`复制到项目`code/viz/`。文档复制到`docs/stage9/`。若已有同名第九阶段文件且内容不同，先比较，不覆盖未知用户工作。包内测试与样例保留在`deliverables/KDM_next_stage/`。

设置环境（以下所有路径均位于项目内）：

```bash
cd /home/g203-4028/projects/knowledge-deficit-mitigation
export HF_HOME="$PWD/cache/hf"
export TORCH_HOME="$PWD/cache/torch"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export MPLCONFIGDIR="$PWD/cache/mpl"
export NLTK_DATA="$PWD/cache/nltk"
export HF_HUB_DISABLE_XET=1
export TMPDIR="$PWD/cache/tmp"
export PIP_CACHE_DIR="$PWD/cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" outputs/logs/stage9 outputs/figures/stage9 docs/stage9
```

沿用`./venv/bin/python`，不要升级torch/transformers主版本。测试额外依赖见`requirements-diagnostics.txt`；只在缺失时安装对应轻量依赖。联网沿用服务器已有代理，不把代理密码或令牌写进日志。

把04文档作为`docs/stage9/PREREGISTER_STAGE9.md`提交，另提交安装代码及源文件指纹。**预登记必须在新的真实模型推理前完成。** Git提交粒度是协议、计算代码、运行结果、图与报告，不要求每几十行提交一次。

## 本地代码测试与清单固定

先使用项目venv运行交付包测试，临时文件明确写入项目：

```bash
./venv/bin/python -m pytest deliverables/KDM_next_stage/tests -q \
  --basetemp="$PWD/cache/stage9_pytest" \
  > outputs/logs/stage9/package_tests.txt 2>&1
./venv/bin/python code/stage9_prepare.py --per-class 4 --geometry-per-class 2
./venv/bin/python code/stage9_pope.py prepare
```

`stage9_prepare.py`应得到400条模型—图像记录；路径分析标记200条模型—图像记录。POPE应得到同50张图像、两个子集、合计600条问题记录。运行后打印每模型/子集/条件的实际数量并提交清单。POPE只下载必要图像；标注指纹不符时停止，不自动接受新版本。

本包使用的真实样例数据是选择列转录。服务器复核高准确率子集下降组合数量、实际准确率端点及名称层最小值；若与包中固定提交不同，在`SOURCE_DIFFERENCES.md`说明版本差异，不把不一致藏在四舍五入里。

## 四模型缓存分析

```bash
./venv/bin/python code/stage9_analyze.py --replicates 2000 \
  > outputs/logs/stage9/cache_analysis.txt 2>&1
```

必须使用真实第六阶段JSONL，不使用`data_sample/synthetic_*`。完整配对失败即停止并定位，不用缺失值填零。保留Qwen3.5-4B原弃权口径。主风险比较是序列概率排序、80%共同可回答集合覆盖率；报告所有输入口径的实际覆盖率。

## 真实模型检查与固定推理

先在Qwen3.5-9B和LLaVA-v1.6-7B分别运行两个样本，确认原处理器、输入、参考分支和自由生成正常。对这两个原图样本真正执行直接解码，与相应第六阶段缓存的token序列比较；记录差异而不是只比较重新计算的分数。`stage9_run.py`已在每分片前两张图像上直接执行这一检查并记录数量，不需要代理另写检查算法。此检查属于实现一致性，不选择有利样本。

`stage9_run.py --limit 2`输出仍写入最终分片文件，定义与正式运行相同，之后会按键复用。两个样本所需时间只用于估算固定任务耗时，不用于决定新的实验强度或选择数据。

正式运行采用四进程、互斥分片：

```bash
./venv/bin/python code/stage9_run.py --model q9b --gpu 0 --shard 0 --shards 2 >outputs/logs/stage9/q9b_0.txt 2>&1 &
p0=$!
./venv/bin/python code/stage9_run.py --model q9b --gpu 4 --shard 1 --shards 2 >outputs/logs/stage9/q9b_1.txt 2>&1 &
p1=$!
./venv/bin/python code/stage9_run.py --model llava16 --gpu 1 --shard 0 --shards 2 >outputs/logs/stage9/llava_0.txt 2>&1 &
p2=$!
./venv/bin/python code/stage9_run.py --model llava16 --gpu 5 --shard 1 --shards 2 >outputs/logs/stage9/llava_1.txt 2>&1 &
p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
[ "$status" -eq 0 ] || exit 1
./venv/bin/python code/stage9_finalize.py --replicates 2000
```

运行器沿用官方VCD/M3ID；原图结果复用缓存。名称路径读取完整词表，既不写大型词表数组，也不一次性加载历史NPZ。见证强度只用于验证数学判定，不加入方法性能表。任何见证不匹配明确记录目标序列、实际序列、强度和数值信息，停止受影响工作；不能调整目标拼写使其通过。

然后运行POPE；复用同一GPU分配与图像分片：

```bash
./venv/bin/python code/stage9_pope.py run --model q9b --gpu 0 --shard 0 --shards 2 >outputs/logs/stage9/pope_q9b_0.txt 2>&1 &
p0=$!
./venv/bin/python code/stage9_pope.py run --model q9b --gpu 4 --shard 1 --shards 2 >outputs/logs/stage9/pope_q9b_1.txt 2>&1 &
p1=$!
./venv/bin/python code/stage9_pope.py run --model llava16 --gpu 1 --shard 0 --shards 2 >outputs/logs/stage9/pope_llava_0.txt 2>&1 &
p2=$!
./venv/bin/python code/stage9_pope.py run --model llava16 --gpu 5 --shard 1 --shards 2 >outputs/logs/stage9/pope_llava_1.txt 2>&1 &
p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
[ "$status" -eq 0 ] || exit 1
./venv/bin/python code/stage9_pope.py summarize
```

如果卡被其他任务占用，不转用2或3。固定最小集合不增加新模型或新数据集。一天预算内不能完成时给出真实完成比例、阻塞原因与剩余任务，不输出“全部完成”。不因统计效应不显著增加更多样本。

## 真实结果图与结果判断

以下调用使用真实输出，不带`--fixture`：

```bash
./venv/bin/python code/viz/stage9_figures.py \
  --package deliverables/KDM_next_stage \
  --risk-csv outputs/tables/stage9/risk_coverage.csv \
  --evidence-csv outputs/tables/stage9/evidence_accuracy.csv \
  --reachability-csv outputs/tables/stage9/name_reachability.csv
```

该命令把新图写入交付包的`figures/real/`。复制其中真实新结果到`outputs/figures/stage9/`，PNG/PDF均保留；前三张固定提交图如遇源版本变动需依据真实最新表更新，不能继续标为最新结果。数学示例只用于概念解释，合成测试示例不复制到正式结果目录。

检查实际图像：坐标、数据量、图例、长标签、字体、范围和来源标记；渲染PDF核对。若只修改布局不改数据，可重绘。任何新的绘图函数都先用实际CSV运行后才提交，不能只通过语法检查。

依据04文档的预登记判据写`docs/stage9/RESULTS.md`。需要分别回答：纠正与致错的真实数量；固定覆盖率风险是否增加；视觉退化是否有效；指定路径是否出现有意义的可达性分化；数学见证是否与实际生成一致；POPE与命名结果在什么条件下相同或不同。不要按照预期方向补写未出现的趋势。

## 提交与最终输出

所有git提交由一个协调进程串行操作，GPU工作进程不自行提交。每次只添加明确文件，不用`git add .`；不提交图像数据、模型、HF缓存、NPZ、字体或合成测试项目全量目录。不重写历史、不force push。用户已授权本轮代码/结果提交和推送；保持实际工作分支并在报告中说明，推送失败如实报告，不声称远程已更新。

建议提交顺序：预登记与设计；分析/运行代码；固定样本清单；真实原始标量结果与汇总；正式图；README及中文结论。使用`README_DRAFT.md`更新旧README时保留已经完成的结果和明确的第九阶段实际状态，不把计划写为完成。

最终提供中文总结：一句话故事、2—3条实际贡献、关键图、每个判据的结果、尚未完成的具体任务（若有）、GPU时间/峰值显存/磁盘增量、最后提交及推送状态。不要输出英文论文全文，不启动第十阶段。
