# CodeX执行任务书：KDM弃权保持研究

你在现有KDM项目中执行已经确定的研究设计。目标是完成代码重构、正式实验、核查、结果图和论文资料。研究主线、方法公式、模型选择依据与评价分母已经确定，不能根据期望结果重新设计。

## 读取文件

每个执行代理独立读取用户原始文件 `source_materials/RESEARCH_WRITING_REQUIREMENTS.md`。安装后路径为 `source_materials/kdm/RESEARCH_WRITING_REQUIREMENTS.md`。然后完整读取PAPER_STORY、THEORY、STUDY_SPEC、ANNOTATION_AND_SCORING、MODELS_AND_RUNTIME、SOURCES及paper/OUTLINE。

主问题是：带弃权引导时，对比计算是否把有根据的不确定表达与需要抑制的内容一起降低；同义表达如何影响这个变化；指令保持解码能否兼顾回答与弃权。

## 写入与设备范围

唯一写入根目录：`/home/g203-4028/projects/knowledge-deficit-mitigation/`。
允许读取已有模型根目录、mprisk代码和对应conda环境。用户已有模型优先复用。新环境如确有必要，使用项目目录内的环境前缀。缓存、下载、日志、临时文件、论文与图片全部在项目目录内。

只允许物理GPU 0、1、4、5。使用 `scripts/worker.sh` 获取设备文件锁；多卡模型同时锁定所使用的两张卡。禁止CPU和磁盘权重卸载，禁止占用其他显卡。12B/13B模型使用两张卡并记录显式映射。每个进程使用它自己的模型spec和环境Python，不修改共享环境的库版本。

禁止覆盖旧原始记录、删除Git历史、强制推送、隐式替换模型、自动跳过错误样本、把未完成的任务统计为完成。真实输出与测试数据分目录保存。

## 研究条件

弃权引导是主提示。UNKNOWN、UNCLEAR、UNSURE、I cannot identify it四种表达完整交叉。无引导提示保留为对照。候选模型列表固定为configs/models.json中的16个；全部候选在完整数据清单上执行直接解码。选择依据为原始语义弃权是否存在。所有选择在读取干预结果之前完成并提交。零弃权条件存入附录清单；选中模型之后不根据效果删除。

不要把二选一问题的No计为弃权。不要把“可能是牛奶”计为完整弃权。上位类别回复保持“给出答案”的语义标签，正确性另行评定。每个输出都获得完整语义标签。

## 仓库迁移

核对远程与本地提交。当前已核查源提交为eceed2246515cb938ad478ee23c94590c990eb5d。保留之后已经完成且相关的工作；如果实际HEAD不同，查看具体差异，不执行强制回退。

先把开发包放入项目的 `deliverables/KDM_research_refactor/`，为包的来源与用户要求创建提交，确保工作树干净。执行：

```bash
PROJECT=/home/g203-4028/projects/knowledge-deficit-mitigation
PACKAGE="$PROJECT/deliverables/KDM_research_refactor"
cd "$PROJECT"
python "$PACKAGE/scripts/install.py" --project "$PROJECT" --package "$PACKAGE"
python "$PACKAGE/scripts/install.py" --project "$PROJECT" --package "$PACKAGE" --apply
PYTHONPATH=src python -m pytest -q
```

迁移后旧code目录及旧README进入archive，原data与outputs保持原样，阶段文档保持历史身份。活动代码使用src/kdm。安装器提取模型构造代码，保留原始源码blob校验。遇到源版本差异，需要记录审阅结论才允许使用 `--reviewed-source-change`。

包内所有目录冲突在移动文件前检查。已有未归档src或scripts时不要直接删除，记录其内容与用途后进行有清单的迁移。最终活动代码不得继续依赖stage1至stage9的实验入口。

## 模型与资源登记

调用 `scripts/discover_models.py` 在用户给定模型根目录发现检查点，生成 `configs/runtime/model_inventory.json`。目录缺失或多个候选均存在时明确解决；不要换成规模或骨干不同的同名模型。

为16个候选填写环境Python、模型路径、处理器、思考模式、dtype及权重文件指纹。已核查mprisk登记有13个视觉文本模型，优先读取相应包装。MiniCPM、Phi等远程结构使用已有prepare/forward实现。对比后端按以下合同实现：

- `encode(text)`：不自动添加特殊词元。
- `decode(tokens)`：只移除模型声明的特殊词元，保留原始tokens。
- `session(image,prompt,reference,seed,need_layers)`：创建独立条件会话。
- `next(prefix)`：在完整相同的已生成前缀上返回下一词元logits；层方法额外返回规定层输出。

MiniCPM的输入字典、图像切块和视觉词元映射使用原有模型包装，不能改成普通图像张量后声称等价。Gemma、Qwen、InternVL的模板与位置状态按各模型实际实现处理。新增适配代码单独提交。

本地vLLM只承担直接解码和独立试答等它支持的普通推理。机制实验必须访问实际条件分数。普查后端与实验后端使用同一检查点和处理器配置，并在固定16条接口检查图像上比较原生贪心generate与新后端完整tokens。这16条只承担软件接口核验，全部研究样本继续完整执行。

```bash
python scripts/fetch_assets.py --root "$PROJECT" --manifest configs/kdm/assets.json --group papers
python scripts/fetch_assets.py --root "$PROJECT" --manifest configs/kdm/assets.json --group vizwiz
python scripts/fetch_assets.py --root "$PROJECT" --manifest configs/kdm/assets.json --group vqa_code
```

所有实际下载记录SHA256；下载故障写明具体文件与错误。阅读已下载论文中的对应公式与源码，完成DEVIATIONS记录。禁止直接用历史文档中未经修正的说法代替正式方法定义。

## 数据与判分冻结

Food-101使用原samples_manifest全部样本，按原group/eval分割。VizWiz验证集全部4319条使用确定的图像分割。将两份转换后的清单按ID合并，检查重复ID、文件存在和总数，并记录各部分数量。

```bash
PYTHONPATH=src python -m kdm.cli --root "$PROJECT" prepare-food \
  --source data/samples_manifest.jsonl --out data/current/food101.jsonl
PYTHONPATH=src python -m kdm.cli --root "$PROJECT" prepare-vizwiz \
  --annotations cache/assets/vizwiz/annotations/val.json \
  --images cache/assets/vizwiz/images/val \
  --out data/current/vizwiz.jsonl
```

具体解压子目录以官方压缩包内路径为准，修正调用路径并写入清单，不移动模型或修改任务。确认Food-101的101类别名集合，保留正确名称、同义写法与单复数，禁止WordNet自动扩展。正文设置一句话，详细核验保存在工程记录。

复制STUDY_SPEC为 `docs/current/PREREGISTER.md`，补充实际文件指纹、模型环境与下载版本，提交并推送。科学条件使用本文固定内容，不能填写尚未发生的结果。

## 完整模型普查

每个模型使用全量合并清单运行census。任务输出中同时保存带引导与无引导回答。记录真实tokens、完整文本、终止状态和推理配置。使用对应环境Python调用：

```bash
bash scripts/worker.sh "$PROJECT" 0 "$MODEL_PYTHON" -m kdm.cli \
  --root "$PROJECT" run --mode census --manifest data/current/all.jsonl \
  --model "$MODEL_KEY" --model-spec "configs/runtime/$MODEL_KEY.json" \
  --gpu 0 --out "outputs/raw/current/$MODEL_KEY/census.jsonl"
```

可以按相同任务身份在允许GPU上分片；最终合并必须验证每条任务ID唯一。模型普通生成接口在 `models/api.py`，其本地服务端口必须固定记录。

使用 `annotation-queue` 生成全量标注队列，使用 `scripts/annotate_responses.py` 调用独立本地语义检查模型。标注器看不到生成模型和方法。完整短语自动标签和其他语义标签都进入同一校验文件。全部弃权变化和歧义标签进行人工复核；不以少量标签估计整体弃权数。

`select`按原始弃权选择研究模型，并提交筛选记录。模型因任务不同产生弃权差异时，记录对应模型—任务条件。不得根据VCD是否有害选择模型。

## 正式行为与方法实验

根据选择记录为每个模型生成正式清单，只包含selected=true的数据任务，并保留这些任务中的全部既定评测样本。选中模型使用对应正式清单执行experiment任务。它包含四种引导、VCD/M3ID的4×4参考提示组合、DoLa/DeCo、参考删除引导对照、两个指令保持实现与CDA视觉迁移。

```bash
bash scripts/worker.sh "$PROJECT" 1 "$MODEL_PYTHON" -m kdm.cli \
  --root "$PROJECT" run --mode experiment --manifest data/current/all.jsonl \
  --model "$MODEL_KEY" --model-spec "configs/runtime/$MODEL_KEY.json" \
  --gpu 1 --out "outputs/raw/current/$MODEL_KEY/experiment_0.jsonl" \
  --shard 0 --n-shards 1
```

支持SID的入选模型加入 `--methods vcd,m3id,dola,deco,sid`，保持原始参考实现、视觉词元选择和因果掩码。若当前架构无法定义所需注意力，报告具体结构原因，其他方法继续保留完整数据；不伪造一个同名近似SID。

指令保持方法：带引导清晰分数加无引导清晰/参考的对数概率差，alpha1。CDA视觉迁移使用正文无动量版本，保留两个空输入校准会话。直接保留原始弃权的对照由完整原回答构造，不能改写为仅恢复一个词元。

每次正式生成结束都运行 `scripts/verify_complete.py`，缺失或重复ID需要解决。对于分片结果，只有全部预定义分片合并后才进行完成判定。按原始输入、方法和种子重试失败任务，已完成记录不重复推理。

## 独立作答与闭集测量

对选中模型所有评测题运行probe，每题10次独立试答，temperature1、top_p1。使用无弃权引导的最佳估计问题，不附加知识或答案。全部正确次数保留；稳定重复错误仍计为错误。

```bash
bash scripts/worker.sh "$PROJECT" 4 "$MODEL_PYTHON" -m kdm.cli \
  --root "$PROJECT" run --mode probe --manifest data/current/all.jsonl \
  --model "$MODEL_KEY" --model-spec "configs/runtime/$MODEL_KEY.json" \
  --gpu 4 --out "outputs/raw/current/$MODEL_KEY/probe.jsonl"
bash scripts/worker.sh "$PROJECT" 5 "$MODEL_PYTHON" -m kdm.cli \
  --root "$PROJECT" closed-probe --manifest data/current/food101.jsonl \
  --model "$MODEL_KEY" --model-spec "configs/runtime/$MODEL_KEY.json" \
  --gpu 5 --out "outputs/raw/current/$MODEL_KEY/closed.jsonl"
```

闭集评分保存全部101类的总对数概率、长度与平均值。可以实现不改变数学结果的缓存复用和批量前向，用相同输入逐项核验后启用。不得变更候选数量、排名公式或样本集合以减少任务。

VizWiz人工不可回答标签与模型重复试答分别使用。没有可靠具体答案的题目不通过全部判错来制造模型知识缺失标签。

## 机制与表达边界

`replay`对VCD、M3ID、DoLa、DeCo和适用的SID保存实际参考分布与组概率关系；`mechanism`在实际直接回复路径上保存四条件分数。每一步条件拥有独立KV缓存、位置偏移和相同输出前缀。对VCD记录完整参考提示组合；M3ID记录激活状态和实际权重。

```bash
bash scripts/worker.sh "$PROJECT" 0 "$MODEL_PYTHON" -m kdm.cli \
  --root "$PROJECT" mechanism --records "outputs/raw/current/$MODEL_KEY/experiment_0.jsonl" \
  --model "$MODEL_KEY" --model-spec "configs/runtime/$MODEL_KEY.json" \
  --gpu 0 --out "outputs/raw/current/$MODEL_KEY/mechanism_0.jsonl"
```

使用 `scripts/complete_response_audit.py` 对全部固定直接回复与登记弃权表达进行完整序列评分。供体使用实际生成tokens，包含模型自己的空格与终止符。候选回复缺少具体回答时显式记录该测量条件，没有建立有限二组比较就不输出虚构概率。

自由生成结果使用全句语义标签，UNKNOWN变成UNCLEAR仍计为弃权。词元分组只用于局部解释，完整序列测量保留每一步归一化。全文关于“同义表达是否同时受到抑制”的结论来自这三份相互关联的记录。

## 汇总与检查

把所有正式回答与probe记录完成标注后，运行 `scripts/build_reports.py`，得到paired、probe、validity、method四组表。程序需要official_vqa_normalizer.py和全部闭集记录。保留每组样本实际分母。

数学测试、逐条件概率差、原生generate等价检查、计数完整性及图表数值分别核查。可使用独立代理完成数值核查和文章结构审查；记录代理真正做过的操作。若运行环境没有子代理工具，使用独立脚本和明确的人工检查记录，不宣称已经使用独立代理。

相对原VCD/M3ID的正确率非劣界固定为−0.01；合理弃权保留率改善及具体纠正保留率完整报告。任何结果都按照已固定指标陈述，不为通过判据更换分母或筛除模型。

## 论文与交付

按照paper/OUTLINE组织中文论文详纲、每段结论与真实图表。相关工作采用SOURCES中正式原文；保留CDA的完整结论及DExperts的公式家族归属。不要把已发表工作缩小成便于比较的版本。

正文研究设定用一句话说明正确性定义，完整实现信息进入附录。主表只呈现有弃权的研究条件；完整模型筛选记录进入附录。历史实验不按阶段复述。

所有图使用真实汇总表重新运行。示例图位于开发包，仅用于程序与版式检查；含数学或合成标记的文件不能作为正式实验图复制。保存绘图输入文件与SHA256、PNG、PDF和渲染检查记录。

最终提交：活动代码、数学说明、冻结协议、全部小型原始JSONL、数据清单、模型/环境身份、完整表、真实图、RESULTS、DEVIATIONS、完整论文详纲。模型权重、图像、巨大分布与缓存不入Git。原有历史保留。细粒度提交后执行普通push，并报告实际远程提交。

任务完成的依据为全部预定义任务已执行、真实结果与代码可追溯、论文表述与证据一致。目标方向的支持程度由得到的结果决定。
