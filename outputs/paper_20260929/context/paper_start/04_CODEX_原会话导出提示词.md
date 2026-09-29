# 交给原Codex会话：KDM论文数据导出

执行位置：**请在生成2026年9月29日KDM_Review成果包的原Codex会话执行。** 本任务继承已经完成的主体实验、最终评分与服务器资产位置。把本文件和 `KDM_Paper_Start_20260929.zip` 一起发入该会话。

以下为完整执行提示词。

---

你负责从KDM当前项目导出论文分析所需的已有数据。当前日期为2026年9月29日，论文目标为NAACL 2027十月ARR，内部提交日为2026年10月11日。研究主体已经完成；本任务围绕现有文件完成一次CPU数据导出，并返回可直接下载的压缩包。

先独立阅读随包 `context/用户科研写作要求_原文.md`，再阅读 `01_论文主线与结果决策.md`、本提示词与当前项目的最终协议、评分说明。优先使用你刚整理成果包时确认的最终资产。

## 一、固定工作范围

读取已经冻结的预测、标注、参考表、图片、配置、来源索引与运行日志。允许编写CPU脚本做格式转换、已有键连接、计数核对和文件复制，新增产出统一写入当前项目下的 `outputs/paper_20260929/`，导出脚本可放在 `scripts/paper_20260929/`。已有路径被占用时，另建带版本号子目录。

禁止重新推理、重新标注、重新提取隐藏状态或词表分布、重新运行模型普查、训练探针、修改提示词、修改评分语义、改变数据划分或方法参数。禁止启动GPU任务和模型服务。禁止重跑已有评分流水线；直接使用最终逐样本评分文件。

保持原始结果和现有标签不变。已有字段缺失时，在导出状态表中记载具体字段与来源，继续导出其余可用内容。禁止静默补值、自动生成新语义标签、用另一个实验替代当前条件。普通行数、键、计数与来源定位足以完成本任务；禁止新增非必要的SHA256或全仓库哈希流程。

本任务产出数据与交接说明，正文写作使用随包已经确定的主线。禁止把本次导出扩大为实验计划或项目重构。

## 二、已知最终来源

当前规范项目目录记录为：

`/home/g203-4028/projects/knowledge-deficit-mitigation`

先读取本地最终manifest、最终报告、`LARGE_ASSETS.json`和相关路径映射。当前项目位置有更新时，以已完成成果的真实路径为准，并记录映射。

重点资产：

- `outputs/annotations/main_results/score_rows.jsonl.gz`：853,248条最终逐样本评分，记录大小约99MB。
- `outputs/annotations/main_results/automatic_behavior.jsonl.gz`：已有逐样本行为标签。
- `outputs/analysis/main_results/controls/selections.jsonl.gz`：484,800条复制控制的既有来源选择。
- `outputs/analysis/main_results/condition_metrics.csv`：352个正式条件。
- `outputs/analysis/main_results/paired_comparisons.jsonl`：372项已有配对比较。
- `data/responses/formal/`：已完成的正式生成记录。
- `data/responses/independent/`：已完成的独立试答。
- `data/responses/candidate/`：已完成的类别排名记录。
- `data/images/`：已有真实输入图像。
- 已完成的 `uniform_reference_gt.jsonl`、历史接受参考表、模型登记、提示词配置、执行日志。

这次上传到ChatGPT的成果包包含汇总表和稿件，服务器大型资产在 `LARGE_ASSETS.json` 中另列。此次导出目标是让后续ChatGPT直接完成逐样本配对、弃权流向与真实案例排版。

## 三、论文已经固定的结果口径

正式主比较为Food-101，eval含101类×24张=2,424张。五模型使用真实检查点：Qwen2.5-VL-7B-Instruct、Qwen3.5-4B、LLaVA-v1.6-Mistral-7B、MiniCPM-V-2.6、Gemma-3-4B-it。352个条件合计853,248条回答。

主正确性字段为已经冻结的 `canonical_name_in_primary_score`；字面评分 `literal_extracted_name_score` 同时保留。弃权字段直接读取完整回复已有判断。

正文选择成果包已有的 **uniform reference**。其定义为正确类别排名>1且十次独立试答正确次数=0。保留既有uniform标签以及历史accepted标签，导出时按模型、样本连接。dev与eval参考记录合计24,240行，论文方法比较使用eval。

某模型和主表达的原有试答支持弃权集合为：该条件Direct已经弃权，且uniform标签=1。方法保留率的分母始终为这一固定集合。以既有数值核对连接：

| 条件 | Direct正确 | VCD正确 | IP-VCD正确 | 原有uniform集合 | VCD保留 | IP-VCD保留 |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA / UNKNOWN | 689 | 852 | 852 | 287 | 33 | 61 |
| MiniCPM / UNCLEAR | 733 | 873 | 903 | 374 | 239 | 249 |

MiniCPM/UNCLEAR的VCD相对Direct纠正集合为226个输入，IP-VCD继续正确212个。复制原有弃权且其余用VCD的控制正确722/2,424。历史accepted标签的LLaVA保留为35到62，MiniCPM为270/428到281/428。两套字段分别命名，明确保留。

## 四、输出1：紧凑逐样本表

将全部853,248条最终评分转换为列式或紧凑压缩表。推荐 `scores.parquet`；当前环境缺少相应库时使用 `scores.csv.gz`，并在schema说明类型。按行读取现有gzip，避免在内存中保存全部长回答文本。

每行必须能定位到唯一正式条件和样本。必需字段为：

`condition_id, sample_id, target_class, correct_canonical, correct_literal, abstain, uniform_reference, accepted_reference, source_file_id, source_line, qa_id, seed`。

这些列使用现有字段的直接映射；内部字符串键可用整数编码降低体积，编码字典一并导出。现有reference表的 `gold_rank`、`independent_correct_attempts`、`independent_attempts` 可放入独立的 `references.parquet` 或 `references.csv.gz`，以model和sample_id连接。完整参考表保持24,240行，并明确split。

同步导出：

1. `conditions.csv`，352行。至少包含模型、split、method、kind、main_marker、reference_marker、replicate及实际完整条件键、n、提示词ID、配置ID。条件ID根据完整已有键生成，防止同名方法在不同控制中混合。
2. `sources.csv`，每个源文件的真实路径、对应来源角色和文件ID。
3. `prompts_and_configs.json`，保存实际提示词与参数，可按已有键去重。先按已有正式生成记录或配置读取，同一条件的实际记录有差别时保存差别及来源。
4. `schema.md`，列出原字段到导出字段的映射、布尔值编码、空值语义、联合主键、样本与条件连接方法。

既有评分代码使用 `sample_id`、`target_class`、`source_path`、`source_line`、`qa_key`、`canonical_name_in_primary_score`、`literal_extracted_name_score`、`abstain`、`behavior`、`score_reason` 和条件字段。本次直接读取冻结文件的真实schema。保留行为类别、评分理由以及已有主类别提取字段，可放在单独的去重QA表以节省空间。

若现有行为标签包含泛称、具体错误等细类，直接保留其原值和标签来源；当前标签只区分回答与弃权时，原样保存，补充状态说明写明可用细度。禁止通过新的规则或模型给完整回复增加语义细类。

## 五、输出2：复制控制的紧凑来源选择

从既有 `controls/selections.jsonl.gz` 转换484,800条选择记录，保留每个control condition、sample_id及最终采用的正式response key或condition_id。提供200行控制条件映射；通过来源选择和正式评分表即可还原控制结果。

这一部分支持后续把Direct、VCD、IP-VCD和复制控制放在同一个准确率—弃权图中。逐行采用已经冻结的选择结果。

## 六、输出3：真实案例候选与图片

优先从LLaVA/UNKNOWN和MiniCPM/UNCLEAR选择12个左右的不同输入作为候选，总数至多16张。每个输入保存Direct、VCD、IP-VCD完整原回答，另附复制控制的来源回答。选样只根据已冻结字段，使用以下既有行为组合：

第一组：Direct弃权且uniform=1，VCD转为不正确具体回答或已有标签中的其他非弃权，IP-VCD保持弃权。优先导出6个以内，用于首页说明。

第二组：VCD把Direct错误或弃权转为正确，IP-VCD继续正确；优先4个以内，用于纠正收益示例。

第三组：IP-VCD相对VCD新增正确回答；优先4个以内，用于共同收益示例。

第四组：从完整矩阵选少量行为不同的补充案例，保留实际分类，服务附录。

每类可用数量按实际命中保存，语义字段沿用现有标签。完整案例文件 `cases.jsonl` 包含真实图像文件名、sample_id、target_class、模型、marker、condition_id、exact_question、各方法exact_answer、已有正确性和弃权标签、uniform/accepted标签、排名、独立正确次数、来源文件与行号。

图像从现有data/images直接复制到 `cases/images/`。为排版生成的缩放预览单独命名，保留原文件与源路径；记录尺寸。禁止生成替代图像、修饰物体内容或改写模型原回答。必要时通过已有路径索引逐个复制，禁止重新下载整套数据。

只扫描选中候选所需的正式回复源文件，按源文件分组，单次顺序读取目标行，减少重复解压。独立试答的完整十条回答只为案例候选导出；全部样本的正确次数与排名采用已有参考表。

## 七、输出4：已有执行成本和版本记录

从已有执行日志、manifest、trace中导出 `runtime_records.csv`。可用字段包括模型、方法、条件、硬件、batch size、记录的wall time、生成token数、prefill/decode时间、缓存方式和峰值显存。每一项附真实源路径与行号或JSON键。原始日志仅记录总体运行时间时，明确该口径；额外计算均从已知字段按公开公式派生。

实际可比较的同设备、同配置记录单独标记。当前有记录的条件计算流数量为VCD两路、IP-VCD三路；耗时倍率按真实日志计算。缺少时间记录的组合写入 `availability.csv`，继续交付数据。

另写 `version_receipt.md`：规范工作目录、实际本地HEAD与分支、远程当前master、成果包生成时间、当前已完成结果目录、未提交文件概况。本轮ChatGPT连接器读到远程master为 `1684809caf5593798cee049f13f0714931d7cf25`，提交日期为2026年9月24日。上传成果包日期为9月29日。请把本地完成状态与该远程状态的关系写清楚，保留所有现有成果。禁止reset、clean、强推及回滚。

如已有局部概率或共享前缀缓存，仅在 `availability.csv` 中列出可用路径、字段、覆盖范围与大小。此次导出保持紧凑，未来图表按实际可用缓存另作明确选择。

## 八、输出5：检查与交付

执行以下CPU检查并保存 `export_checks.json`：

- 正式评分恰好853,248行，352个唯一条件，每条件2,424个唯一eval样本。
- 每条件101类各24个输入；实际存在的split字段与eval一致。
- canonical、literal和abstain已有最终字段均为已决值；逐条件正确、错误、弃权计数对齐原condition_metrics。
- uniform与accepted按完整模型—样本键连接，记录连接成功数量；两套参考值各自保持原样。
- 用导出表复算上面的LLaVA与MiniCPM核心计数，并与本包40组配对的点估计对齐。
- 200个复制条件、484,800条来源选择均能连接回已完成正式记录。
- 每个真实案例的完整原文、图片和来源定位齐全。

发现计数差别时，写出具体条件、差值和来源，保留原数据；已正确导出的部分继续交付。统计检查记录实际执行内容，脚本核查按脚本核查命名。

最终返回：

`KDM_Paper_Data_Export_20260929.zip`

内容至少包含README、schema、conditions、sources、scores、references、control selections、真实cases与images、runtime_records、availability、version_receipt、export_checks和实际执行脚本。README说明已导出行数、总大小、字段缺口及下一步读取入口。

压缩后体积优先控制在30MB左右。需要拆分时使用独立可解压的 `KDM_Paper_Data_Export_Core_20260929.zip` 与 `KDM_Paper_Data_Export_Cases_20260929.zip`，核心表与图片分开；避免依赖专用软件的分卷格式。保留全部正式条件，使用压缩与去重控制大小。

最终回复直接给下载链接及实际完成概况。整个任务复用已完成的研究资产，主体模型与标注保持冻结。
