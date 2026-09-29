# 历史材料与恢复

活动目录使用稳定命名：`workflows/formal/`、`workflows/main_results/`、`data/responses/`、`data/annotations/`、`data/reference_gt/`、`outputs/annotations/main_results/`、`outputs/analysis/main_results/`和`paper/`。

## 服务器归档

归档根：`/home/g203-4028/projects/knowledge-deficit-mitigation-archive/20260929/`。

| 归档 | 内容 | 验证 |
|---|---|---|
| `historical_outputs.tar` | 13,045个历史输出/标注/执行记录，24,068,464,640字节 | 全部文件字节、SHA、符号链接和历史FIFO元数据逐项匹配 |
| `historical_outputs_manifest.json` | 每个源路径、大小、类型、SHA和恢复信息 | SHA256 `85840b89f415c326070da86b63ee4959ab3e5969529f77b67a56961b47343470` |
| `historical_code_documents.tar` | 134个退休路径中的517个文件，5,314,560字节 | 逐文件SHA验证；tar SHA256 `ba96c369ef777837c8623a3b5fa4b6f907840299cedbbef9c749069ab42fa583` |
| `historical_documents_20260929.tar.gz` | 43份阶段监控文档和5份论文阶段草稿 | 归档内各文件SHA与源文件逐项匹配 |
| `historical_support_materials.tar` | 6260个旧打包材料、阶段数据、执行记录和论文详纲文件，348,190,720字节 | 逐文件SHA验证；tar SHA256 `0cb3ee97965a8e378bf07e391fe3bd97e4710cd4ec75be8b189ec4947e5f392d` |
| `historical_admission_materials.tar` | 456个历史准入、执行验证及原任务材料文件，492,390,400字节 | 逐文件SHA验证；tar SHA256 `02cd9b615e42d0f6ac6526260dcdef7b6cddd54dd7f8e164df14bfe3a10f5bb5` |
| `pre_closeout/` | 替换前README和工作约定 | 原字节保存 |

历史输出tar的SHA256为`1be2bc3495964bf4d27c0d5adde497bd99c5bee4019808685e8e0b3c38e185df`。恢复时先在单独目录解包，再按manifest逐项校验。正式数据目录的manifest记录压缩文件SHA、原源SHA、行数和身份；原来的执行身份和作者记录可从历史包定位。

## 本地整理副本

`E:\OneDrive\文档\KDM-Archive\20260929\playground_staging`保存219个历史工作文件，共90,110,272字节；`playground_staging_sha256_manifest.json`记录其原路径与内容SHA。所有复制文件已经校验。本地批量删除步骤被自动审批拦截，原文件目前仍保留。服务器正式仓库的整理独立执行。

## 权重与原始数据

最终运行配置指向的16模型权重和原始图像保留。服务器共享模型目录维持原路径。归档覆盖被替换的实验阶段与执行记录，正式生成与标注由稳定路径和对应manifest访问。

## 6403与K100部署副本

6403旧部署已归档到相邻knowledge-deficit-mitigation-archive/20260929/historical_deployment.tar，19,275,120,640字节，SHA256 76304d62a9a2964d8c2994694b4d3e00b0553d89aaf015a4e12006f2bc7e294f；5786个当前文件重新校验后移除51个退休入口。K100对应归档12,250,767,360字节，SHA256 0adb9666197d6b676383348be73e325a84b4c3d99d6de05b67d4c57fad21424e；700个文件重新校验后移除11个退休入口。权重缓存、环境、当前图像与数据、Git历史继续保留。逐项退休回执在outputs/records/main_results/deployment_6403_retirement.json和deployment_k100_retirement.json。

## 最终退休与活动依赖

中央历史输出的204个入口已移出活动目录。运行配置继续使用`outputs/records/image_content_catalog.json`，该文件从校验档案恢复，SHA256为`4021e893d8f9b9c74d9d98470bda7a887ce922c4edf8d657e1bfd3d69aaf858d`。完整退休记录位于外部归档的`main_results_final_cleanup_receipt_final.json`。已闭合标注队列、重复版本、旧顶层运行清单、阶段打包材料与首次验证日志均保存到外部档案；当前评分与GT使用稳定入口及冻结的最终判断包。

最终验证包含353项CPU测试、五个模型的完整运行计划检查、95项来源证明、简单弃权控制的结果重放和差异检查。回执为`outputs/records/main_results/final_validation.json`。4028保存图像和正式运行环境；发布包包含最终代码、文档、原始回答、标注、GT、统计与论文，可用于CPU分析复现。6403、K100和本地工作目录接收相同发布包，保留各主机既有模型缓存与环境。

## 审阅材料整理

2026年9月29日审阅准备期间，中央服务器再次清理43个旧接收包、完成传输的暂存文件、临时检查脚本/图片及重复GT文件，并退休3份已释放的旧队列锁。6403清理3个重复GT文件，K100清理3个重复GT文件和6个编译冒烟文件，本地最终目录清理3个重复GT文件。合计61份文件副本、2,985,991字节，均先复制并校验后移出活动目录。

各主机相邻归档的`review_preparation/retired_files/`保存原字节；中央`review_preparation/retired_locks/`保存旧锁文件。汇总清单位于`outputs/records/review_package/cleanup.json`。文档修改前的副本存于`review_preparation/pre_edit/`。最终GT分布保留在`outputs/annotations/reference_gt/`根目录；回归测试、模型权重、原始数据和运行环境继续保留。

审阅压缩包由`docs/REVIEW.md`导航，包含论文、完整核心文档、条件/配对结果、图表、GT、判断记录及关键实现。每个文件在包内`MANIFEST.json`记录SHA256，大型运行资产由`LARGE_ASSETS.json`索引。
