# 数据与来源复核

执行者：Codex data_assets 代理。独立完整读取科研写作要求、论文主线、理论、冻结研究协议、标注评分、模型运行、来源和论文提纲后执行。本文记录软件与资产验收，不代表研究结果。

## Food-101 与别名

完整复用原始 `data/samples_manifest.jsonl` 的 4,848 条记录，原 group/eval 各 2,424 条；覆盖 101 类，4,848 个图像路径全部存在。映射输出为 `data/current/food101.jsonl`，group 映射 dev，eval 保持 eval，并额外保留 source_part、source_record。原始清单和图像未修改。

逐项读过 `configs/kdm/food_aliases.json` 的 101 类、147 个名称。保留提供的类别名、直接同义词、拼写变体及单复数，没有引入 WordNet、泛称或上位词。101 类与原清单一致，无重复别名或跨类同名；指纹及逐类审查见 `outputs/records/food_alias_review.json`。该记录是代理审查，协议要求的人工确认尚未获得，不将其写作人工批准。

## 接口验证专用清单

`data/current/interface16.jsonl` 固定原 Food 清单顺序中的前 16 个不同类别，每类第一张图。`outputs/records/interface16_selection.json` 保存来源清单指纹、选取规则和完整 ID。这 16 张只验证原生 generate 与后端的软件等价性，研究数据仍使用完整清单。

## 官方下载与故障

所有新资产位于 `cache/assets`，不覆盖旧 data 或 references。`scripts/fetch_assets.py` 按资产将成功 SHA256、字节数、时间，以及失败类型、错误全文持久化到带时间戳的 `outputs/records/assets_*.json`；单项失败后继续独立下载，最后非零退出，已有记录不会因后项错误丢失。

原 VQA raw.githubusercontent.com 地址在本机验证 TLS 时出现自签名证书链错误；失败记录保留。随后明确改为 GitHub 官方 API 的固定 Git blob `e4ff7887d53195f12856ab1e9087e69abe2e75c8`，继续启用 TLS 校验，解码后验证 blob SHA1 完全一致。原源码 SHA256 为 `f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8`。提取到 `src/kdm/models/official_vqa_normalizer.py` 的只有官方六项初始化字段及两个归一化方法；没有替换评分定义。

论文从 ACL Anthology 或 arXiv 官方地址下载，实际文件指纹在下载记录中。纯文本由 pdftotext -layout 从对应 PDF 生成，位置为 `cache/assets/references/papers/*.txt`。已取得 13 篇 PDF 并逐一成功转为非空纯文本，包含新增 DoLa 与 DeCo；每份 PDF/文本指纹、字节数见 `outputs/records/reference_asset_review.json`。获取和转换全文不等于已经通读或证实论文论断。

DeCo 官方无版本 URL 明确返回 v2；下载固定到同一 v2。首次并行分段因服务对不同 Range 返回不同 ETag 被拒绝，失败记录完整保留。第二次使用固定版本 URL，逐段核验响应文件名、Last-Modified、Content-Range 与字节数，再按顺序拼接 32,629,411 字节；全文 SHA256 为 `20ed2b353eeb09cac8fbd9bffb88cf4fcca59e0e82af8657f528697a540c185e`。两个版本探针的首 1024 字节一致，首段完整 4,078,677 字节在两个 URL 下也一致。原始尝试和恢复证据见 `outputs/records/DeCo*_download.json`。所有论文最终仍由统一下载器记录 SHA256，已有完整文件没有重新传输。

## VizWiz 与合并清单

官方验证标注已核验为 4,319 题/唯一图像，每题恰有 10 份答案；官方可回答 2,934、不可回答 1,385。所有官方答案及其 confidence、answer_type、answerable 和原始图像标识保留。分割规则固定为：原始图像文件名 UTF-8 字节的 SHA256 前八位按十六进制转整数，mod 5 为 0 属于 dev，其余 eval。与模型或干预结果无关。

图像下载、解压与全量存在检查已完成；`data/current/all.jsonl` 包含 9,167 个唯一 ID，缺图为 0。Food dev/eval 各 2,424；VizWiz dev 818、eval 3,501。逐项证据、原始来源及清单指纹见 `outputs/records/data_manifest_review.json`。生成操作要求 4,319 条，检查图像存在；合并保留全部源记录与分割，拒绝重复 ID。清单写入先完成验证再原子替换，失败不会截断既有清单。

## 验证

下载失败持久化/继续、官方归一化提取、完整清单、分割保留、缺图、重复 ID、异常时旧清单保留与 ZIP 路径穿越均有回归覆盖。测试临时路径限定项目 cache，最终命令和结果见 `outputs/records/data_asset_test_result.json`。
