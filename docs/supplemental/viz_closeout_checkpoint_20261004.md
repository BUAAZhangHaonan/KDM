# Viz 收尾检查点：2026-10-04 02:08 北京时间

DoLa/SID 的 1,515 条有限补测、CDA 的 108 路径/803 位置和定点重放核对已完成。主结果和论文未改写。全量 Viz 能力交叉验证继续，不能计为全部实验完成。

- 核心五模型新增十答：30,910/175,050 条已生成；17,860 条不可变分片和 pilot 已验收；待生成 144,140 条。
- 两个 Mistral 尾片共 17,460 条，跨片键交集为零。CPU 复用已评分 3,349 条，14,111 条待语义判断，对应 13,074 个新增去重 QA。
- 扩展四模型旧待标队列 46,439 个 QA，实际 Luna 加 root 裁定已接受 2,181 个，剩余 44,258 个。它与新增核心五模型 QA 分账，不把生成量计为标注量。
- 当前来源仅从 `accepted_sources.json` 的 SHA 验证清单加载。关键词脚本生成的 batch0008 草稿被拒收，已由实际逐条 Luna 判断和 root 复核替代。

当前生产：6403 GPU0/1 的 Gemma s2/s3；K100 的 Mistral s1。A100 有限队列粗估剩 3.3–4.5 小时；K100 Mistral 约 0.8–1.1 小时，随后保留的 Mini s0 约 0.2–0.4 小时。标签不含在 GPU ETA 内，未对完整语义闭合作完成承诺。

K100 原事件等待器因 Python 缺少 `os.pidfd_open` 于启动时失败。已核验 x86_64 / Linux 5.15 / glibc 2.35，使用同一 pidfd 内核接口的 syscall 434 修复。实际新进程 2953645，starttick 98922635，FD3 为 `anon_inode:[pidfd]`。没有增加定时器或睡眠轮询。Mistral supervisor 2929123 未中断；原 Mini reservation SHA 保持，仍在 35,010 条完整唯一键验收后才启动。

当前证据目录：`outputs/paper_core_20261002_dev_viz/closeout_20261004/`。`CURRENT_STATE.json`、`RUN_STATUS.md` 和 `KDM_Nine_Current_Review.zip` 为当前入口；冻结原始归档作为来源依赖保留。等待器原失败回执及新修复回执分别保存。

新增 CPU 入口 `workflows/paper_core/receive_viz_independent_piece.py` 已在真实 s2 尾片运行通过：输入来源 SHA、原始 seed/配置、有限概率、完整 expected-key 集、零重复/缺失与 pilot 零交集全部通过；官方 Viz 共识得分保持连续值，未把未决记错，未启动 GPU。
