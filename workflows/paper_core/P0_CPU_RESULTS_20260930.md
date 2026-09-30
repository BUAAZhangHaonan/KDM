# 核心五模型 P0 CPU 实际结果

执行日：2026-09-30。中央证据目录为 `outputs/paper_core_20260930/run_20260930_core_p0`；独立报告见该目录 `P0_CPU_REPORT.md`。本地完整镜像为 `E:/OneDrive/文档/Playground/kdm_remaining11_20260930_140337/core/p0`。

新增入口 `p0_cpu.py` 只做 CPU 来源绑定与既有结果复算。它读取实际服务器 `outputs/paper_20260929` 与交接包，保存新产物，旧五模型数据、中文 v1、英文正文及 remaining11 检查点均保留。

- 原冻结 352 条条件 × 2,424 输入 = 853,248 条决定与包内 NPZ、原数量和指标完全一致。canonical、literal、abstain、uniform-reference、QA ID 逐条差异均为 0；35,552 项条件—类别配额均为 24。
- 全部 24,240 条 Food dev/eval 参考具有 10 次独立试答，统一 GT 公式与原决定差异为 0。
- `frozen/all_conditions_352.csv` 包含全部 Acc/P/R/F1 与数量；`guided_same_marker_4_per_model.csv` 的 48 条同措辞条件与 `guided_cross_reference_16_per_model.csv` 的 192 条交叉条件独立保存。
- `transitions_reference_stratified.csv` 包含 412 对条件 × 18 个 C/E/A—参考分层单元 = 7,416 行；每对分母 2,424，分层总数与原参考集合一致。
- 既有无引导 Direct 的 12,120 条 Food eval raw 全部匹配当前样本、task key、prompt、配置、seed、原身份及完整注册 runtime，0 缺失/重复/差异，505 项 101×24 类别配额通过。使用最终 293,344 条闭环标签及既有 exact-QA 判断接续评分。两 Qwen 条件完整，另三模型各有 2 条主菜边界，完整 Acc 保持待决；六条完整原文在 `native_direct/pending_primary_rows.jsonl`。
- 方法目录 `method_inventory_362_rows.csv` 保留 352 条冻结条件、5 条既有无引导 Direct、5 条待生成无引导 VCD；后者缺 12,120 条。native、guided、IP、reference-off、CDA 与 M3ID 参数分列。
- 20 个 CDA 条件的固定 representative101 共 2,020 条已有回答、18,545 步四熵校准 trace。式 6/7 最大复算误差 0；零和扩展和负 wa 保留。已有资产未保存 Ha 和全分支 logits，式 4 的全向量无法重算。
- 有限核对 Gemma3-12B、Gemma3-4B、GLM-4.6V 三模型原 census 共 55,002 条、24 个真实分组，当前键/配置/来源检查通过。HF 登记分别是 `google/gemma-3-12b-it`、`google/gemma-3-4b-it`、`zai-org/GLM-4.6V-Flash`。原用户历史名称、数值和未知原始来源身份保留。
- 历史 Food dev+eval4,848/旧别名评分与当前 eval2,424/主菜评分，以及历史 VizWiz4,319 与当前补充 eval3,501，分母和口径均不同。六行历史表未宣称从其原 raw 重算；当前补充表与当前 census 的不同身份另列。范围差值未当作方法效应。

实际验收：包内 `recompute_frozen.py`、4 个数学测试、5 个措辞选择测试、`p0_cpu.py` 主处理、`--history-only`、`--verify-output` 均退出 0。数学/合成选择测试不计为经验模型生成。运行使用中央 `venv/bin/python`；新 GPU 生成、新 API 调用均为 0。

来源和逐项证据见 `P0_CPU_SUMMARY.json`、`P0_CPU_ACCEPTANCE.json` 及各阶段 validation/test JSON；大数据和 trace 按上述数据路径管理。后续主正确性裁定只追加新版本，无需重复生成既有 Direct。
