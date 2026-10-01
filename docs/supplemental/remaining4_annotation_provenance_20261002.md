# 补充标注来源撤回与接续

仅针对 `outputs/supplemental/remaining4/` 的新增标注。发现部分批次复用旧标签或以程序默认值写入，原先“逐条阅读完整问答”的回报已撤回；所有原文件保留。

`compose_semantic_authority.py` 接受有限、精确绑定路径与 SHA256 的 `invalid_actual_read_sources`。被撤回来源同时保持行为和回答质量待判，不能凭旧的 read/span 字段重新成为已决。后续真实阅读的独立来源与 root 裁定按既有优先级接续。

实际验收来源：`invalid_read_provenance_20261001_2337_v2/source_config.json`、`semantic_authority_20261001_2337_v12_invalid_read_holds/` 与 `received249666_20261001_2340_provenance_held/`，均位于上述独立输出目录。27 个来源声明涉及 3,700 条 QA，其中 94 条后来真实 root 裁定保留，3,606 条当前选用来源失效并保持待判。31,800 条来源绑定不等于语义标注全部完成。

CPU 回填验收：249,666 条唯一键、零重复；159,831 条 Food 记录保持不变，VizWiz 原始官方分数与参考标签零改动。弃权与质量的未决值继续保存为空，程序写入耗时不计作 Luna 标注速度。原五模型冻结输出与论文正文不改动。
