# 已验证导出清理

2026-10-04，当前九模型有效交付为 `outputs/paper_core_20261002_dev_viz/final_review_package_complete_accepted_sources_v2_20261003/KDM_Nine_Complete_Review.zip`，SHA256 `f12f2c90e00e0f4134a9e8e4d7842c813d3001dcbbc8d8d1e3edaee1a39c1d48`。

已删除574个被替代导出文件，共239,567,745字节：旧Food v5 ZIP、旧开发集columnar ZIP、final v1 ZIP及其解压副本、final v2验收临时解压副本。两个旧输入ZIP的192个逻辑成员均在当前包内逐字节保留；final v1仅四项说明/状态元数据发生更新，独有执行来源和历史回执保留。

`configs/kdm/review_archive_members_20261004.json` 保存两个历史输入ZIP的成员名称、原SHA、大小以及当前ZIP成员定位。活动assemble/finalize入口读取此映射，验证当前ZIP及每个逻辑成员SHA后重用相同内容。真实192成员已通过两个活动入口的读取和文件导入核对；无需重新生成模型回复或评分。

详细逐文件替代清单、删除回执、四项旧元数据摘要、单一结果索引保存在 `outputs/paper_core_20261002_dev_viz/closeout_20261004_cleanup/`。历史构建回执中的旧导出路径通过 `cleanup_plan.json` 和成员映射定位。原始五模型、全部有效raw、已验标签、参考、图像、权重、DoLa/SID配对原结果保持。

五台服务器项目outputs目录四层内已检查导出ZIP；此次可验证冗余集中在中央。具有独有来源或当前评分依赖的中间文件不按名称删除。清理仅整理已证实替代的产物，不把删除数量当作实验完成量。
