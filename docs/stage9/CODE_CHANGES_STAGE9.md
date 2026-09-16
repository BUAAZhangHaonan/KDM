# 第九阶段 · 对交付包代码的修改说明（CODE_CHANGES_STAGE9）

依据 `00_SUPPLEMENT_FOR_ZCODE.md`（本文件优先级高于包内 00—10 文档）与包内
`05_PROMPT_FOR_ZCODE.md` 的授权（"允许修复本包自身的确定性代码问题，但任何
影响输入、评分、选择规则、候选约束、停止词或统计口径的变化必须先写清原因并
提交"），对 `deliverables/KDM_next_stage/code/` 安装到 `code/` 的副本做以下
修改。`deliverables/` 内的包原件保持逐字节不变（`MANIFEST.sha256` 校验通过）；
包内测试 31 项全部通过（`outputs/logs/stage9/package_tests.txt`）。

以下修改全部在**任何真实模型推理之前**提交。

## 1. `stage9_run.py`：三噪声种子稳健性（补充说明 §3.2，影响统计口径，已写明原因）

- `probe_sequence(em, inputs, references, seeds, ids, validate)` 接收**三个**参考
  分支：种子 = 既有逐样本规则 `sha256(file:vcd) % 2^31` 加固定偏移 0、1、2
  （`NOISE_SEED_OFFSETS=(0,1,2)`），不改动原实验种子。
- 每条前缀上对三个参考分支分别求精确区间并逐步求交；记录
  `robustness=[{seed_offset, seed, interval}×3]` 与
  `robust_stable = 三种子"区间是否为空"结论一致`。
- 见证生成仅在 `robust_stable` 且主种子区间非空时执行（用**原种子**的参考分支
  与区间内部见证强度），不稳定路径不做见证生成、不进入主要结论。
- 恒等式校验（`identity_error ≤ 1e-8`）保留在主种子分支上，逐位置执行。
- 修改原因：包内区间判定条件于一次参考噪声实现，补充说明 §3.2 明确要求三
  种子稳健性，且"区间为空/非空在三种子间翻转"的路径必须单列为不稳定。

## 2. `stage9_run.py`：`--methods` 与 `--no-geometry`（补充说明 §3.1 的执行顺序）

- `run()` 本就接受 `methods` 参数；仅将其接入 CLI（`--methods direct,vcd,m3id`
  默认全量）并新增 `--no-geometry`。用于先只跑**直接解码**（原图复用缓存、
  两个退化条件新推理）→ 运行 `stage9_validity.py` 判定 → 再补齐 vcd/m3id 与
  路径分析。分阶段不改变任何科学定义：键相同、续跑逻辑相同、正式矩阵不变。

## 3. `stage9_validity.py`（新增，补充说明 §3.1）

- 读取三条件的直接解码记录，按模型判定：短边 64 或短边 32 准确率落在
  [0, 0.10]，或三条件中任意两对（orig–64、orig–32、64–32）的类别配对自助
  95% 区间包含 0（不可分辨），则该模型上退化不构成有效证据强度操纵。
- 输出 `outputs/tables/stage9/evidence_validity.json`；判据与自助次数（2000、
  类别聚类、种子沿用 `stage9_common.cluster_bootstrap_pair`）在文件内冻结。
- 判定无效时：如实报告，写明"本轮不具备检验'数据集难度'假设的条件"，不更换
  退化种类、不追加条件、不调参数。

## 4. `stage9_finalize.py`：四类路径分类与不稳定单列（补充说明 §4 与 §3.2）

- 新增模块级 `path_class()`，按补充说明 §4 的四类操作化逐路径分类：
  `nonempty_contains_default`（默认强度已足够）/ `nonempty_not_at_default`
  （强度选择问题）/ `empty_target_outside_support`（支持集合问题）/
  `empty_constraints_incompatible`（排序约束不相容，含空区间 reason 为
  `incompatible_constraints` 与 `target_never_overtakes_competitor`）；
  另有 `unstable_across_seeds` 与 `outside_sequence_budget`。
- 新表 `path_reachability.csv`（逐路径：三种子区间经原始 JSONL 保存，表内含
  分类、是否含默认强度、见证与验证结果）与 `path_classification.csv`
  （模型×条件×类别计数）。
- `name_reachability.csv`（图 7 数据源）的文件级归类只使用**通过稳健性检查**
  的路径；全部实测路径均不稳定时归类 `unstable_across_seeds`。
- `reachability_robustness.json`：操作化声明、通过比例
  （`pass_fraction = n_stable / n_paths`）、`appendix_demotion_required`
  （通过比例 < 1/3 时为真，几何结果降级为附录材料）与不稳定路径清单。
- 修改原因：补充说明 §3.2/§4 的直接要求；分类不表示对象知识有无。

## 5. `stage9_pope.py`：COCO 图像下载的 TLS 主机名例外（环境问题，不改科学定义）

- 现象：本服务器网络路由上，`images.cocodataset.org` 的 S3 边缘节点只出示
  默认 `s3.amazonaws.com` 证书，其 SAN 不含 `images.cocodataset.org`（2026-09-17
  实测：直连与代理、openssl 与 python 一致；`*.s3.amazonaws.com` 通配符只覆盖
  单标签主机，路径风格与 Host 头改写均返回 404/NoSuchKey）。
- 处理：仅对该图床主机关闭**主机名校验**，保留**证书链校验**
  （CERT_REQUIRED，Amazon 根签发）；官方标注文件仍走完整校验并按 Git blob
  SHA-1 指纹验证。
- 补偿控制：下载图像逐张 sha256 记入 `data/stage9/pope/source_manifest.json`
  供事后审计；PIL 解码校验保留。
- 不改变：图像身份（由官方标注文件名决定）、问题文本、提示词、解析与判分。

## 6. `viz/stage9_figures.py`：图 7 增加不稳定类别

- `observed_reachability()` 的堆叠条形图类别加入 `unstable_across_seeds`
  （标签 "Unstable across noise seeds"），与 finalize 输出保持一致；其余
  绘图函数未动。

## 7. 未修改项

`stage9_common.py`、`stage9_geometry.py`、`stage9_analyze.py`、`stage9_prepare.py`
与 `stage6_*` 全部未改动（与包内 `MANIFEST.sha256` 记录一致）。
