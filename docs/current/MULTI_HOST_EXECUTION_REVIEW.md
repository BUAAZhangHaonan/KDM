# 多主机执行接入审查

本轮仅修改 4028 项目代码，运行 CPU 测试；未写 6403、未启动 GPU、未修改共享环境或模型数值实现、未提交。

`configs/runtime/hosts.json` 是唯一模型到主机分配：4028 保留十四模型；6403 的 Qwen3-VL 和 GLM 顺序使用物理 GPU 1。登记 hostname、项目根、允许卡与 UUID。worker 在建立项目锁前检查主机身份和卡，正式模型入口还检查继承 fd 20+、GPU 数量、准确 Python/包版本和本机权重分片 size。

`protocol.validate_freeze` 只读可同步的证明、清单、来源和冻结身份，不访问任何主机的权重路径。`validate_execution_runtime` 是生成新 native/SID 证明前的本机门，不要求尚未生成的 native proof；`validate_runtime` 在本机门之后加入已完成证明。新 6403 证明绑定主机、卡 UUID、registry/resolver/catalog 哈希。4028 原十四份 native proof 实际逐一通过现门，未要求补写历史 host 字段或重跑。

全部图片入口通过 `execution.resolve_image_path`：清单字节和 sample/image_path 原文不改；仅将原 Food 和 VizWiz 前缀映射到批准项目目录。拒绝前缀相似、..、缺文件及 symlink 逃逸。6403 每个实际文件按原 catalog 的 size/SHA256 验证；进程内 inode/size/mtime/ctime 不变时复用验证结果，改变后重新核对。原 4028 路径保持相同。catalog 必须覆盖原清单全部 9167 图片。任务 ID 不变；迁移后的环境、spec、执行主机和冻结 run identity 必须重新登记，不能复用或重标旧 sidecar。

新处理器全清单计数支持 `--output` 独立路径，拒绝覆盖；真实检查目标主机、Python、包版本和权重，CPU 不初始化 GPU。外部环境源码保留实际绝对路径。资源门检查三阶段(layer/instruction_vcd/cda)当前环境/处理器、实际最大 VizWiz 样本、固定全部分支的 0/1/2 前缀、有限数值、host/GPU 和源码。这是有限前缀资源证明，不是 32-token 最坏峰值证明。Food SID 还核对原固定首图 ID 与内容 SHA。

同一静态全十六计划以 `--host` 分别启动。每主机完成只记 `host_complete`，整体 `complete` 保持 false。同步全部原始 JSONL 和原 identity sidecar 后，`--verify-panel` 对完整十六模型的来源、冻结、主机、配置、任务与覆盖执行现 provenance 门，通过后才写 panel_complete.json。无 SSH/RPC 自动连接、重试或替换。

## 验证

- 全 CPU：`venv/bin/python -m pytest tests -q --basetemp=$PROJECT/cache/pytest/model_adapters`，287 passed，23.30s；TMPDIR 为项目 cache/tmp，禁写 pycache。
- 原 4028 十四个 native 文件证明全部通过；原图读取映射保持原路径。
- `bash -n scripts/worker.sh`、所有 Python AST 解析、`git diff --check` 通过。
- 实际 `verification/execution_check.py`：28 个 CLI 子进程通过。记录：`outputs/verification/cli_fixture_20260919T124553845225Z/CLI_REVIEW.json`。其 mock 端到端覆盖仍不包含新正式 freeze 下的 mock census->selection；完整消费门通过独立来源/host 单元变异测试验证，不伪造正式模型证明。

## 新主机命令接口

先同步相同已提交源码、registry/catalog、完整原清单和小证明文件。以下 `$PROJECT` 是 6403 批准根，`$PY` 是最终 spec 登记既有 Python，`$KEY` 分别 qwen3vl/glm46v；这份说明未执行命令。

```bash
cd "$PROJECT"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PROJECT/cache/tmp" "$PY" verification/check_full_visual_counts.py "$KEY" --output "outputs/verification/${KEY}_6403_full_visual_count_v1.json"
bash scripts/worker.sh "$PROJECT" 1 "$PY" -m kdm.models.verify --spec "configs/runtime/$KEY.json" --manifest data/current/interface16.jsonl --out "outputs/verification/${KEY}_6403_native16_v1.json"
bash scripts/worker.sh "$PROJECT" 1 "$PY" verification/sid_reference_check.py --spec "configs/runtime/$KEY.json" --output "outputs/verification/${KEY}_6403_sid_reference_v1.json" --device cuda:0
bash scripts/worker.sh "$PROJECT" 1 "$PY" verification/max_input_resource_check.py --spec "configs/runtime/$KEY.json" --stage layer --output "outputs/verification/${KEY}_6403_max_layer_v1.json"
```

资源的 instruction_vcd 与 cda 各用独立进程/输出，参数不变；先将新 count 的 visual_count_verification 状态登记 passed。全部证明完成、登记、冻结提交后才可分别：

```bash
"$PY" scripts/run_census_panel.py --execute --host 6403 --run-dir outputs/records/census_6403_RUN
# 4028 用其调度 Python，同一源码计划：--execute --host 4028
# 同步所有原始 ledger + sidecar 后，在中央根：
"$PY" scripts/run_census_panel.py --verify-panel --run-dir outputs/records/census_panel_complete_RUN
```

`resource_verification.records` 必須列 layer/instruction_vcd/cda 三个真实输出；`visual_count_verification.record` 必须指新 count。新 spec 无需 execution_host 字段，host 分配唯一来自 hosts.json。
