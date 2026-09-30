有限接续队列由 `plan.json` 显式列出已有claim、未claimed阶段、依赖、GPU所有权和阻塞原因。当前数据阶段为Food-101。VizWiz实际缺口与官方答案审计保存在 `run/assets/vizwiz/audit_20260930`；Qwen9独立试答使用已经验证的独立队列，正式方法矩阵保留为后续阶段。

在各host的实际项目根目录执行：

```bash
python3 workflows/supplemental/remaining11/event_queue.py check --queue-id queue_cpu_20260930
python3 workflows/supplemental/remaining11/launch_event_queue.py --checked-queue-id queue_cpu_20260930 --queue-id queue_food_20260930
```

每次ID独占，不会覆盖既有队列。`check` 从实际入口目录导入标准库Queue、torch和transformers，再执行原runtime CPU准入与原generator `--check-plan`；Popen子进程退出通过Linux pidfd事件验证。`execute`重新检查准入后，通过pidfd等待原claim或本队列worker退出，无定时器和GPU轮询。状态、原始错误和有限事件列表保存于 `run/queues/HOST/ID/queue_state.json`，CPU检查与worker日志位于同目录。

`launch_event_queue.py` 要求当前计划与指定真实CPU检查完全一致，启用节点全部通过检查后，启动独立session并保存实际PID、完整命令、检查来源SHA与独占launcher日志。2026年9月30日实际三个runner的状态记录在 `run/queues/4028/queue_food_gemma_20260930_1040`、`run/queues/4029/queue_food_vicuna_20260930_1040`、`run/queues/k100/queue_vizwiz_independent_20260930_1040`。现有claim继续原区间。

当前启用的接续项目包括4028 Gemma12候选3835条→独立44870条、4029 Vicuna Food正式191496条，以及K100 Qwen9 VizWiz独立35010条。Gemma依次等待原formal和candidate成功完成，使用原tf553双卡0/1。Vicuna等待原Phi候选进程退出并独占GPU2，在完成事件触发时核对UUID与至少23552MiB可用显存。Qwen9等待原Food候选成功完成后使用GPU0释放的第二个槽，保留Food正式槽；原native311加载前要求61440MiB可用显存，max2与原32-token参数不变。以上预算属于原输入与源码的容量计算，新的全量GPU峰值仍需实际记录。已claimed区间与Intern formal失败范围保持。

6403 OneVision和MiniCPM4.5的两个v3 replacement已由root显式启动，实际claim纳入 `existing_claims`。原零生成导入失败经 `released.json` 绑定完整原键区间与指定replacement，原owner、keys、空raw目录、records与日志保留。计划中的对应启动节点保持禁用，双卡后续等待包括这两个实际claim在内的全部占用者退出。

不同模型之间的 `after_finished` 只承担资源释放顺序，前序失败会保留其来源并允许独立已准入项目继续；`requires_success` 对明确登记的阶段前置条件要求真实生成完成回执。失败项目不重试，已完成分片继续供评分。`finite_queue_exhausted` 仅表示有限队列结束，科学验收与标签/reference完成另有检查点。

6403 Llava13预留GPU0/1，等待GLM两个claim、Qwen3两个shard及OneVision/MiniCPM两个v3 claim全部退出。其双卡wrapper执行原generator并继承fd20/21的真实独占GPU锁，与所有共享slot互斥；原runtime、device_map、精度、seed及32-token配置保持。CPU检查回执和实际exclusive-worker命令分别记录。

K100 Gemma12预留被原双卡要求、单授权GPU、冻结版本不匹配、缺少checkpoint和真实路径准入阻塞。4029 OneVision/MiniCPM尚缺原checkpoint/processor身份、runtime路径和容量证明；6403 Llava13保留双卡资源等待与完整准入检查。VizWiz正式方法矩阵保留为后续独立阶段。科学完成状态依据完整键覆盖、标签与参考验收。

每次显式运行 `state.py` 会组合中央真实进度与最近跨机快照，更新紧凑 `CURRENT_STATE.json`、全条件分母表以及 `PROGRESS_ETA.json`、`PROGRESS_ETA.csv`。最近三个已完成分片记录原完成时间、SHA及实际method/kind数量，只读取不可变raw。两个分片间隔用于计算当前吞吐；不足两个完成分片时，平均速度注明包含加载时间。按模型和阶段区分复用、不可变完成、活动partial、仍claimed、真实失败held与尚未claimed，Food和VizWiz独立统计。耗时投影保持当前并行度和方法组合，后续方法、图像与回答长度变化都会影响速度。Luna边界标注与root复核耗时需要独立实际测量；已完成raw先做主答案提取，是否应该弃权的交叉验证后置。
