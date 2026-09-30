# 四模型提取与同算法缓存交付（2026-10-01 03:51）

本轮 Food-101 的范围为 InternVL3.5-8B、OneVision、Phi3.5、Qwen3-VL。原生无引导 VCD/M3ID 各一个 NONE/NONE 条件，八个条件均已生成封存，各 2,424 条，共 19,392 条。CPU 检查点 `outputs/supplemental/remaining4/native_cpu_20261001_0226/checkpoints/v4_19392_all_native_closed` 已核对键、101×24 配额及 canonical/literal/abstain 未决均为零。十次独立试答、排名与参考连接仍在接续。本轮没有新增 guided VCD/M3ID 网格。

`native.py` 复用原参数和算法，由同次首八条生成核对真实 VCD/M3ID 算子；`references.py` 将选定模型的试答与排名交给原 `remaining11.generate`。原模型精度、双卡映射、图像处理、tokenizer、模板、seed、token 预算和候选评分均未修改。主五模型数据与源入口保持冻结。

## 缓存实际准入

v1 的 `prefill_cache.py`、`prefill_gate.py`、`cached_references.py` 与其测试保留原实现及已生成排名。v2 的对应 readonly 文件限定已验证的 DynamicCache/DynamicLayer，逐候选复制缓存结构并共享只读 prefill tensor；不支持的缓存形态停止，不自动降级。

`outputs/supplemental/remaining4/cache_gate_readonly_20261001_internvl_fast8/success.json` 保存实际八题、808 个闭集候选 sum/mean 分数与 rank 零差，以及 58,176 次 base tensor 相等核对。原八题分数来自实际 v1 gate，来源 SHA 明确保留，不能将其称为同次加载的计时对照。生产入口绑定实际成功回执、注册表、模型、原输入与缓存源码 SHA。

本轮一次三 worker 实际共享窗口生成 85 个完整输入，508.7 秒，合计约 5.985 秒/排名输入。该数来自完成事件前后完整键差量，不是单 worker 时间除以三。可核查来源为 `outputs/supplemental/remaining4/receiving_manifest_20261001_0226/rank_three_shared_window/result.json`；负载变化或类别输入差异会改变后续耗时。

## 所有权与当前接续

6403 使用独立 `6403_host_registry.json`：物理 GPU0/1、每卡最多三个槽，InternVL 仍为原双卡 36 层 18/18 映射，其每次加载保留 22,528 MiB free 守卫及继承锁；OneVision/Qwen3/Phi 的原单卡守卫为 28,672 MiB。d4030 的独立登记只允许物理 GPU0/1 和原 InternVL 双卡 runtime。Phi 六个原 3090 排名 worker 保留原分片区间与其他共享任务。

有限排名队列 PID 1049915 使用进程完成事件与至多 64 键的新 claim，不新建定时器、GPU 轮询或科学失败重试。已有失败、retirement、partial、owner/keys 和完成来源都保留。新任务只领取全局完成键之外的互斥区间。

最新有限 CPU 快照均位于 `outputs/supplemental/remaining4/receiving_manifest_20261001_0226/`，以下是不同采集时刻的事实，不能冒充实时总表：

| 阶段 | 采集时间（北京时间） | 完整回答/排名键 | 已封存可接收 | 未完成 | 按该次实测负载的条件 ETA |
|---|---|---:|---:|---:|---:|
| InternVL 十答 | 03:31 | 13,353/48,480 | 12,880 | 35,127 | 8.74 h |
| InternVL 排名 | 03:20 | 455/4,848 | 408 | 4,393 | 7.30 h |
| Phi 排名（六分片） | 03:20 | 2,587/4,848 | 独立收据列出 | 2,261 | 最慢原 owner 8.22 h |

显式十答 immutable 清单为 `reference_immutable_manifest_final_event_20261001.json`，26 parts/12,880 rows；接收者只能按真实旧接收/评分回执排重，不读取活动 raw。原始十答 attempts/replicate、temperature=1、top_p=1、32 token 预算与来源保留。是否应弃权的交叉验证已后置，不能把提取完成视为参考 GT 或该判断已完成。

K100 的旧四模型软件环境已出现真实 `no kernel image available`。同 checkpoint、精度、方法和模板的软件兼容提案已保存，用户尚未确认新的 Torch/TV 身份；因此新兼容 runtime 的模型 GPU 生产仍未启动。原软件失败不重试，也不把最小 CUDA 算子通过当模型准入。

本轮同步中央的八份 cache 源码来自 6403 实际冻结字节，原 v1/v2 success 绑定 SHA 均相同；同步不改变任何活跃 producer。小型源码由 root 按明确文件清单提交，大型 raw、标签、权重、图像与结果保持数据路径管理。
