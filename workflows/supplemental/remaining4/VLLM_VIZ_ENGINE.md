# VizWiz 十次试答：已登记独立引擎cohort

OneVision/Qwen3-VL 的旧Food十答采用独立vLLM引擎cohort；本次只把相同已接受实现接到VizWiz注册eval3501×10。原vLLM0.24.0、Torch2.11.0+cu130、Transformers5.12.1、各自checkpoint/dtype、temp1/top_p1/max32、稳定sample/model/replicate seed、首个真实请求加九个并发prefix-cache请求保持原样。该cohort明确不声称native随机抽样或log probability完全相等；两种原生precision分别为OneVision float16、Qwen3-VL bfloat16。

原实现按历史production源码SHA从已归档提交取回到独占outputs source snapshot。`vllm_viz.py` 仅作dataset/explicit missing-key/claim调用适配；实际v1 SHA `b475b205128b28a334bc582136e428a370fbac4fcd59ff2abfb4fbb003fdc125` 已在首次GPU加载前记录且保留，不修改已运行字节。`vllm_viz_registry_20261001.json` 只允许K100物理GPU0、独占原卡锁、加载前至少92160MiB free，原输入logical prefix在本机保留字节映射。

2026-10-01原TF5.5.3 native环境已分别实际生成两模型8个完整Viz comparator（formal0），保存图像、问题、processor tensor、模板、unexpanded/expanded prompt IDs、EOS及greedy全部token和所选log probability。旧Food门检不能代替本次Viz门检。

OneVision v1 实际完成8输入×10试答，图像/processor tensor/expanded IDs/EOS/finite log probabilities/十稳定seed/cache核验全部通过，8个greedy token序列与native对应相同。末端Ledger identity漏带既有verify_output要求的`shard/n_shards/base_config`而拒绝封存；原raw80、sidecar、claim和failure完整保留。这是记录接口故障，不能宣称原sidecar已经通过。

`vllm_viz_recording.py` 是独立记录适配：仅补齐上述三个原登记字段，并显式记录实际caller与adapter源码SHA。CPU `--seal-existing-gate` 从原80稳定键追加新derivative，既有verify_output实际通过且逐项核对全部科学字段相等；没有新的GPU生成调用。source manifest与实际门检比较在 `outputs/supplemental/remaining4/dispatch_20261001_1600/derivatives/onevision/vizwiz/independent/gate80_recording_seal_20261001_2050/`。原v1失败回执仍为失败；新派生80只计一次，后续严格只claim未生成34930。

Qwen3-VL也实际完成自己的8输入×10试答，图像、processor tensor、完整提示、expanded IDs、EOS和稳定seed核验全部通过。greedy token序列7/8与原生环境相同，剩余一个序列的实际差异在独立引擎cohort中明确保存。其原80同样通过CPU记录派生验收，路径为 `derivatives/qwen3vl/vizwiz/independent/gate80_recording_seal_20261001_2055/`。两模型160条实际门检产物及42个源成员已严格接收到 `received_vllm_actual_gate80_2105/received_manifest.json`，源SHA检查通过，原失败来源保持。

K100新的OneVision尾部34930条已于20:55:59实际启动，使用固定记录适配SHA `1f041f9fd973c303a25c56c0f563183b7069fae64e2152fe8bdcef78b08f27af`。首个520条分片经过现有`verify_output`实际验收，原科学参数保持，完整source/owner/keys/identity随分片保存。后续Qwen3-VL34930条依该producer完成事件接续。八输入门检时长仅为小批测量，完整队列耗时使用后续封存分片计时。任何真实模型/输入语义故障停止对应cohort，不改变processor、采样参数或precision。正式pipeline/当前五模型冻结入口保持原样。
