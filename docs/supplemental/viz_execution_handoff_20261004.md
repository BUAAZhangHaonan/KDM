# 核心五模型 Viz 全量十答接续

2026-10-04 01:34 北京时间。这里只报告新一轮 GPU 提取；语义标注仍是独立工作。

## 已验证

- 5 模型各8图80答，共400答，已真实闭合：精确样本/replicate key、零重缺、10个稳定seed、原prompt展开、原EOS终止、有限logp、实际前缀cache。
- 中央 `received_pilots/<model>/pilot8/` 是通过 SHA 的收件镜像，`pilot_validation.json` 是有限全量验收。实际生产原件仍在对应服务器 `generation/`；镜像不能再计一遍。
- 原12个源码从 `b81fdafb078310e2fe7e15ea0f42e68566cc362e`逐SHA恢复并提交，中央重复source_snapshot的12文件已经逐SHA确认替代后删除，回执 `source_snapshot_retirement.json`。
- 6403原vLLM是 `/home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python`，不是不存在的项目vllm024。K100原vLLM是项目 `.environments/vllm024/bin/python`。两者3包版本精确匹配历史已接受记录。
- 4028/4029/d4030授权项目环境的有限检查未定位vLLM环境；没有安装新栈或为了上3090降低32K/maxseq/精度。

## 正在运行和下一张卡

|服务器/卡|进程|实际工作|
|---|---:|---|
|6403 GPU1|supervisor2124049，worker2124050|Gemma `tail_s3of4`，01:34已1210答，后续从共享有限queue取下一未claim尾片|
|K100 GPU0|supervisor2929123|Mistral4个尾片，首片01:32已4840答，完整十答图生成均值1.561秒/图|
|6403 GPU0|supervisor2129421，worker2129422|机制exit0后已于01:32接Gemma `tail_s2of4`，01:34加载中0答|
|K100完成后|pidfd事件等待进程2933428|Mistral35010答真正验收后，接已保留Mini `tail_s0of4` 8740答；先8个未生成图真实准入并计入本片|

6403 GPU0已执行以下命令，作为启动记录保留，**不要再次启动**。启动前UUID匹配、显存14MiB/利用率0、原机制进程已退出、GPU锁可取；新s2与s3及pilot键交集均0，回执 `a100_gpu0_join_receipt.json`。

```bash
/home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python \
 outputs/paper_core_20261002_dev_viz/closeout_20261004/viz/launch_queue.py \
 /home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation \
 6403 0 qwen25vl qwen35_4b minicpm26 gemma3_4b
```

每模型余3493图分4片。`generation/queue_claims.json` 在每服务器保存实际派发；6403的Mini s0标为reserved_external，排除本地派发。`minicpm26_k100_reservation.json` 已保存到6403、K100和中央。K100实际既有权重在 `cache/models/MiniCPM-V-2_6_core20260930`，CPU配置/processor/权重登记通过。Mini旧源host6403和新target host均如实记录，不宣称跨硬件逐token数值等价；生成算法及参数原样，真实输入准入继续检查。

若要收回尚未启动Mini保留片：先按PID/starttick停止自有事件等待进程2933428、确认Mini还未产生claim/raw，保存撤销回执；然后将6403 queue_claims中的reserved_external条目移出当前队列，才可重新派发。不要只编辑一台服务器副本。

## 成本（不是完成量）

最初8图中去除每模型首图JIT，后7图热生成均值：Q25 1.559s、Q35 1.391s、Mini .810s、Gemma4.137s、Mistral1.556s/完整十答图。对应3493图为1.51、1.35、.79、4.01、1.51 GPU小时。这里只是8图计时外推；Gemma当前23图含首次编译的均值5.506s，Mistral254图实际含prepare/落盘1.83s。预计两张A100占用约4–5小时，K100 Mistral约1.8小时后再Mini约0.2–0.4小时，须按真实分片完成更新。

## 已修复的运行问题

1. 新launcher漏原K100 worker的PATH，ninja原文件未被找到，0条生成。已恢复原vllm024/bin和CUDA PATH，不安装依赖、不改参数。失败证据在K100 `failures/llava16_mistral_pilot8_missing_launcher_PATH`。
2. Mistral80答写完后旧caller引擎退出延迟，后续尾片被GPU锁拦住，0条生成。验证80答SHA、PID/starttick后仅对自有进程组发SIGTERM；新caller使用官方engine_core.shutdown(timeout=30)。失败来源 `failures/llava16_mistral_tail_s3of4_previous_pilot_lock`，成功pilot不重跑。

## 代码提交

`ba1060d8` 源恢复；`48b6f7e8` Viz caller；`fa1857dd` 互斥key与真实输入捕获；`1cdf17ec` 有限多卡尾片队列；`c46244ee` 显式engine退出；`4f0fcd53` 已登记K100 Mini保留片准入。均只提交，root统一推送。

完整生产后以每模型35010唯一key验收，核心5共175050。需从各host取回已完成不可变分片，更新中央来源索引；不可把当前partial raw或400个pilot当成全量完成。CPU评分与语义标注可以按完成分片接续。
