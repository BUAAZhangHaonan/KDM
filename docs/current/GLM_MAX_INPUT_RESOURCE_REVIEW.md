# GLM旧单卡最大视觉输入资源补充

三种真实session组合均已按旧单卡BF16原生spec首次执行，全部在空前缀OOM。原始错误与此前成功分支保留，没有重试、降分辨率、改dtype、关闭分支、分配器启发式或offload。三个进程已实际退出，最后PID322122不再存在，物理GPU0已释放（6MiB，0%）。这是资源限制，不能将原生接口或已通过SID参考核验改称结构不支持。

范围只覆盖视觉token数最大的实际VizWiz样本（6030视觉token），按空/1/2固定前缀调度layer、instruction_vcd和CDA三种真实session组合。由于均在空前缀失败，尚未执行的分支和1/2前缀不能声称通过。不进行答案采样/科学打分，不声称覆盖所有问题长度或全部32生成步的绝对峰值。

|组合|失败位置|已完成有限forward数|申请/当时空闲|峰值allocated bytes|峰值reserved bytes|耗时秒|
|---|---|---:|---|---:|---:|---:|
|layer|forward/0/main_need_layers|0|1.71GiB / 826MiB|23814273536|24809308160|406.8|
|instruction_vcd|forward/0/noise_reference|1|1.71GiB / 976MiB|24421913088|24886902784|707.4|
|cda|forward/0/abstention_image|2|1.71GiB / 982MiB|24422600192|24901582848|1187.2|

layer只构造一个need_layers图像分支；instruction VCD构造三图，main通过后noise_reference失败；CDA构造三图加两text，prior_text与context_image通过后abstention_image失败，尚未到null_prior/null_context。各组是独立进程，前一组实际退出才运行下一组，未叠加不同方法组制造额外负载。

三份资源记录的内嵌spec与outputs/records/spec_history/glm46v_single_gpu_before_dual.json逐字典完全相同。CDA启动命令仍指canonical路径：冻结路径协调消息到达前它已读取旧spec、完成加载并开始前向；当时确认canonical与冻结副本逐字节相同，之后进程不再读取canonical。此后新双卡配置/进程不改变这三份旧单卡结果。此次仅写新记录/补充review，原计数、Qwen记录与PREREG等封口文件未修改。

原始记录与SHA（同名log保留完整traceback）：
- `outputs/verification/glm46v_max_input_resource_v1_layer.json` SHA256 `732c57c401c09b09e7e67c58fb06179222b36558ee1dfec3f0d154577e26c5a5`
- `outputs/verification/glm46v_max_input_resource_v1_instruction_vcd.json` SHA256 `24c0fe2cc157e944f48ad33dc95dc5d06051264ae74a521a6a06f72799081724`
- `outputs/verification/glm46v_max_input_resource_v1_cda.json` SHA256 `3a71b9cad8d0b14a157422b746dc34b6767f58653bd792d2b9920981826d3259`

进程/退出码、两种峰值、完整原始OOM、日志SHA、冻结spec身份汇总在 `outputs/records/glm46v_max_input_resource_v1_summary.json`。双卡已另获授权，由独立核验任务继续；本文件不预填双卡通过结论。
