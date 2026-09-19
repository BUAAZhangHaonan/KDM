# 6403模型与既有环境只读盘点

建议后续两模型使用 `/home/team/zhanghaonan/miniconda3/envs/mprisk/bin/python`，保持checkpoint和原始输入参数，并在目标机重新完成身份与真实接口核验。现有专用Qwen/GLM环境都缺accelerate，不能直接满足当前HFBackend显式device_map构造。没有修改任何环境，也没有在6403创建文件/缓存或加载GPU权重。

主机6403=172.17.43.38（oem-PowerEdge-T640），登录zhanghaonan。只读快照GPU1为A10080GB，81920MiB中使用14MiB；GPU0已占80671MiB，不使用。driver595.71.05，nvidia-smi声明CUDA13.2；mprisk torch构建CUDA13.0，因此驱动声明的CUDA能力覆盖该构建版本。该盘点禁用CUDA可见设备且所有import/processor检查后cuda.is_initialized=False，尚未以GPU初始化或实际kernel证明运行兼容。

|环境|torch包版本|transformers|accelerate|Pillow|关键导入|
|---|---|---|---|---|---|
|lvshuyang Qwen3-VL-8B|2.9.0|5.5.4|缺失|12.1.0|Qwen3/GLM/Auto模型与processor可导入；accelerate失败|
|lvshuyang GLM-4.6V|2.9.0|5.0.0rc1|缺失|12.1.0|Qwen3/GLM/Auto模型与processor可导入；accelerate失败|
|zhanghaonan mprisk|2.13.0+cu130|5.5.3|1.14.0|12.2.0|上述及torchvision/safetensors均通过|

4028原环境为torch2.9.0+cu128、transformers5.17.0、accelerate1.12.0、Pillow12.0.0。mprisk缺einops/flash-attn，但所需两模型及processor原生导入不依赖它们，此盘点未安装或启用替代attention。具体torchvision/tokenizer/numpy/source SHA见独立环境JSON。

两个6403模型位于`/home/team/lvshuyang/Models/{Qwen3-VL-8B-Instruct,GLM-4.6V-Flash}`。对比4028当前登记路径：Qwen11个、GLM9个小文件的SHA全部一致，包括config、tokenizer、preprocessor和chat模板等实际存在文件；各4分片名称/字节数/safetensors header SHA全部一致；Hub metadata的revision+etag分别16/14条全部匹配。Qwen固定revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b，GLM固定411bb4d77144a3f03accbf4b780f5acb8b7cde4e。没有无谓读取几十GB tensor payload，故不将这些身份字段称作重新计算完整权重哈希。

目录和权重归lvshuyang；所查模型目录及权重权限为777，zhanghaonan可读（权限也允许写，但本任务明确只读，未写）。专用环境目录775归lvshuyang，mprisk目录775归zhanghaonan。所有文件逐项权限与metadata在checkpoint inventory记录中。

mprisk实际CPU构造得到Qwen3VLProcessor/Qwen2VLImageProcessor及Glm46VProcessor/Glm46VImageProcessor，两者image_processor.to_dict与4028实际全量count记录逐字段无差异；同本地模型config/processor文件身份相同。原生尺寸计数的两例(1936,2592)/(384,512)分别为Qwen4860/192、GLM6030/252，与原输入规则相符。这只是原生配置与计数检查；transformers源码版本不同，不能据此声称像素或logits完全一致、也不能继承4028旧证明。

初次processor inventory尝试dict(MultiModalData)导致报告序列化TypeError，发生在native计算完成后，并非模型/processor不支持。原记录未覆盖；`6403_mprisk_processor_counts_v1.json`改为读取返回值命名属性，保存真实计数及CUDA未初始化证据。

完整索引/逐文件SHA：`outputs/records/6403_readonly_inventory_summary_v1.json`。全部证据仅写4028项目根。本盘点未触碰已授权但由主线程准备的新6403隔离目录，也未启动任何GPU模型。
