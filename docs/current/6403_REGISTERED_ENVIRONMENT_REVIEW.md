# 6403已登记来源环境补充

本补充取代最初盘点在三套候选内建议的mprisk选择，但不覆盖初版盘点证据。优先使用现成的 `/home/team/zhanghaonan/.venvs/mprisk-kv-transformers-5.5.3/bin/python`。它就是configs/runtime/environments.json已明确登记的mprisk-tf553.source_python，实际存在并链接至mind-py311解释器，未扫描其他环境。

实际torch2.6.0+cu124、torchvision0.21.0+cu124、transformers5.5.3、accelerate1.6.0、tokenizers0.22.2，与4028重建环境登记的核心版本一致。torch/torchvision/accelerate/PIL/safetensors及Qwen3、GLM、AutoProcessor/AutoModelForImageTextToText/AutoConfig全数导入成功，CUDA未初始化。驱动595.71.05声明CUDA13.2，覆盖该CUDA12.4构建版本；尚未进行GPU kernel/模型验收。

源环境有已发生的版本漂移：numpy2.4.6（4028登记2.2.4）、Pillow12.2.0（11.1.0）、scipy1.17.1（1.17.0）、requests2.34.2（2.32.5）。不能将它称为逐包等同于4028验证环境，也不能据此复用Qwen3/GLM在4028另一套TF5.17环境的旧证明。后续以真实当前环境身份重新进行原生输入/16图、SID及资源检查；不升级、降级或修改共享环境。

证据为outputs/records/6403_environment_registered_tf553_inventory_v1.json和6403_registered_environment_recommendation_v1.json，原始三环境盘点及最初建议保持历史字节。本次只增加上述指定来源环境的只读导入记录，无GPU权重加载，所有盘点记录仅写4028。
