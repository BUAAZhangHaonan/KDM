# 第三阶段跨模型可用性清检(运行前实检,2026-09-13)

## 纳入(4 族 4 个模型,全部实测加载+解码通过)

| 模型 | 架构族 | 磁盘 | 加载 | direct | LCD(DoLa) | VCD/MIB | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct | qwen3_vl(原生) | 8.3G | 9s | 'Pizza' | OK | OK | patch_embed Conv3d 病态路径,以等价 matmul 替换(数值验证 max|diff|=0.008);图像最大边限 640px(实测原生 9.5s/次 → 0.29s/次) |
| llava-v1.6-mistral-7b-hf | llava_next(原生) | 15G | 36s | 'Pizza' | OK | OK | anyres 高分辨率 prefill(~2.4k token),~1.2s/次 |
| InternVL3_5-4B | internvl_chat(远程代码) | 8.9G | 8s | 'Pizza' | OK | OK | 远程 processor 与 transformers 5.17 不兼容(tokenizer.start_image_token 缺失);绕过:手动 ChatML 模板+单 448 方形裁剪+image_flags=1+显式 img_context_token_id;另需补 all_tied_weights_keys 属性与 timm 依赖;~1.0s/次 |
| (基线)Qwen3.5-4B/9B | qwen3_5(混合线性注意力) | — | — | — | — | — | 第二阶段已全量,第三阶段复用其主实验记录 |

## 排除(按任务书第九节写明原因)

| 候选 | 原因(实测) |
| --- | --- |
| GLM-4.6V-Flash (glm4v, 20G) | 加载与解码可用(enable_thinking=False 后正常答 `<|begin_of_box|>...`),但稳态吞吐实测 **16.5 秒/次解码**(3 样本均值;预填 ~300 token),全流程(分层 2424+验证 1200+闭集 2400+主配置 4800×双分支)约需 **2 天/卡**,超出本阶段预算;按第九节排除,不以缩减流程的方式保留 |
| MiniCPM-V-2_6 | 权重目录仅 48K(不完整),无法加载 |
| Ministral-3/8B | 目录不存在,且为纯文本模型 |
| Qwen3.5-35B-A3B | bf16 约 67G,超出单张 3090(24G)显存 |
| InternVL3_5-8B | 与 4B 同族;因 4B 已代表该族且时间有限,未启用(非不可用) |
| Qwen3-VL-8B-Instruct | 与 4B 同族;同上 |

注:任务书建议的"不同族大尺寸"未能满足(GLM 为唯一候选大尺寸不同族,因吞吐排除);以 4 个 4-8B 级、3 个不同族(qwen3_vl / llava_next / internvl_chat)+基线族(qwen3_5)覆盖架构多样性。
