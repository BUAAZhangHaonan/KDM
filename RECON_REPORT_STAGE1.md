# 勘察报告(阶段 0–2 小时)

日期:2026-09-13。本报告记录运行前的环境勘察结果:模型清单、数据结构核实、运行条件与关键工程发现。

## 1. 硬件与运行环境

- GPU:6 × NVIDIA GeForce RTX 3090(每张 24 GB)。GPU 0 有他人进程占用约 3.3 GB,本实验使用 GPU 1(Qwen3.5-4B)与 GPU 2(Qwen3.5-9B)。
- 磁盘:项目所在文件系统可用 919 GB。
- Python:项目内自建 venv(`venv/`,基于 miniconda py311 环境的 `--system-site-packages`),关键版本:
  - Python 3.11.14
  - torch 2.9.0+cu128(来自宿主环境)
  - transformers 5.17.0(装入项目 venv;宿主为 4.57.6,**不支持 `qwen3_5` 模型类型**,必须升级才能加载任务书指定的 Qwen3.5 系列模型)
  - accelerate 1.12.0、flash-linear-attention 0.5.2(为 Gated DeltaNet 层提供 Triton 快速核)
- 缓存重定向:所有脚本统一设置 `HF_HOME/TORCH_HOME/MPLCONFIGDIR/XDG_CACHE_HOME` 指向项目内 `cache/` 子目录,不在项目外新建任何文件。

## 2. 模型勘察(/home/g203-4028/Models 全部内容)

| 模型 | 参数量(实测/估计) | 支持图像输入 | 显存需求(bf16) | 备注 |
| --- | --- | --- | --- | --- |
| GLM-4.6V-Flash | ~10B | 是(glm4v) | ~21 GB | 可用,但与 Qwen 系重复 |
| InternVL3_5-4B | ~4.7B | 是(internvl_chat,需 remote code) | ~9 GB | 本机 transformers 缺 internvl_chat 注册 |
| InternVL3_5-8B | ~8.5B | 是(同上) | ~17 GB | 同上 |
| llava-v1.6-mistral-7b-hf | ~7.6B | 是(llava_next) | ~15 GB | 可用 |
| MiniCPM-V-2_6 | — | — | — | **目录内无权重文件**(仅 assets/README),不可用 |
| Ministral-3-3B-Instruct-2512 | ~3B | 否(纯文本) | ~6 GB | 不适用 |
| Ministral-3-8B-Instruct-2512 | ~8B | 否(纯文本) | ~16 GB | 不适用 |
| Qwen2.5-Omni-3B | ~6B | 是(qwen2_5_omni) | ~12 GB | 音视频全模态,加载复杂 |
| Qwen2.5-Omni-7B | ~11B | 是(同上) | ~22 GB | 同上 |
| Qwen3.5-35B-A3B | ~36B(MoE,激活~3B) | 是(qwen3_5) | ~72 GB,超出单卡 | 不采用 |
| **Qwen3.5-4B** | **4.54B(实测)** | **是(qwen3_5)** | **~9.3 GB(实测)** | **选用(小模型)** |
| **Qwen3.5-9B** | **~9.7B** | **是(qwen3_5)** | **~19 GB(bf16)** | **选用(大模型)** |
| Qwen3-VL-4B-Instruct | ~4.4B | 是(qwen3_vl) | ~9 GB | 备选 |
| Qwen3-VL-8B-Instruct | ~8.8B | 是(qwen3_vl) | ~17 GB | 备选 |

**选型结论**:按任务书规则选用 Qwen3.5-4B 与 Qwen3.5-9B——两者原生支持图像、参数量差异明显(4.5B vs 9.7B,2.1 倍)、各自可装入单张 3090(9B 以 bf16 约 19 GB)。两卡并行,GPU 1 跑 4B,GPU 2 跑 9B。

### 关键工程发现(影响全部实验的实现方式)

1. **思考模式**:Qwen3.5 是思考型模型,默认输出推理文本。实验中通过 chat template 的 `enable_thinking=False` 关闭,保证答案为确定性的短回答。
2. **视觉塔 Conv3d 病态慢**:Qwen3.5 视觉塔的 patch_embed(Conv3d,kernel=stride=16/2 非重叠)在 cuDNN 下对本形状选择极慢算法,784 patch 需 7.7 s。因卷积核与步长完全重合,该卷积数学上等价于每个 patch 的一次线性投影。我们用等价矩阵乘重写(实测逐元素最大误差 0.008,为 bf16 舍入水平;耗时 0.1 ms,约 7000×加速)。此改动不改变任何计算结果,只影响速度,并在代码中数值验证。
3. **Gated DeltaNet 慢速回退**:文本模型 3/4 层为线性注意力(Gated DeltaNet)。安装 flash-linear-attention 后 prefill 从 ~1.7 s 降至 0.09 s(211 token),单步解码 0.065 s。
4. **逐 token 解码框架**:所有方法(直接回答/三种抑制方法/补救信号)共用同一手写贪心解码循环,直接控制每步输出分布,保证各方法在完全相同的输入、提示词与解码环境下配对比较。

## 3. 数据核实(LVIS v1)

### 3.1 来源与指纹

- 验证集标注:`https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip`,md5 `87734a7f895990b9552075d7ce723e27`,解压后 201,235,232 字节。来源为 LVIS 官方发布地址(与 TensorFlow Datasets 1.4.0 构建脚本所用一致)。
- 训练集标注:`https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip`(350,264,821 字节)。单连接下载过慢(约 10 KB/s),改用 24 路分片并行下载同一 URL 的同一文件后拼接,内容不变。**用途:验证集低频类别候选图像不足 600 张(仅 322 张),按任务书第六节扩大候选池;两组的候选池均优先取验证集,不足部分由训练集补足,逐样本记录来源。**
- 许可:COCO 图像遵循 Flickr 原始许可(COCO 汇总),LVIS 标注遵循 CC BY 4.0。仅用于学术评测。

### 3.2 字段结构核实(实测,非假设)

解析 `lvis_v1_val.json`,顶层键:`info, categories, annotations, images, licenses`。

- `categories[i]` 实际字段:`id, name, synonyms(list), def, synset, frequency, image_count, instance_count`。
  **频率标记字段确为 `frequency`,取值为单字母:`f`(高频档,405 类)/`c`(中频档,461 类)/`r`(低频档,337 类)。**
- `images[i]` 实际字段:`id, coco_url, flickr_url, height, width, license, date_captured, neg_category_ids, not_exhaustive_category_ids`。
  **图像直接地址字段确为 `coco_url`,指向 `http://images.cocodataset.org/{train2017|val2017}/xxxxxxxx.jpg`,即图像全部位于 COCO 2017。**
- `annotations[i]` 字段:`id, image_id, category_id, bbox, area, segmentation`。

分组规则:高频组 = `frequency=='f'` 的类别;低频组 = `frequency=='r'` 的类别。完全依据标注,无模型或人工判断。

### 3.3 图像选取条件(实现口径)

1. 目标类别实例是该图所有 LVIS 标注中面积最大者,且实例面积 ≥ 图像面积 15%(主体位置、明确可见的确定性代理)。
2. 该图在目标频率档内只出现一个类别(可含多实例)。
3. 图像下载成功且 PIL 可解码。

存在性判断的负样本:从**同一频率组**的其他类别中,选取属于该图 `neg_category_ids`(LVIS 标注方确认图中不存在)的类别替换构造,保证"不存在"判定有官方依据。

### 3.4 标准答案

类别 `name` + `synonyms` 全部同义词;另经 WordNet 上位词链接受直接上位词(如模型答 "bird" 而标注为具体鸟种时判正确)。判分为确定性字符串/词表规则,不使用语言模型评分。

## 4. 可用性结论

- 模型:满足条件,两模型并行可行。
- 数据:验证集标注已核实;训练集标注用于扩大低频候选池;图像按 `coco_url` 逐张下载。
- 依赖:项目 venv 内 transformers 5.17.0 可加载 Qwen3.5 并完成图像→文本生成;fla 与 Conv3d 等价改写使单样本前向进入 0.3 s 量级,主实验在时间预算内可行。

无阻塞项。
