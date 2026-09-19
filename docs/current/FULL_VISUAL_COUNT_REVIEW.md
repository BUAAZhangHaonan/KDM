# 全量原生视觉 token 数与固定 SID rank100 可计算性

本检查覆盖完整9,167条原清单：Food101 dev/eval各2,424；VizWiz dev818、eval3,501。没有筛图、改变split/group、图像尺寸、processor参数、选层或rank，没有运行GPU/权重前向，也没有改变任务selection或其它四个方法。**Qwen2.5-VL、Qwen3-VL与GLM的完整VizWiz条件不能按同一固定SID rank100协议运行**：各有真实输入视觉位置不足100。不能只保留大图计算该条件的SID指标。其Food完整条件的数量下界满足100，但仍需要对应实际SID接口通过。GLM的单图SID真实数值核验正在进行，不以本count检查替代。

MiniCPM2.6和4.5在Food与VizWiz都存在64个原生视觉位置；因此两个完整数据任务均存在固定topk100未定义输入。Phi全量最低457，无此数量障碍。下列判断是**前置原生输入约束**，与干预结果无关；数量足够只是必要条件，Gemma原生mask差异与Qwen3.5第二层结构问题不会因token足够消失。

## 实际方法与证据边界

逐一读取全部图像头（零失败），得到181种真实(height,width)。调用每模型spec环境实际安装processor的原生计数函数，保留runtime类、完整processor配置、源文件SHA、完整清单/header SHA及命令。Qwen2.5/Qwen3使用其`_get_num_multimodal_tokens -> get_number_of_image_patches -> smart_resize`；该smart_resize与实际预处理相同，随后patch/merge计算与实际image_grid_thw一致。GLM实际类是 **Glm46VProcessor/Glm46VImageProcessor**，不是从model_type名称猜测Glm4v处理器；其smart_resize显式使用temporal_factor=2，并和实际resize方法共享同一公式。

Qwen两个模型与GLM先检查8张真实尺寸边界图：最小/最大计数、100上下最近的实际尺寸、最小/最大边长/面积/纵横比去重；调用未改参数的`FamilyModel.build`，核对input_ids中的真实image id数量与image_grid_thw计数。随后发现377张VizWiz带EXIF orientation6，补全全部头部方向检查，并验证这些三模型的原生计数函数对全部181尺寸的高宽交换不变，确认EXIF候选图已在原有边界选择中，按ID去重没有增加图数；动态模型最终各8张实际processor核对均一致。Mini、Phi走其现有remote adapter的直接PIL/RGB路径；EXIF候选也已在原有选择中，最终分别6、7张，实际image_bound/负token计数仍一致。完整记录中`gpu_initialized=false`，没有加载模型权重。

Mini计数直接调用实际image processor的`get_slice_image_placeholder`并统计其原生unk视觉占位符；源函数由实际get_sliced_grid确定原图与slice数量，各段64。真实边界核对读取`inputs.image_bound`的半开区间并集，**不把段间文本分隔符计入视觉数**。Phi使用checkpoint实际`calc_num_image_tokens_from_image_size`，该函数逐步复现HD_transform的旋转/缩放/padding尺寸；实际边界再核对`-1e9 < input_id < 0`的native embedding写入位置，num_crops保持4。

这不是面积除常数的近似估算，也不是对少量模型生成结果做外推。精确计数取决于已记录的原生shape函数和相同预处理配置；真实边界核对验证这些函数与实际processor路径一致。未来processor/参数/清单改变，应重新检查。此记录只建立数量条件，不声称对9,167图全部执行过模型attention或SID logits。

## 全部dataset/split计数

|模型|dataset|split|完整n|最小视觉token|最大视觉token|<100样本数|
|---|---|---|---:|---:|---:|---:|
|qwen25vl|food101|dev|2424|162|324|0|
|qwen25vl|food101|eval|2424|180|324|0|
|qwen25vl|vizwiz|dev|818|4|6417|7|
|qwen25vl|vizwiz|eval|3501|4|6417|23|
|qwen3vl|food101|dev|2424|128|256|0|
|qwen3vl|food101|eval|2424|128|256|0|
|qwen3vl|vizwiz|dev|818|70|4860|10|
|qwen3vl|vizwiz|eval|3501|64|4860|37|
|glm46v|food101|dev|2424|162|324|0|
|glm46v|food101|eval|2424|180|324|0|
|glm46v|vizwiz|dev|818|12|6030|7|
|glm46v|vizwiz|eval|3501|9|6030|23|
|minicpm26|food101|dev|2424|64|192|873|
|minicpm26|food101|eval|2424|64|192|857|
|minicpm26|vizwiz|dev|818|64|640|71|
|minicpm26|vizwiz|eval|3501|64|640|287|
|minicpm45|food101|dev|2424|64|192|873|
|minicpm45|food101|eval|2424|64|192|857|
|minicpm45|vizwiz|dev|818|64|640|71|
|minicpm45|vizwiz|eval|3501|64|640|287|
|phi35|food101|dev|2424|457|757|0|
|phi35|food101|eval|2424|757|757|0|
|phi35|vizwiz|dev|818|757|757|0|
|phi35|vizwiz|eval|3501|457|757|0|

全部<100的样本ID、split、原图(height,width)和精确token数均在各JSON的`below100`，没有只保存示例。Qwen2.5/GLM各30张，Qwen3共47张；Mini每版2,088张，其中Food1,730、VizWiz358。原始manifest未改。

## 其余固定架构的来源下界

`outputs/verification/sid_fixed_visual_lower_bounds.json`保存实际config/processor/model-pack源码路径、SHA、关键行和每个dataset/split的完整n。这里提供源码保证下界，不伪造全量观测最大值。

|模型|保证下界|原生来源理由|数量之外的限制|
|---|---:|---|---|
|LLaVA1.5-7B/13B|576|336×336固定crop；14 patch；CLS+1与default selector -1抵消|仍使用各自实际SID接口证据|
|LLaVA1.6-Mistral/Vicuna|至少576|anyres pack保留576个global/base_image_feature，再拼接unpad细图与newline；不会用细图替换global|实际各环境源码分别指纹记录|
|OneVision|至少729|注册num_image_tokens729/full策略，384//14的global feature保留于anyres拼接|仍需原生decoder/SID接口|
|InternVL3.5-8B|至少256|448/14的32×32patch经0.5²下采样为256/tile；dynamic_preprocess至少1tile，thumbnail只增加|当前明确授权双卡factory已分别通过native16与SID，单卡OOM历史保留|
|Gemma3-4B/12B|至少256|注册image_seq_length256；原图始终有256个位置，pan-and-scan只增加image块|原生bidirectional/sliding mask不等于固定官方causal定义，不能据此标SID适用|

Qwen3.5-4B/9B的固定第二层为linear_attention而非本协议所需self_attn，已由独立结构记录判定固定参考不成立，不用token下界掩盖这一点。Phi虽有原生理论至少313的HD结构下界，本清单实际最小457已通过完整尺寸枚举及真实processor边界核对。

## 小型证据与复现

- `outputs/records/sid_visual_image_headers.json`：9,167条header/181尺寸，约559KB；`sid_image_exif_orientation_check.json`保存方向统计与377个非默认ID。
- `outputs/verification/{key}_sid_full_visual_count.json`：上表6模型的完整分组、所有不足100列表、181种尺寸计数、实际边界样本/输入id计数/grid、spec、脚本与processor源身份；同名log保留实际命令输出。
- `outputs/verification/{qwen25vl,qwen3vl,glm46v}_visual_count_source.txt`：执行时读出的实际类、shape utility与预处理源码，避免混用其它模型版本公式。
- `outputs/verification/sid_fixed_visual_lower_bounds.json`：其余8个固定下界配置/来源逐项指纹。

在项目根复现某模型（只CPU；MODEL_PYTHON取该spec environment_python）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HOME="$PROJECT/cache/hf" XDG_CACHE_HOME="$PROJECT/cache/xdg" TMPDIR="$PROJECT/cache/tmp" "$MODEL_PYTHON" verification/check_full_visual_counts.py qwen25vl
```

脚本不接受resize/rank覆盖参数。先读取既有全量header记录并核验当前manifest SHA，再运行原生计数与真实边界。各记录的command字段给出实际环境Python；此检查未改任何runtime spec或方法计划。

最终边界核对与脚本身份：

- qwen25vl: 8 real processor checks passed; script SHA256 `5e0fefa95aec4e415f2839afff0f29d41401d1d80477ed453a8fd3972d6dc84c`.
- qwen3vl: 8 real processor checks passed; script SHA256 `5e0fefa95aec4e415f2839afff0f29d41401d1d80477ed453a8fd3972d6dc84c`.
- glm46v: 8 real processor checks passed; script SHA256 `5e0fefa95aec4e415f2839afff0f29d41401d1d80477ed453a8fd3972d6dc84c`.
- minicpm26: 6 real processor checks passed; script SHA256 `5e0fefa95aec4e415f2839afff0f29d41401d1d80477ed453a8fd3972d6dc84c`.
- minicpm45: 6 real processor checks passed; script SHA256 `5e0fefa95aec4e415f2839afff0f29d41401d1d80477ed453a8fd3972d6dc84c`.
- phi35: 7 real processor checks passed; script SHA256 `5e0fefa95aec4e415f2839afff0f29d41401d1d80477ed453a8fd3972d6dc84c`.

派生文件修订说明：增加全量EXIF orientation记录与三个动态模型181种尺寸交换不变性检查后，在同名派生count JSON/log路径重新执行写入。拟加入的EXIF候选`vizwiz:VizWiz_val_00000003.jpg`已存在于全部六模型的原选择（Qwen/GLM/Mini第2项、Phi第4项），按ID去重没有增加样本；最终仍为动态模型各8项、Mini各6项、Phi7项。重新计算确定性selector并逐ID对照当前记录，六模型的边界ID集合与原selector完全一致。此前代理声称新增一项并得到9/7/8项的文字报告不正确，此处明确撤回；JSON的实际观察数没有相应增加。

当前边界观察来自扩充脚本重新执行，不能冒称初次执行文件的逐字节留存。初版JSON/log及脚本未单独另存，也未进入Git，因此无法从仓库恢复其原始字节/脚本SHA或调用日志。仓库内可审计的命令、脚本SHA和全部观察以当前扩充记录为准。此次说明不改变全量计数与SID任务适用性结论。后续资源边界核验采用新版本不可覆写输出，不覆盖本批count记录。
