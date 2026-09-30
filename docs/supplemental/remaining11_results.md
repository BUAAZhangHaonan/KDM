# 剩余11模型补充结果

2026年9月30日评分检查点 `v7_luna_reference1220_20260930` 已接入63,572条正式Food-101评分及351,162条独立试答评分记录，共414,734条。正式登记为832个条件、2,016,768条样本。11个Direct/UNKNOWN条件的主正确性与弃权均全量已决，共26,664条；加上Phi35的三个完整方法条件，可分析14个条件、33,936条。其余818个条件仍有生成或标注缺项。

分析固定使用Food-101 eval 2,424张图像、101个类别各24张；dev参考同样每类24张。主正确性为 `canonical_name_in_primary_score`，完整提取名称的字面分数独立保存。现有五模型的正式评分、表格和正文继续保持原样。

| 完整Direct/UNKNOWN条件 | 正确数 / 2424 | 主正确率 | 弃权数 / 2424 | 弃权率 | 回答覆盖率 |
|---|---:|---:|---:|---:|---:|
| gemma3_12b | 1381 | 56.97% | 1 | 0.04% | 99.96% |
| glm46v | 1308 | 53.96% | 6 | 0.25% | 99.75% |
| internvl35_8b | 623 | 25.70% | 37 | 1.53% | 98.47% |
| llava15_7b | 346 | 14.27% | 1 | 0.04% | 99.96% |
| llava15_13b | 522 | 21.53% | 77 | 3.18% | 96.82% |
| llava16_vicuna | 527 | 21.74% | 249 | 10.27% | 89.73% |
| minicpm45 | 1493 | 61.59% | 13 | 0.54% | 99.46% |
| onevision | 626 | 25.83% | 1 | 0.04% | 99.96% |
| phi35 | 687 | 28.34% | 22 | 0.91% | 99.09% |
| qwen35_9b | 1181 | 48.72% | 651 | 26.86% | 73.14% |
| qwen3vl | 1158 | 47.77% | 517 | 21.33% | 78.67% |

这11个完整条件显示不同的作答与弃权水平，当前数据支持上述描述性观察。Gemma3-12B与GLM-4.6V的既有Direct边界已完成追加裁定。Gemma3-12B与InternVL的字面敏感性分数各有3条未决，其主正确性与弃权均已全量决定。

Phi35的M3ID、DoLa、DeCo在 `kind=main`、主标记和参考标记均为UNKNOWN、双侧guided=true、replicate=0的条件下已全量已决。三个比较均与相同主提示Direct逐样本配对，prompt SHA和seed差异为0。下表的置信区间使用预注册101类别配对cluster bootstrap；单位为百分点。

| Phi35条件 / 2424 | 正确数 | 弃权数 | 主正确率 | 相对Direct差值及95%区间 | 新增正确 / 原正确丢失 |
|---|---:|---:|---:|---:|---:|
| Direct | 687 | 22 | 28.34% | — | — |
| M3ID | 707 | 5 | 29.17% | +0.83 [+0.17, +1.49] | 31 / 11 |
| DoLa | 681 | 13 | 28.09% | −0.25 [−4.04, +3.42] | 197 / 203 |
| DeCo | 529 | 670 | 21.82% | −6.52 [−9.74, −3.30] | 68 / 226 |

这些数值仅覆盖表内三个完整注册条件。其余模型、标记组合及instruction-preserving、CDA、SID条件仍有缺项。M3ID对应的保留Direct原弃权复制控制已完成2,424条选择：按Direct弃权选回原回答22次，正确706次（29.13%），弃权22次。选择规则仅读取Direct是否弃权。Phi35尚缺全量类别排名，参考连接为0/2,424，因此这三个方法与控制的参考弃权召回、合理弃权保留及依赖参考的结论保留空值。原始身份、来源cohort和逐条选择同时保留。

参考GT统一为正确类别排名大于1且10次独立试答正确次数为0。53,328个dev/eval登记参考键中25,342个已完成，比v5新增930个；27,986个仍缺排名、独立试答或主正确性裁定。下表精确列出覆盖。

| 模型 | 完整参考dev / 2424 | 完整参考eval / 2424 | eval参考缺项 |
|---|---:|---:|---:|
| gemma3_12b | 192 | 169 | 2255 |
| glm46v | 300 | 296 | 2128 |
| internvl35_8b | 0 | 0 | 2424 |
| llava15_7b | 2424 | 2424 | 0 |
| onevision | 2424 | 2424 | 0 |
| minicpm45 | 1630 | 1667 | 757 |
| phi35 | 0 | 0 | 2424 |
| qwen3vl | 2424 | 2424 | 0 |
| qwen35_9b | 847 | 849 | 1575 |
| llava15_13b | 0 | 0 | 2424 |
| llava16_vicuna | 2424 | 2424 | 0 |

llava15_7b、onevision、qwen3vl、llava16_vicuna的dev与eval参考已全量闭环。其Direct/UNKNOWN条件的参考指标如下；其余模型参考指标继续保留空值。TP、FP、FN均使用完整2,424样本及对应参考阳性集合，单次弃权的100%精确率同时保留实际计数。

| Direct模型 | 参考阳性数 | TP / FP / FN | 弃权精确率 | 弃权召回率 |
|---|---:|---:|---:|---:|
| llava15_7b | 1296 | 1 / 0 / 1295 | 100.00% | 0.08% |
| onevision | 813 | 1 / 0 / 812 | 100.00% | 0.12% |
| qwen3vl | 364 | 188 / 329 / 176 | 36.36% | 51.65% |
| llava16_vicuna | 1179 | 200 / 49 / 979 | 80.32% | 16.96% |

参考缺项分别保留缺失排名、缺失attempt、已生成attempt但正确性未决的数量，历史accepted参考字段另列。是否应该弃权的额外交叉验证继续后置。

实际读取 `data/reference_gt/legacy_accepted_reference_gt.jsonl` 的14,544个唯一模型—样本键：llava16_mistral、minicpm26、qwen25vl各4,848条，dev/eval各2,424条。剩余11模型在此文件的历史accepted覆盖均为0，对应参考值保留null。机读回执位于独立VizWiz评分检查点的 `legacy_accepted_reference_coverage.json`，保留原文件路径、SHA及逐模型计数。

补充分析入口为 `workflows/supplemental/remaining11/analysis.py`。输入目录为 `outputs/supplemental/remaining11/run_20260930_140337/scores/v7_luna_reference1220_20260930`，检查点为 `outputs/supplemental/remaining11/run_20260930_140337/analysis/checkpoint_v7_20260930_1845`。每次接续使用独占检查点目录，保存逐样本合并表、832条件覆盖表、参考表、964个登记配对的完成与缺项、440个登记复制控制的覆盖及逐样本选择。当前3个配对、1个复制控制完整；961个配对和439个控制仍待补齐。v5及更早产物继续保留。

配对计算复用已发布主结果的类别cluster bootstrap函数和复制控制统计函数，固定2,000次重采样、随机种子20260929。同主提示Direct按model、sample、main_marker、guided、replicate连接，并检查主提示SHA和seed。完整instruction-preserving条件同时比较对应base方法与保留Direct原弃权的复制控制；选择分支仅使用Direct是否弃权。原有合理弃权集合由Direct弃权且参考GT为真定义，纠正收益和纠正保留率保留完整分母。

图表位于检查点的 `figures/direct_unknown.png`、`figures/unknown_method_effects.png` 及同名PDF。`cases.jsonl` 包含63条真实回答案例及其问题、样本、分数、原始来源、行号、身份和逐行SHA。例：llava15_13b在 `food101:apple_pie_eval_025.jpg` 回答“Apple pie”，主正确性为1；同模型在 `food101:apple_pie_eval_024.jpg` 回答“Cheesecake”，主正确性为0；在 `food101:apple_pie_eval_032.jpg` 回答“UNKNOWN”，弃权为真。实际验证核对14份输出SHA、全部14个条件的101×24配额，以及33,936条真实记录的身份配对和选择不变性；两份PNG均已进行视觉检查。

正式缺失样本键为1,953,196条，连同已生成条件中的未决正确性与弃权数量，保存在检查点 `condition_coverage.csv` 和 `missing_keys.jsonl.gz`。完整方法矩阵仍待接续。VizWiz的官方答案与人工可回答性结果另见 `remaining11_vizwiz_results.md`，使用其独立登记和分母。
