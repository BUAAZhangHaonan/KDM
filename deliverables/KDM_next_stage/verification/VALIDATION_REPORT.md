# 本地执行与核对记录

## 执行环境与结论

验证日期：2026-09-16。Python 3.13.5，NumPy 2.3.5，pandas 2.2.3，Matplotlib 3.10.8，Pillow 12.3.0，PyMuPDF 1.26.7，pytest 9.0.2，torch 2.10.0+cpu。CUDA不可用。环境完整JSON见`environment.json`。

最终测试命令为`python -m pytest -q tests --disable-warnings`，结果 **31 passed in 4.26s**，日志见`pytest_final.txt`。所有七个绘图函数实际执行，输出七组PNG/PDF；每个PDF在PyMuPDF中重新渲染，逐图查看并核对标签、图例与来源标记。页数、图像尺寸、PDF哈希和字体类型见`figure_validation.json`。

真实模型的推理耗时、3090显存和VLM接口兼容性不在本地测试结果中。没有运行真实第九阶段模型实验，没有连接服务器运行命令，也没有推送远程提交。

## 计算模块的覆盖

| 模块 | 实际执行的检查 | 未声称完成的事项 |
|---|---|---|
| `stage9_geometry.py` | 14,400个随机单步强度判定与直接argmax比较；边界、同分、支持排除、序列求交集、见证；对数概率恒等式与有限差分；严格JSON | 不把这些数值测试当作VLM上的新发现 |
| `stage9_common.py` | 配对转移、同分覆盖率、序列分数、ECE、类别自助、JSON错误及路径限制 | 不反向生成真实逐样本数据 |
| `stage9_analyze.py` | 种子912合成记录经完整统计路径，32行模型—子集—方法—分数结果和160条曲线记录 | 四模型真实缓存尚待服务器读取 |
| `stage9_prepare.py` | 项目内合成图像与清单，独立分层、确定性选择、路径检查 | Food-101图片未下载到本环境 |
| `stage9_run.py` | CPU模拟引擎贯通输入条件、解码、教师强制、见证生成、原图缓存复用、断点续跑与直接token一致性检查 | 真实S6Model/3090执行未进行；模拟引擎不是模型替代基线 |
| `stage9_finalize.py` | 完整矩阵核对、证据曲线、配对变化、跨条件效应与指定名称类别输出 | 无真实新增经验结果 |
| `stage9_pope.py` | 模拟网络响应的字节下载、Git blob指纹、图像解码、正负平衡、两个模型流程与汇总 | 公网下载与真实VLM回答未进行 |
| `viz/stage9_figures.py` | 七个函数逐一输出PNG/PDF，来源标记、单页、非Type3字体及渲染核对 | 模拟示例不是第九阶段结果 |

五个带CLI的运行脚本均执行`--help`确认参数可解析，见`cli_help.json`。只有计算库没有独立CLI。图形脚本除CLI外执行全部绘图函数。

## 图与数据的对应关系

01、02使用`data_sample/stage6_core_selected.csv`：从固定提交真实表格转录的32行选择列。01只展示其中16个高准确率组合，02展示全部32个组合。01中的纠正/致错数量与600张图像分母和端点准确率逐行一致。

03使用`data_sample/correction_selected.csv`：四条真实校准轨迹，仅作选择性示例，不代表所有组合。

04使用代码中三个显式三token分布。三个指定目标的区间分别为空、严格大于0.5、为空；第二个区间在图中只显示0—4。相等时按原token索引处理。原分数、参考分数和精确区间保存在对应JSON文件中。

05的数据是合成逐样本记录经实际统计代码生成的`synthetic_risk_coverage.csv`。06的数据是固定的版式验证曲线`synthetic_evidence_accuracy.csv`。07的数据是固定的版式验证分类记录`synthetic_name_reachability.csv`。后两者不来自CPU模拟模型的经验趋势，也没有预期结果的含义。

## 图形检查后的修改

首次导出暴露了NumPy布尔边界不能被严格JSON序列化的问题，已在`Interval.to_dict()`中转换为Python标量并新增测试。首图标题与图例发生重叠，已调整垂直间距并重绘。置信度图的长纵轴标题改为两行，图例移开数据点。最终交付的是修改后重新运行、重新渲染的图。

所有最终图均检查：无裁切标题，无覆盖数据的图例，无乱码，PDF保持矢量文本，来源标记清楚。合成图保留`SIMULATED FIXTURE - NOT KDM RESULTS`，数学示例保留`MATHEMATICAL EXAMPLE`。

## 再现方法

在交付包目录运行测试与08文档中的绘图命令即可复现本地核对。完整模拟流程可运行`python tests/build_fixtures.py`；它在包内`verification/fixture_project/`生成测试项目，若已有同名目录会拒绝混写。包内只保留精简的样例CSV与验证记录，不附带这个临时测试项目。服务器运行测试时按ZCode提示词指定项目内临时路径。
