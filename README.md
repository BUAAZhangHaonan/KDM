# KDM

KDM 的核心研究代码包含多模态生成、VCD、M3ID、DoLa、DeCo、SID、CDA、指令保持方法、完整回答评分、参考真值处理和统计分析。各方法按模型架构及登记条件运行，适配范围由对应入口和验证依据确定。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/kdm/` | 解码算子、模型适配、任务矩阵、来源验证、完整回答评分与分析 |
| `configs/kdm/` | 模型目录、类别别名、方法适用规则 |
| `configs/supplemental/` | 原生基线的科学条件定义 |
| `configs/examples/` | 使用既有权重的配置示例 |
| `environments/` | 不同模型家族的环境定义 |
| `workflows/formal/` | 冻结面板生成及来源验证 |
| `workflows/main_results/` | 参考真值、正式评分、对照和汇总分析 |
| `workflows/paper_core/` | 正式原生与指令保持实验所需计算、适配和分析 |
| `workflows/supplemental/` | 扩展模型、正式登记验证及新增原生 M3ID 入口 |
| `workflows/general_vqa_direct/` | General VQA 输入准备、生成和评分 |
| `workflows/hallusion_blind/`、`hallusion_reference/`、`hallusion_h100/` | Hallusion 完整回答、独立参考和 H100 协议入口 |
| `LICENSES/` | 第三方许可证；来源说明同时保留在对应源码中 |

部分文件名保留原实验命名，因为其他正式入口仍导入其计算或验证函数。任务启动、收集和转运的旧命令入口已从纯验证模块中移除。

## 安装和入口

基础包使用 Python 3.11 或更新版本：

```bash
python -m pip install -e .
kdm --help
python workflows/formal/generate.py --help
```

分析功能可安装 `python -m pip install -e '.[analysis]'`。模型推理使用该模型登记的既有 Python/CUDA 环境、权重目录、精度和物理 GPU。`environments/` 给出已有模型家族的依赖定义，实际运行仍须满足对应入场证明。配置示例中的路径和版本字段须填写实际值。

新增的 VizWiz 原生 M3ID 使用固定 512 样本面板。`workflows/supplemental/native_m3id_viz_20261010/run.py` 是 4028 core4 入口；`run_6403.py` 和 `run_k100.py` 保留另外两台机器的实际计算版本。三个入口的架构适配和资源身份不同，须使用各自的 SOURCE、固定样本列表、输入、身份和入场证明。历史登记引用的原 `run.py` 名称及源码 SHA 保存在原运行目录中，重放登记任务使用其原绑定目录。

## 本地研究资产

实验原始输出、输入图片、真值和标签、评分表、注册 SOURCE、host registry、模型权重、论文与运行回执都作为本地研究资产保存，不进入本仓库的当前代码树。入口需要这些本地输入；代码克隆本身不提供实验数据。

4028 项目中的 `assets/merged/` 保存从既有清点合并的有效资产；`assets/current/<来源>/` 保存本次新增实验的登记材料及完整原始行截点；`provenance/consolidation_20261010/` 保存来源映射、校验和本次合并报告。按 SHA 复用相同内容，模型权重继续使用原目录。

实验仍在进行，完整行截点不代表生成、语义标注或评分全部完成。原运行目录的源码保持原字节；本次核心提交通过隔离工作树完成。历史冻结面板要求其原方法和源码 SHA，不能将当前源码替换到历史运行目录后放宽验证条件。
