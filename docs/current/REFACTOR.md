# 活动代码重构

## 迁移对象

目标仓库必须是Git工作树，且当前修改已经提交。迁移记录保存实际HEAD和 `code/stage3_engine.py` 的Git blob。预期blob为fc94a965cfc9b3bbba94ca2cc63aea3018c36fd6。

安装器把现有 `code/` 与README移动到 `archive/pre_refactor_<commit>/`，新活动代码安装至 `src/kdm/`。`outputs/`、`data/`、`docs/stage1`至`docs/stage9`和提交历史全部保留。无需重写旧结果。

新文档进入 `docs/current/`，配置进入 `configs/kdm/`，原始写作要求进入 `source_materials/kdm/`。历史阶段脚本不再位于活动代码搜索路径，旧科学记录仍可追溯。

## 提取内容

从原始模型文件通过AST提取FamilyModel与InternVLModel的构造、输入准备以及必要前向接口。所有旧训练和实验循环、旧评分规则、旧解码变体均不进入新模块。迁移器在写入前检查所有目标位置，遇到冲突终止并列出路径。

新解码模块分别实现VCD、M3ID、DoLa、DeCo以及SID参考接口；每个算子都能与独立公式单元检查。旧SID区间start/end解释修正为start/length，保留因果掩码，关闭接口时恢复原始attention.forward。模型共享的rope_deltas分别保存到各会话。

## 命令

```bash
python /项目内解压路径/scripts/install.py --project "$PWD" --package /项目内解压路径
python /项目内解压路径/scripts/install.py --project "$PWD" --package /项目内解压路径 --apply
PYTHONPATH=src python -m pytest -q
```

`--reviewed-source-change`只用于已经逐行审阅并记录变化的真实源版本差异。不要为了让安装通过而直接添加。

## 提交

归档迁移、算子与测试、模型接口、冻结协议、数据清单、完整普查、正式推理、机制测量、汇总与论文材料分别提交。每次提交说明修改的对象和原因。只执行普通push；推送冲突时保留本地结果并明确报告。
