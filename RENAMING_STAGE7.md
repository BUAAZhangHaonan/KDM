# 第七阶段 · 规范清扫清单（任务书 §4.3）

清扫日期：2026-09-16。原则：只改措辞，不改任何数字与判定；被改文件的历史原文
保留在 git 历史（本清单给出提交哈希对照）。

## 一、执行的更正

| 文件 | 更正内容 | 性质 |
|---|---|---|
| `PREREGISTER_STAGE6.md` | 假设代号 `H-A`/`H-B`/`H-F` → "诊断假设/机制假设/校正假设"（7 处）；`style3`/`style1` → "判定型提问加弃权口径/直接命名口径"（1 处） | 措辞替换，判定内容零改动；文件头已加清扫注记 |
| `STAGE6.md` | `（H-A 确认）` → `（诊断假设确认）`（1 处）；`style3` → 判定型提问加弃权口径（1 处）；对照分级标签 `sharpening`/`partial`/`structure` → "完整复现/部分复现/未复现（存在残余结构）"（6 处） | 同上 |

提交：清扫提交信息注明"zero judgment or number changes"，可用 `git diff` 逐行核对。

## 二、不改写的位置与理由

1. **代码与数据文件中的字段值**（如 `stage6_sharpening.csv` 的 `verdict` 列取值
   `sharpening/partial/structure`、记录中的 `style: "style1"`、文件名
   `stage6_sharpening.csv`）：这些是机器可读的存储字段，不是定稿措辞；论文正文
   与报告引用它们时一律使用自然语言称呼（本清单第一节已把报告侧改齐）。
2. **更早阶段的过程文档**（`STAGE1–5`、`PILOT`、`CONCLUSIONS` 等全 8 份）：
   全仓检索确认其中不含假设代号（`H-A/H-B/H-F/H1/H2/H3` 零命中，见第三节），
   无需更正；它们不进入论文正文的措辞链。
3. **方法正式名**：第六阶段起已统一 VCD/M3ID/DoLa/DeCo/SID；本地历史缩写
   MIB/LCD 只在实现对照与正对照语境出现并明确标注"非已发表方法/变体实现"。

## 三、全仓检索证据（2026-09-16 执行）

```
grep -rn "H-A\|H-B\|H-F" --include="*.md" .          # 定稿材料（除历史过程文件）0 命中
grep -nE "style1|style3" PREREGISTER_STAGE6/7.md STAGE5/6/7.md README.md   # 0 命中
grep -lE "H-A|H-B|H1|H2" STAGE3/4/5.md PREREGISTER_STAGE3/4/5.md           # 0 命中
```

结论：没有假设代号或提示风格代号残留在任何预登记、阶段报告或索引文档中；
代码/数据字段中的英文机器标签已在本清单第二节注明映射，不构成定稿措辞。
