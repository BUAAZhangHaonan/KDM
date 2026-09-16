# 第八阶段 · git 历史与大对象核查报告（任务书 5.1）

核查日期：2026-09-15。核查对象：`/home/g203-4028/projects/knowledge-deficit-mitigation`
（远端 `ssh://git@ssh.github.com:443/BUAAZhangHaonan/KDM.git`，master）。

## 1. `venv_sid` 是否残留在最终推送历史中

- `git log --all --oneline -- venv_sid` → **零输出**：从任何引用（含分支与标签）
  可达的历史中没有对该目录的任何引用。第七阶段的处理（`git add -A` 误提交 →
  推送被拒 → `git rm -r --cached venv_sid` + `.gitignore` + amend 后重推）在
  推送前已把该提交从分支历史上移除；被拒推送的对象经 quarantine 丢弃，未进入
  远端。
- 结论：**公开历史干净，无需历史重写。**

## 2. 仓库体积

- `.git` 目录合计 227 MB（两个 pack：217 MB + 9.6 MB，另 11 个松散对象）。
- 可达历史对象（`git rev-list --objects --all`，未压缩合计约 107 MB，压缩后为主
  pack 的一部分）：`outputs/` 占 103.4 MB——即各阶段 raw jsonl 证据文件在多次
  提交中的累积版本；`data/` 2.2 MB、`code/` 0.7 MB、其余可忽略。
- 可达历史中最大的 8 个对象全部是 `outputs/raw/*.jsonl`（3.1–4.6 MB/个），
  属于应随论文公开的实验证据。

## 3. 本地不可达对象（不随克隆/推送传播）

- `git fsck --unreachable` 显示本地存在不可达大对象；抽样验证为 ELF 64 位共享
  库（venv_sid 环境的二进制），未压缩合计 823 MB，压缩后约占主 pack 的一半。
  它们来自被 amend 撤销的那个提交，仅存于本地仓库。
- 可选清理方案（**仅上报，未执行**）：`git gc --prune=now` 可将其删除，
  本地 `.git` 将缩至约 100 MB；亦可等待两周保留期后自动清理。因任务书禁止
  擅自重写/清理历史，此处只记录方案。

## 4. filter-repo 事件留痕复核

- 第六阶段曾因 filter-repo 丢失 nltk wordnet 缓存，恢复方式（重新下载官方
  nltk_data 同名语料）已记录于 `STAGE6.md` §5 第 5 条，本阶段复核确认在档。

## 5. 结论

最终推送历史零 `venv_sid` 残留；`.git` 体积主要由合法证据文件（outputs/raw）
与本地不可达的已撤销二进制构成；后者不影响远端与公开克隆，清理方案已上报待定。
