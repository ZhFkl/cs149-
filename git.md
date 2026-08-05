# Git 常用指令速查

## 目录

- [1. 创建并初始化本地仓库](#1-创建并初始化本地仓库)
- [2. 绑定远程仓库(GitHub 等)](#2-绑定远程仓库github-等)
- [3. 日常开发工作流(最常用)](#3-日常开发工作流最常用)
- [4. 拉取远程更新](#4-拉取远程更新)
- [5. 分支操作](#5-分支操作)
- [6. 查看历史](#6-查看历史)
- [7. 撤销与回退(小心使用)](#7-撤销与回退小心使用)
- [8. 暂存手头工作(stash)](#8-暂存手头工作stash)
- [完整示例](#完整示例)

---

## 1. 创建并初始化本地仓库

```bash
mkdir my-project          # 新建目录
cd my-project
git init                  # 初始化 git 仓库(生成 .git 目录)
```

## 2. 绑定远程仓库(GitHub 等)

```bash
# 先在 GitHub 上创建一个空仓库,然后绑定
git remote add origin git@github.com:你的用户名/仓库名.git

# 查看已绑定的远程仓库
git remote -v

# 如果绑定错了,可以修改
git remote set-url origin git@github.com:你的用户名/新仓库名.git

# 删除绑定
git remote remove origin
```

## 3. 日常开发工作流(最常用)

```bash
git status                     # 查看当前状态(哪些文件改了、没提交)
git add file.txt               # 暂存单个文件
git add .                      # 暂存所有改动
git commit -m "说明这次改了什么"   # 提交到本地仓库
git push -u origin main        # 第一次推送(建立分支追踪)
git push                       # 之后的推送直接用这个
```

## 4. 拉取远程更新

```bash
git pull                       # 拉取远程并合并到本地(= fetch + merge)
git fetch                      # 只拉取,不合并(先看看远程有什么变化)
git merge origin/main          # 再手动合并
```

## 5. 分支操作

```bash
git branch                     # 查看本地分支(* 是当前分支)
git branch -a                  # 查看所有分支(含远程)
git branch dev                 # 创建 dev 分支
git switch dev                 # 切换到 dev 分支
git switch -c dev              # 创建并切换到 dev(一条命令)
git switch main                # 切回主分支
git merge dev                  # 把 dev 合并到当前分支
git branch -d dev              # 删除已合并的分支
git branch -D dev              # 强制删除分支
```

> 老版本用 `git checkout dev`,等价于 `git switch dev`。

## 6. 查看历史

```bash
git log                        # 提交历史
git log --oneline              # 简洁版(一行一个提交)
git log --oneline --graph      # 带分支图,看合并结构很直观
git diff                       # 查看未暂存的改动
git diff --staged              # 查看已暂存的改动
```

## 7. 撤销与回退(小心使用)

```bash
git restore file.txt           # 丢弃文件的工作区改动(未 add)
git restore --staged file.txt  # 取消暂存(已 add 但未 commit)
git commit --amend             # 修改上一次提交(消息或内容)
git reset --soft HEAD~1        # 撤销上一次提交,改动保留在暂存区
git reset --hard HEAD~1        # 彻底丢弃上一次提交及改动(不可恢复!)
git revert <commit-id>         # 生成一个反向提交来撤销,安全,适合已推送的提交
```

## 8. 暂存手头工作(stash)

```bash
git stash                      # 临时存起当前未提交的改动
git stash pop                  # 恢复最近一次的 stash 并删除记录
git stash list                 # 查看所有 stash
```

---

## 完整示例

```bash
# 1. 本地初始化
git init
echo "# My Project" > README.md
git add README.md
git commit -m "Initial commit"

# 2. 绑定 GitHub 上的空仓库并推送
git branch -M main
git remote add origin git@github.com:zhangsan/my-project.git
git push -u origin main

# 3. 开分支开发新功能
git switch -c feature-login
# ... 写代码 ...
git add .
git commit -m "Add login page"
git push -u origin feature-login

# 4. 合并回主分支
git switch main
git merge feature-login
git push
```

**记住一个口诀**:`add`(暂存)→ `commit`(提交)→ `push`(推送),这就是 Git 日常 90% 的操作。
