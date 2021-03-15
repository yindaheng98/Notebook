# 【纯转载】从 git commit 中永久删除某个文件

## 背景

最近在帮别人看他的一个项目，项目通过 git 管理。在我 `git clone` 到本地的时候，发现这个工程目录十分大。经过分析，他将很多无谓的文件也 commit 上去，例如：

```
venv/
*.pyc
.idea/
```

等等目录，这些文件以及目录并没有必要上传，于是我便和他沟通，征得他同意后开始大改...

**一般情况下，并不建议执行 `git push -f` 这个操作**

## 操作

如果我新增一个 commit 把这些不必要的文件删掉，并且使用 `.gitignore` 来主动忽略，这是可行的。不过这些文件依然会在 commit 历史中出现，别人 `git clone` 的时候也会被下载下来，因此打算从 commit 历史中完全删掉这些文件，就像他们从来没有出现过一样。

下列代码整理自互联网

```
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch venv/ -r' --prune-empty --tag-name-filter cat -- --all
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now
git push --force --verbose --dry-run
git push --force
```

其中，要删去的是目录 `venv/` ，由于是目录所以我加上了 `-r` 参数。

## 缺陷

正如上面所提到的，一定要尽可能少用这个功能，虽然他可以减少我们 git 目录的大小，但是这会对其他协作者产生极大的影响。

所有受到影响的 commit 的 ID 都会被重写，另外如果 commit 像我这样有 GPG 签名的话，就无法执行这个操作了。