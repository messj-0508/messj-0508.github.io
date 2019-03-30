---
title: Git 撤销合并操作
date: 2019-01-31 13:01:23
tags: [git]
---
**利用Merge操作合并分支时，可能会出现一些错误，需要撤销合并。这里介绍如何撤销已经上传至github远程仓库的方法**

当你使用 git merge 合并两个分支，你将会得到一个commit。执行 git show 之后，会有类似的输出：

``` bash
commit 19b7d40d2ebefb4236a8ab630f89e4afca6e9dbe
Merge: b0ef24a cca45f9
......
```
其中，Merge 这一行代表的是合并所用到的两个分支(parents)。举个例子，通常，我们的稳定代码都在 master 分支，而开发过程使用 dev 分支，当开发完成后，再把 dev 分支 merge 进 master 分支：

``` bash
a -> b -> c -> f -- g -> h (master)
           \      /
            d -> e  (dev)
```
g 是 merge 后得到的代码，g 的两个 parent 分别是 f 和 e。

当你撤销合并，需要添加-m参数来指定撤销合并至哪条分支(parent)。
在你合并两个分支并试图撤销时，Git 并不知道你到底需要保留哪一个分支上所做的修改。从 Git 的角度来看，master 分支和 dev 在地位上是完全平等的，只是在 workflow 中，master 被人为约定成了「主分支」。

于是 Git 需要你通过 m 或 mainline 参数来指定「主线」。merge commit 的 parents 一定是在两个不同的线索上，因此可以通过 parent 来表示「主线」。m 参数的值可以是 1 或者 2，对应着 parent 在 merge commit 信息中的顺序。
因而，撤销g的合并操作恢复至原主分支f上：
``` bash
# g为merge后的索引号
git revert -m 1 g
```
从而变成：
``` bash
a -> b -> c -> f -- g -> h -> G -> i (master)
           \      /
            d -> e -> j -> k (dev)
```
此外，由于撤销操作，则在下一次dev与master合并时，merge操作不会合并d、e两个版本代码。因为git认为已经合并或没有合并的需要。此刻，由于新的要合并的dev是在原有d、e版本上开发的（此刻dev已修复bug），这样合并会出错。

因而，需要先撤销G再合并，G为先前撤销合并恢复至主分支操作生成的编号。
``` bash
git checkout master
git revert G
git merge dev
```

参考：https://blog.csdn.net/sndamhming/article/details/56011986