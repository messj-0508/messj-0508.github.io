---
title: GitHub设置无密码登录
date: 2019-04-03 20:07:01
tags: [git]
---


**GitHub项目的授权方式有两种方式：Https和SSH。**

Https可以随意克隆github上的项目，而不管是谁的；而SSH则是你必须是你要克隆的项目的拥有者或管理员，且需要先添加 SSH key ，否则无法克隆。

https url在push的时候是需要验证用户名和密码的；而 SSH在push的时候，是不需要输入用户名的，如果配置SSH key的时候设置了密码，则需要输入密码的，否则直接是不需要输入密码的。



## 一 安装ssh证书

------

1. **首先需要检查你电脑是否已经有 SSH key ，在 git Bash 客户端，输入如下代码：**

```bash
    $ cd ~/.ssh
    $ ls
```

```
	这两个命令就是检查是否已经存在 id_rsa.pub 或 id_dsa.pub 文件，如果文件已经存在，那么则跳过步骤2。
```



2. **创建一个 SSH key **

```bash
    $ ssh-keygen -t rsa -C "your_email@example.com"
```

```
    代码参数含义：
    -t 指定密钥类型，默认是 rsa ，可以省略。
    -C 设置注释文字，比如邮箱。
    -f 指定密钥文件存储文件名。

    接着又会提示你输入两次密码（该密码是你push文件的时候要输入的密码，而不是github管理者的密码），可以不输入密码，直接按回车。
```



3. **添加你的 SSH key 到 github上面去**

   a. 首先你需要拷贝 id_rsa.pub 文件的内容，你可以用编辑器打开文件复制，也可以用git命令复制该文件的内容，如：

```bash
		$ clip < ~/.ssh/id_rsa.pub
```

  	b. 登录你的github账号，从又上角的settings进入，然后点击菜单栏的 SSH key 进入页面添加 SSH key。
  	c. 点击 Add SSH key 按钮添加一个 SSH key 。把你复制的 SSH key 代码粘贴到 key 所对应的输入框中，记得 SSH key 代码的前后不要留有空格或者回车。title随意。



4. **测试一下该SSH key**

```bash
    $ ssh -T git@github.com
```

​	当你输入以上代码时，会有一段警告代码，如：

```
    The authenticity of host 'github.com (207.97.227.239)' can't be established.
    # RSA key fingerprint is 16:27:ac:a5:76:28:2d:36:63:1b:56:4d:eb:df:a6:48.
    # Are you sure you want to continue connecting (yes/no)?
```

​	输入 yes 既可。

​	如果你创建 SSH key 的时候设置了密码，接下来就会提示你输入密码

​	注意：输入密码时如果输错一个字就会不正确，使用删除键是无法更正的。

​	密码正确后你会看到下面这段话，如：

```
    Hi username! You've successfully authenticated, but GitHub does not
    # provide shell access.
```

​	如果用户名是正确的,你已经成功设置SSH密钥。



------

## 二 已有的项目切换到使用SSH方式连接

​	安装ssh证书，每次push pull 都需要输入git密码，原因是使用了https方式 push。

1. 在terminal里边 输入  git remote -v ，可以看到形如一下的返回结果：

```bash
    origin  https://cleey@github.com/cleey/phppoem.git (fetch)
    origin  https://cleey@github.com/cleey/phppoem.git (push)
```

2. 安装以下方式更换成ssh方式的：

```bash
    git remote rm origin
    git remote add origin git@github.com:cleey/phppoem.git
    git push origin 
```

