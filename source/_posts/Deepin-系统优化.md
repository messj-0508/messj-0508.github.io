---
title: Deepin 系统优化
date: 2019-07-26 15:17:12
tags: [Ubuntu, Tutorial]
---


## 一 . 修改系统默认编辑器
系统默认编辑器是：nano
```
# 打开Terminal，输入以下指令：
sudo update-alternatives --config editor

# 输入所要变更的编辑器的编号（Terminal上会有提示）
```

为方便复制粘贴，修改Terminal的设置。点击Terminal右上角菜单标志——设置。
1. 将终端中复制粘贴的快捷键设置为 Ctrl+C 和 Ctrl+V 。
2. 将终端中搜索的快捷键设置为 Ctrl+F 。
3. 将终端中放大缩小的快捷键设置为 Ctrl+N 和 Ctrl+M 。
4. 将光标中“选中光标时自动复制到剪切板”选项勾上。

## 二 . 设置sudo不用输入密码
```
# 打开Terminal，输入以下指令：
sudo visudo

# 在编辑器中更换"%admin ALL=(ALL) ALL"为以下指令：
%admin ALL=(ALL) NOPASSWD: ALL
```

## 三 . CPU优化
问题：Deepin Linux 15.10升级后CPU不会自动降频造成过热
```
# 打开Terminal，输入以下指令：
sudo gedit /etc/default/grub

# 编辑grub文件，其中两行改为如下：
GRUB_CMDLINE_LINUX="splash quiet "
GRUB_CMDLINE_LINUX_DEFAULT="intel_pstate=disable"

# 保存退出后更新一下grub
sudo update-grub

# 然后，重启系统。
sudo reboot
```

## 四 . 修改WIFI配置文件
问题：Deepin 15.8/Ubuntu 18.04用intel无线网卡速度跑不满
```
# 打开Terminal，输入以下指令：
sudo vim /etc/modprobe.d/iwlwifi.conf

# 然后把iwlwifi.conf里面的11n_disable=1改成
11n_disable=8

# 保存并重新启动
sudo reboot
```

## 五 . 镜像源（Apt软件源）的修改
```
# 打开Terminal，输入以下指令：
sudo vim /etc/apt/sources.list

# 在编辑器中更换 http://packages.deepin.com 为 
https://mirrors.tuna.tsinghua.edu.cn

# 更新
sudo apt-get update
sudo apt-get upgrade
```

此外，deepin或是其他Linux（APT）系统中可能无法使用 add-apt-repository 命令。请执行以下操作，确保该命令能够被使用：
```
# 使用以下两个安装命令（旧系统版本使用第一条，新版使用第二条）
sudo apt-get install python-software-properties
sudo apt-get install software-properties-common

# 更新
sudo apt-get update
sudo apt-get upgrade
```

## 六 . 安装pip并修改pip源
```
# 安装pip和pip3：
sudo apt install -y python-pip python3-pip

# 更换pip源（以下更改用户pip源，若希望全局有效，直接 sudo vim /etc/pip.conf ）
cd ~
sudo mkdir .pip
sudo vim .pip/pip.conf

# 添加以下内容至pip.conf文件
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

## 七. 登录和唤醒免密
1. 设置自动免密登录

点击设置——账户——example（你的账户）
将"自动登录"开关打开（勾上清空钥匙串密码）
将"无密码登录"开关打开（勾上清空钥匙串密码）

1. 设置唤醒免密

点击设置——电源管理
将"唤醒显示器时需要密码"开关关闭
将"待机恢复时需要密码"开关关闭

## 八. 常用商店软件
1. 搜狗输入法
2. 百度网盘
3. 迅雷
4. 微信
5. QQ
6. VS Code
7. Annaconda

## 九 . 命令行软件
1. 安装Git
```
sudo apt-get install git
git config --global user.email "464306924@qq.com"
git config --global user.name "messj-0508"
```
设置GitHub的ssh 连接，参考[《GitHub设置无密码登录》](https://mwessj.xyz/2019/04/03/GitHub%E8%AE%BE%E7%BD%AE%E6%97%A0%E5%AF%86%E7%A0%81%E7%99%BB%E5%BD%95/)
### 2.安装Typora
1. 下载[二进制免安装包](https://typora.io/linux/Typora-linux-x64.tar.gz)
2. 解压缩到用户目录下
3. 添加指令路径
```
# Terminal
sudo vim .bashrc

# vim 在最后补充两句
export PATH=$PATH:~/Typora
alias typora='Typora'

# Terminal（重新加载bash配置）
source .bashrc
```
1. 添加菜单启动项（可选）
```
# Terminal
cd /usr/share/applications
sudo vim Typora.desktop

# Vim
[Desktop Entry]
Version=1.0 # 版本号
Name=Typora    # 将要在启动器显示的名字
Comment=a markdown editor  # 说明
Exec=/xx/xx/Typora  # 可执行程序路径，一定要是完整的绝对路径
Icon=/xx/xx.png  # 程序图标
Terminal=false 
Type=Application 
Categories=Editor;
```
1. 安装nodejs和npm（可选，hexo博客需要）
```
sudo apt install nodejs-legacy
sudo apt install node
sudo apt install nodejs-bin

sudo npm install -g n
sudo npm install hexo-cli -g

cd ~/Desktop
git clone git@github.com:messj-0508/messj-0508.github.io.git hexo

cd hexo
npm install
```



## 十 . VS Code 配置优化