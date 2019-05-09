---
title: 'subsystem of Windows-System:Ubuntu16.04'
date: 2019-05-03 14:31:53
tags: [Ubuntu, Tutorial]
---

## Unit 1 : How to install subsystem?

**Step1 : Open Microsoft Store and search the key word: "Ubuntu" ，then choose the product: "Ubuntu 16.04 LTS" . **

![Sample Picture](Windows-System-Ubuntu16-04\1.PNG)

**Step2 : Click the button: Get , and wait for it to finish**

It's ok! You can launch it , but maybe you can have the problem bellow.

**Problem1 : WslRegisterDistribution failed with error: 0x8007019e .**
``` vim
1. OPen the project "Windows PowerShell(Admin)"
2. Enter the command "Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux"
3. When the process was completed , reboot your system.

```

## Unit 2 ：How to use it?

### 1. Query Ubuntu's  version

**Enter the command:**

``` shell
cat /etc/os-release
```

![Sample Picture](Windows-System-Ubuntu16-04\2.PNG)

### 2. Update the software source of Ubuntu

**Enter the command as follow:**

``` bash
cd /etc/apt
sudo cp sources.list sources.list.bak
sudo vi sources.list
```
**Delete all content of the file , and enter the content as follow:**

``` vi
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```
**Enter the command to update software:**

``` bash
sudo apt-get upgrade 
sudo apt-get update 
```

### 3. Access the file of Windows System

You can access the file in "/mnt" . For example , if  accessing the file located in "C:\Users\" , you can enter the command:
``` bash
cd /mnt/c/Users/
```