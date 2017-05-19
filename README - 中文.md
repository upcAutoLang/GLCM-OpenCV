该仓库中内容是笔者研究灰度共生矩阵(GLCM)后，根据其原理调用了OpenCV库写出的C++代码。

# 一、原理解释
对GLCM的研究，笔者写了博客。地址如下：
http://blog.csdn.net/ajianyingxiaoqinghan/article/details/71552744

# 二、文件说明

- src：源代码所在路径
	- glcm.h：GLCM算法头文件源码
	- glcm.cpp：GLCM算法实现源码
	- main.cpp：测试GLCM源码
- lib：动态链接库生成路径
- image： 程序测试所用图像存储路径
- CMakeLists.txt：该工程CMake文件

# 三、工程生成教程
## 1. 笔者的工作环境：

- 操作系统：Ubuntu 14.04 LTS
- OpenCV版本: 2.4.9
- 编译条件：
	- 已编译且安装OpenCV
	- 已安装CMake

关于Ubuntu 14.04下OpenCV的安装，笔者写的教程如下：
CSDN：http://blog.csdn.net/ajianyingxiaoqinghan/article/details/62424132 
GitHub：https://github.com/upcAutoLang/Blog/issues/1

## 2. CMake该项目
进入终端，进入GLCM_OpenCV路径，输入以下指令：
```bash
cmake ./
make
```
即可编译该工程。
生成文件路径：/GLCM_OpenCV/bin
生成库文件路径：/GLCM_OpenCV/lib

# 四、测试效果
用/GLCM_OpenCV/image/miska.jpg做测试，结果如下：
![](./image/Test_Result.png)

Debug版本下，测试程序中计算出的该算法效率输出如下：
```cpp
Time of Magnitude Gray Image: 1.38906ms
Time of Calculate Texture Features of the whole Image: 4126.57ms
```
Release版本下，测试程序中计算出的该算法效率输出如下：
```cpp
Time of Magnitude Gray Image: 0.452412ms
Time of Calculate Texture Features of the whole Image: 1291.15ms

```

