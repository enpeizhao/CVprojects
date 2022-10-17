# 30.火印结印手势控制Tello无人机

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20221017110221.png?x-oss-process=style/wp" style="zoom:50%;" />





## 说明：

* 本项目是本人为学员授课做的应用演示



## 一、硬件：

* macOS、Windows CPU都可以
* Tello 无人机

## 二、软件环境安装：

* mediapipe
* djitellopy
* pytorch

## 三、用法：

* 下载附件`30.Tello_vision.zip`（权重、字体、图片），解压到项目根目录，[下载地址](https://github.com/enpeizhao/CVprojects/releases)
* YOLOv5训练自己的结印手势，我训练的数据集图片较少，泛化能力较差，如需使用需要自己训练，训练完的权重放在`weights`目录下，也可参考我的权权重；

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img4.%E7%BB%93%E5%8D%B0%E9%A1%BA%E5%BA%8F.png?x-oss-process=style/wp" style="zoom:50%;" />

* 启动`python 9.dsp.py`程序即可；
* 动作指令：

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img9.%E9%AA%A8%E9%AA%BC%E7%82%B9%E5%8A%A8%E4%BD%9C.png?x-oss-process=style/wp)



## 四、其他信息

* 多进程分工细节：

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img11.%E5%A4%9A%E8%BF%9B%E7%A8%8B%E5%88%86%E5%B7%A5.png?x-oss-process=style/wp)

* 状态指令代码：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img10.%E6%8C%87%E4%BB%A4%E7%8A%B6%E6%80%81%E7%A0%81.png?x-oss-process=style/wp" style="zoom:33%;" />







### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

