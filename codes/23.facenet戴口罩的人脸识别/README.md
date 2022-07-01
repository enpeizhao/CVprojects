# facenet戴口罩的人脸识别

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_7913.PNG?x-oss-process=style/wp)



## 说明：

* 本项目是本人为RESNET、inception授课做的应用演示。


## 一、硬件：

* Windows 
* macOS

## 二、软件环境安装：

* TensorFlow 2
* pytorch 1.10
* OpenCV

## 三、用法：

* 下载压缩包`23.facenet_mask_recognition.zip`解压到`demo_mac.py`同级目录：[下载地址](https://github.com/enpeizhao/CVprojects/releases)

  <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220701105256.png?x-oss-process=style/wp" style="zoom:50%;" />

* 人脸库照片放到`images/origin`下，命名规则为`人名_序号.jpg`，如`恩培_1.jpg`，`恩培_2.jpg`，一个人建议放个1~3张清晰度较好的图片

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220701110037.png?x-oss-process=style/wp" style="zoom: 50%;" />

* Windows 运行：`python demo_windows.py --threshold={识别阈值}`如`python demo_windows.py --threshold=1`，阈值越低要求越严格
* macOS运行：`python demo_mac.py --threshold={识别阈值}`

## 四、其他信息：

* 模型架构：inception_resnet_v1
* 训练集：1.4W人80W张图片（一半是合成的戴口罩数据集）

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220701105743.png?x-oss-process=style/wp" style="zoom: 25%;" />

* 测试ACC：1.2W张测试样本上，ACC最高为97.1%；
* 训练时间：A100 145EPOCH，28个小时
* ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img1.png?x-oss-process=style/wp)
* ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img2.png?x-oss-process=style/wp)





### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

