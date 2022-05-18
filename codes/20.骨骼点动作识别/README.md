# 骨骼点动态动作识别

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220518201724.png?x-oss-process=style/wp)



> 本项目是本人授课使用，请仅做个人学习、研究使用。
>
> 训练关键点检测的代码暂时仅对学员开放，这里使用`movenet`。



## 一、硬件：

* Windows、树莓派、macOS都可

## 二、软件：

* Pytorch 
* Tflite 

## 三、用法：

* 下载相关文件：[下载地址](https://github.com/enpeizhao/CVprojects/releases)
  * 下载训练好的权重文件压缩包`lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite`，放到到`weights`
  * 将各个动作片段放到至`./data/action_train`目录下
  * 配音文件放到`voice`目录下；
* 运行`demo.py`文件即可。



### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

