# openvino扑克牌检测

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220617173131.png?x-oss-process=style/wp)



## 说明：

* 本项目是本人为Pytorch和TensorFlow模型转openvino模型授课做的应用演示


## 一、硬件：

* Windows CPU、macOS CPU（Intel）、树莓派+NCS2

## 二、软件环境安装：

* 安装openvino推理引擎：`pip install openvino==2021.4.0`
* Opencv 
* Python 3.7

## 三、用法：

* 下载媒体`poker.mp4`、权重文件`openvino_poker_weights.zip`解压到yolov5_demo.py同级目录：[下载地址](https://github.com/enpeizhao/CVprojects/releases)
* CPU, GPU, FPGA, HDDL：
  * 媒体文件Demo：`python yolov5_demo.py -i poker.mp4 -m weights/ir_model_no196.xml   -d CPU`
  * 摄像头Demo：`python yolov5_demo.py -i cam -m weights/ir_model_no196.xml   -d CPU`

* 树莓派+NCS2：
  * 媒体文件Demo：`python yolov5_demo.py -i poker.mp4 -m weights/ir_model_no196.xml   -d MYRIAD`
  * 摄像头Demo：`python yolov5_demo.py -i cam -m weights/ir_model_no196.xml   -d MYRIAD`


## 四、其他信息：

* openvino权重文件由YOLOV5 n 模型转换而成，在2W张52类扑克牌训练集上训练了120个epoch后的测试精度mAP_0.5:0.95如下

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220617174806.png?x-oss-process=style/wp" style="zoom:50%;" />

* confusion matrix 如下，（标签第二个字幕代表色号：d、h、s、c分别代表方块、红心、黑桃、梅花）

  <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220617174352.png?x-oss-process=style/wp" style="zoom: 25%;" />



* 随机抽样检测结果：

  ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220617174548.png?x-oss-process=style/wp)



### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

