# OCR车牌识别

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220504215359.png?x-oss-process=style/wp" style="zoom:50%;" />



> 本项目是本人授课使用，请仅做个人学习、研究使用。



## 一、硬件：

* PC端运行：Windows10或11（最好有GPU）
* MACOS（比较卡）

## 二、软件：

* Pytorch 
* YOLOv5
* paddlepaddle
* opencv 
* 其他组件：`pip install -r requirements.txt`

## 三、用法：

* 参考[YOLOv5官网](https://github.com/ultralytics/yolov5)，将YOLOv5 clone到本项目目录（当前YOLOv5目录为空，替换即可）；
* 下载相关文件：[下载地址](https://github.com/enpeizhao/CVprojects/releases)
  * 下载训练好的权重文件压缩包`car_plate_weights.zip`，解压到`weights`
  * 下载字体`MSYH.ttc`至`fonts`目录下；
  * 下载媒体文件`car_plate.mp4`至`test_imgs`目录下（该视频是本人拍摄，禁止他用）
* 三阶段（车身检测+文字检测+文字识别），运行`python demo_with_yolo_pretrained.py`即可；
* 二阶段（车牌检测+文字识别），运行`python demo_with_yolo_custom.py`即可；





### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

