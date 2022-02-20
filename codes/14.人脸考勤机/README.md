# 人脸考勤机

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220220094051.png?x-oss-process=style/wp" style="width:400px;" />



## 一、硬件：

* Windows10或11（无需GPU）或MacOS 都测试可行
* 普通RBG USB摄像头

## 二、软件：

* Python：3.7
* opencv 
* Dlib

## 二、用法：

使用`python demo_full.py --{参数名}={参数值}`

```
  -h, --help            show this help message and exit
  --mode MODE           运行模式：reg,recog 对应：注册人脸、识别人脸
  --id ID               人脸ID，正整数不可以重复
  --name NAME           人脸姓名，英文格式
  --interval INTERVAL   注册人脸每张间隔时间，秒
  --count COUNT         注册人脸照片数量
  --w W                 画面缩放宽度，默认相机宽度1/2
  --h H                 画面缩放高度，默认相机高度1/2
  --detector DETECTOR   人脸识别使用的检测器，可选haar、hog、cnn
  --threshold THRESHOLD
                        人脸识别距离阈值，越低越准确
  --record RECORD       人脸识别是否录制
```



### 2.1、下载模型与字体

**2.1.1 [下载模型文件](https://github.com/enpeizhao/CVprojects/releases/tag/Models)放到`./weights`目录下**：

* [下载地址](https://github.com/enpeizhao/CVprojects/releases/tag/Models)

* 文件列表

  * `dlib_face_recognition_resnet_model_v1.dat`

  * `haarcascade_frontalface_default.xml`

  * `mmod_human_face_detector.dat`

  * `shape_predictor_68_face_landmarks.dat`

**2.1.2下载`Songti.ttc`字体文件放到`./fonts`目录下**

* [下载地址](https://github.com/enpeizhao/CVprojects/releases/tag/font)

### 2.2、注册人脸：将人脸特征写入`./data/feature.csv`

用法：

`python demo_full.py --mode=reg --id={人脸ID，正整数} --name={人脸姓名，英文格式} --interval={注册人脸每张间隔时间，秒} --count={注册人脸照片数量} --w={画面宽度} --h={画面高度} `

如：

`python demo_full.py --mode=reg --id=1 --name='Enpei' --interval=3 --count=3`



### 2.3、识别人脸开始考勤：将考勤记录写入`./data/attendance.csv`

用法：

`python demo_full.py --mode=recog --detector={人脸识别使用的检测器，可选haar、hog、cnn} --threshold={人脸识别距离阈值，越低越准确} --record={人脸识别是否录制}`

如：

`python demo_full.py --mode=recog --detector=haar --threshold=0.5 --record=True`



### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

