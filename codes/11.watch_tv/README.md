# AI分析看电视行为

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_6178.PNG?x-oss-process=style/wp"  style="width:300px;" />

#### 一、功能：

* 人脸识别：检查谁在看
* 头部姿态估计：检查是否在看
* 距离估计：检查是否离电视太近

#### 二、硬件：

* Windows10或11（无需GPU）或MacOS 都测试可行
* 普通RBG USB摄像头

#### 三、软件：

* python 3.7.10

`pip`安装一下依赖包

```
dlib
opencv-contrib-python（可能需要先卸载opencv-python：pip uninstall opencv-python）
```

[点击下载权重文件](https://github.com/enpeizhao/CVprojects/releases/tag/Models)`shape_predictor_68_face_landmarks.dat`，放入`./assets`目录。

#### 四、使用方法：

`python demo.py --命令=参数`

```
  -h, --help            显示帮助
  --mode MODE           运行模式：collect,train,distance,run对应：采集、训练、评估距离、主程序
  --label_id LABEL_ID   采集照片标签id.
  --img_count IMG_COUNT
                        采集照片数量
  --img_interval IMG_INTERVAL
                        采集照片间隔时间
  --display DISPLAY     显示模式，取值1-5
  --w W                 画面宽度
  --h H                 画面高度
```

##### 4.1 自定义人脸识别照片

项目使用最简单的opencv内置的人脸识别算法（精度有限），需要识别自己的人脸请按以下步骤：

1. **采集照片：**`python demo.py --mode='collect' --label_id={人脸ID} --img_count={该ID采集照片数量，一般1-3张即可} --img_interval={照片采集间隔（秒）}； `
2. **修改label：**修改`./face_mode/label.txt`文件，每行代表一个人，如`1,恩培`表示`label_id=1`的人脸叫恩培；
3. **训练模型：**`python demo.py --mode='train'`训练人脸识别模型，模型是`./face_model/model.yml`；

##### 4.2 设置距离参数

项目使用相似三角形原理估算（`f`为相机焦距）：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20211226120754.png?x-oss-process=style/wp" style="width:600px;" />



公式中的`a`取值为人眼角像素距离，参数在`demo.py`第27行`self.eyeBaseDistance = 65`，如果觉得距离不准，请修正这个参数，运行`python demo.py --mode='distance'`，保持相机和人眼睛高度一样，距离相机1.5m处。

##### 4.3 主程序

`python demo.py --mode='run' --w={宽度} --{高度} --display={显示模式}`，`display`取值对应：

* 1：人脸框
* 2：68个人脸关键点
* 3：人脸梯形框框
* 4：人脸方向指针
* 5：人脸三维坐标系







### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

