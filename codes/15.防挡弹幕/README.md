# 防挡弹幕

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220305210006.png?x-oss-process=style/wp)





## 一、硬件：

* Windows10或11（无需GPU，有最好）或MacOS 都测试可行

## 二、软件：

* Python==3.8
* TensorFlow
* imgaug
* opencv 
* pixellib

## 二、用法：

```
optional arguments:
  -h, --help     show this help message and exit
  --video VIDEO  要处理的视频
  --mode MODE    运行模式：mask,compound 对应：生成蒙版图片、合成视频
  --danmu DANMU  弹幕文本文件
```



### 2.1、GitHub仓库需下载相关文件，gitee无需

* 2.1.1 下载字体 `MSYH.ttc` 至 `./fonts`目录下，[下载地址点这里](https://github.com/enpeizhao/CVprojects/releases/tag/font)
* 2.1.2 下载预训练模型 `mask_rcnn_coco.h5`至`./weights`目录下，[下载地址点这里](https://github.com/enpeizhao/CVprojects/releases/tag/Models)
* 2.1.3 下载演示视频`15.demo.mp4`至`./videos`目录下，[下载地址点这里](https://github.com/enpeizhao/CVprojects/releases/tag/media)

### 2.2、生成蒙版文件

命令：`python demo.py --video={mp4视频地址} --mode=mask`，如`python demo.py --video='./videos/demo.mp4' --mode=mask`，系统会在`mask_img`文件夹下生成每帧画面的蒙版图，类似下图：

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220305210903.png?x-oss-process=style/wp)



### 2.3、合成弹幕视频

命令：`python demo.py --video={视频地址} --mode=compound --danmu={弹幕txt文件地址}`，如：`python demo.py --video='./videos/demo.mp4' --mode=compound --danmu=danmu.txt`，渲染后的视频在`record_video`目录下。



### 

### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

