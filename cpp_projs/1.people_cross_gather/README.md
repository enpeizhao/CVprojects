# 1. 人员越界及聚众

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img202303171105294.png?x-oss-process=style/wp" style="zoom:50%;" />





## 说明：

* 本项目是本人为学员授课《C++ TensorRT高性能部署》的项目演示

## 一、硬件：

* Ubuntu GPU

## 二、软件环境安装：

* TensorRT 8.4 
* C++ 14

## 三、用法：

* 下载权重`yolov5s_person.onnx`，解压到`weights`，[下载地址](https://github.com/enpeizhao/CVprojects/releases)
* `cmake . -B build && cmake --build build`
* 在你的机器上构建engine：` ./build/build [onnx_file_path] [calib_dir] [calib_list_file]`
* 运行`./build/runtime_thread ./weights/yolov5.engine {视频文件}  2 50 0 2000000`
* 使用VLC等客户端读取视频流：`rtmp://localhost:1936/live/mystream`



### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

