# C++ Deepstream TensorRT多流检测追踪

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img202304061135614.png?x-oss-process=style/wp)





## 说明：

* 本项目是本人为学员授课《C++ TensorRT高性能部署》的项目演示

## 一、硬件：

* Ubuntu GPU

## 二、软件环境安装：

* TensorRT 8.4 
* C++ 14

## 三、用法：

* 参考[cpp_projs/1.people_cross_gather](https://github.com/enpeizhao/CVprojects/tree/main/cpp_projs/1.people_cross_gather) 先build engine。

* `cmake . -B build && cmake --build build`

* ```
  # 读取文件
  ./build/ds_app file:///app/4.ds_tracker/media/sample_720p.mp4
  # RTSP
  ./build/ds_app rtsp://localhost:8555/live1.sdp
  ```

* 使用VLC等客户端读取视频流：`rtsp://127.0.0.1:8554/ds-test`



### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

