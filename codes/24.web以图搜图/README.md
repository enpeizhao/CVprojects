# Web以图搜图系统

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220715195118.png?x-oss-process=style/wp" style="zoom: 33%;" />





## 说明：

* 本项目是本人为metric-learning以及模型部署授课做的应用演示。


## 一、效果演示：

| 相似人脸搜索                                       | 相似图片搜索                                      |
| -------------------------------------------------- | ------------------------------------------------- |
| <img src="./md_img/face.gif" style="zoom: 50%;" /> | <img src="./md_img/img.gif" style="zoom: 50%;" /> |



## 一、服务架构

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgdesign-Page-1%20(2).jpg?x-oss-process=style/wp" style="zoom: 67%;" />



## 二、部署服务

准备工作：下载压缩包24.web_img_search.zip 解压到项目同级目录：[下载地址](https://github.com/enpeizhao/CVprojects/releases)

### 2.1、部署后台控制器

> 控制器在flask_server文件夹，需要编译docker Image再运行

```shell
# 在本项目根目录编译docker Image
docker build -t img_search_controler:v1 . 

# 在本项目根目录运行
docker run -p 5000:5000 -v "$(pwd)/flask_server:/app/" -d --name system_controler img_search_controler:v1

```

### 2.2、部署TensorFlow相似图片推理引擎



```shell
# 进入custom_models目录

# 运行镜像
docker run -d --name tf_img_feat -p 8501:8501 `
    -v "$(pwd)/vgg16:/models/vgg16" `
    -e MODEL_NAME=vgg16 `
    tensorflow/serving 
    
# 会生成以下POST API服务（控制器内部调用）
# url：http://localhost:8501/v1/models/vgg16:predict
```

### 2.3、部署pytorch相似人脸推理引擎

```shell
# 进入custom_models/facenet目录

# 运行镜像
docker run -d --name torch_face_embedding `
        -p 8080:8080 `
        -p 8081:8081 `
        -p 8082:8082 `
        -p 7070:7070 `
        -p 7071:7071 `
         -v "$(pwd)/model_store:/home/model-server/model-store" pytorch/torchserve torchserve --start --model-store model-store --models facenet=facenet.mar
         
         
# 会生成以下POST API服务（控制器内部调用）        
# http://localhost:8080/predictions/facenet
```

### 2.4、前端

> 前端使用VUE编写，源码在`front_end_source`中

```shell
# 在本项目根目录执行
docker run -d --name front_end -v  "$(pwd)/dist:/usr/share/nginx/html:ro"  -p 8083:80 nginx

# 浏览器中打开
http://localhost:8083
```



## 三、使用方法

* 人脸搜索：人脸库位置：`flask_server\static\images\faces\origin`，将自己的图片放到下面，然后将`flask_server\static\images\faces\croped`下的图片清空；
* 相似图片搜索：图库位置`flask_server\static\images\voc\JPEGImages`，将自己的图片放到下面即可；
* 重建索引： 浏览器中打开`http://localhost:8083`点击重建索引即可（有索引才能检索到）；



### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

