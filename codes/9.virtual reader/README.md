# 虚拟点读机

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20211211154451.png?x-oss-process=style/wp" style="width:200px;" />

硬件：

Windows11、GPU：nvdia GTX 1060 、普通RBG相机

软件：

* conda 
* Python 3.7
* CUDA 10.2 
* cuDNN7.6.5
* mediapipe 0.8.9
* Paddlepaddle 2.2



使用步骤：

1. 满足硬件条件（需要GPU）和软件条件
2. 安装PaddleDetection：[根据官网安装](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/README_cn.md)
3. 安装PaddleOCR：[根据官网安装](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/README_ch.md)
4. 下载PaddleDetection 识别模型，并deploy后（查看官网教程：[Python端预测部署](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/deploy/python)），将生成的`infer_cfg.yml, model.pdiparams, model.pdiparams.info, model.pdmodel`文件放到`baidu_pp_detection/models`下，类似这样：

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20211211153029.png?x-oss-process=style/wp)

[推荐下载](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/docs/featured_model/LARGE_SCALE_DET_MODEL.md)`cascade_rcnn_dcn_r101_vd_fpn_gen_server_side`模型，它支持676个类别识别（[详情查看这里](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/dataset/voc/generic_det_label_list_zh.txt)），且经过我的测试，速度较好。

5. [下载OCR推理模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/models_list.md)，解压后分别放到`baidu_pp_ocr/models/`文件夹下，类似这样：

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20211211153818.png?x-oss-process=style/wp)

文本检测模型推荐下载`ch_PP-OCRv2_det_infer`，文本识别模型推荐下载`ch_PP-OCRv2_rec_infer`



6. 运行`python demo.py`即可实时离线识别和OCR。





### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

