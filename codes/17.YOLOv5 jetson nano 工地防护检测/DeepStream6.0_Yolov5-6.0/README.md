# DeepStream6.0_Yolov5-6.0
基于DeepStream6.0和yolov5-6.0的目标检测

1、关于Yolov5n.engine模型的生成，参考 https://github.com/wang-xinyu/tensorrtx

2、关于调用deepstream部分，参考 https://github.com/marcoslucianops/DeepStream-Yolo

在jetson平台上，进入路径： /opt/nvidia/deepstream/deepstream-6.0/sources/

git clone https://github.com/rscgg37248/DeepStream6.0_Yolov5-6.0

cd DeepStream6.0_Yolov5-6.0/nvdsinfer_custom_impl_Yolo

打开makefile，设置CUDA版本，因为是jetson平台，所以设置成“10.2”

make 

即可生成所需的.so文件
