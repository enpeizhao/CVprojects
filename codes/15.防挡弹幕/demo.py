"""
!Date: 2022-03-02
!Author: Enpei
MASK蒙版生成及弹幕合成

"""

# 导入相关包
import cv2
import numpy as np

import pixellib
from pixellib.instance import instance_segmentation


# 导入弹幕管理模块
from danmu import Danmu_layer

import os
from PIL import Image

import time


from argparse import ArgumentParser

class VideoProcess:
    """
    防挡弹幕
    1.将视频切割为一帧帧的MASK，并存入图片
    2、根据MASK文件构造弹幕图层
    3、结合两个图层，显示并输出最终带弹幕的视频

    """

    def __init__(self,videoFile):

        """
        初始化

        @param videoFile MP4 视频文件
        """
        # 操作文件
        self.videoFile = videoFile

    
    def video2Masks(self):
        """
        将视频处理为一帧帧的MASK并保存
        
        """
        # 读取视频
        cap = cv2.VideoCapture(self.videoFile)

        # 高度和宽度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 分割模型
        segment_frame = instance_segmentation()
        segment_frame.load_model("weights/mask_rcnn_coco.h5")

        target_classes = segment_frame.select_target_classes(person = True)
        

        # frame位置
        frame_index  = 0
        while True:
            ret,frame = cap.read()

            if not ret:
                print('处理完毕')
                break

            results, output = segment_frame.segmentFrame( frame,segment_target_classes = target_classes, show_bboxes = False)

            # 获取MASK
            masks = results['masks']
            instance_num = len(results['class_ids'])
            # 将多个人的MASK拼成一个
            if instance_num > 0:

                # 构造原图大小的灰度图片，默认黑色
                black_ing = np.zeros(output.shape[:2])

                # 遍历每个人
                for index in range(masks.shape[2]) :

                    black_ing = np.where(masks[:,:,index] == True,255,black_ing)

                # 存储照片
                cv2.imwrite('./mask_img/'+str(frame_index)+'.jpg',black_ing)

                # 输出信息
                print("第{num}帧已存储MASK".format(num=frame_index))
            else:
                print("第{num}帧无数据".format(num=frame_index))

            frame_index +=1

            # 显示
            cv2.imshow('Video Demo',output)

            # 退出条件
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def compound_video(self,danmu_txt):
        """
        合成弹幕视频
        1、读取原视频
        2、逐帧处理
        3、弹幕+MASK + 原视频

        """

        
        cap = cv2.VideoCapture(self.videoFile)

        # 帧数
        frame_index = 0


        # 高度和宽度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FPS
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (width,height))


        # 弹幕管理器
        new_danmu_layer = Danmu_layer(width,height,danmu_txt)


        while True:
            ret,frame = cap.read()

            if not ret:
                print('处理完毕')
                break

            # 加载本帧的弹幕
            danmu_img = new_danmu_layer.compound_tracks(frame_index)

            # mask文件
            mask_file = './mask_img/' + str(frame_index) + '.jpg'

            if os.path.exists(mask_file):
                mask_img = cv2.imread(mask_file)
                # 转为GRAY单通道图
                mask_img = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)

                # 遮罩处理
                # 将弹幕转为numpy图像
                danmu_img_np = np.asarray(danmu_img)
                # 处理alpha通道
                danmu_img_np[:,:,3] = np.where(mask_img==255,0,danmu_img_np[:,:,3])

                # 还原为PIL格式
                danmu_img = Image.fromarray(danmu_img_np)

            # 合成原图与弹幕
            # 将frame转为RGBA四通道图
            frame_rgba = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
            # 转为PIL格式
            frame_pil = Image.fromarray(frame_rgba)

            # 拼接
            img_compose = Image.alpha_composite(frame_pil,danmu_img)

            # 转为CV2格式
            img_compose = np.asanyarray(img_compose) 

            # 转为BGR通道
            img_compose = cv2.cvtColor(img_compose,cv2.COLOR_RGB2BGR)

            videoWriter.write(img_compose)

            frame_index+=1

            # 显示
            cv2.imshow('Video Demo',img_compose)

            # 退出条件
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        videoWriter.release()
        cap.release()
        cv2.destroyAllWindows()

            



# 参数
parser = ArgumentParser()

parser.add_argument("--video", type=str, default='./videos/15.demo.mp4',
                    help="要处理的视频")   


parser.add_argument("--mode", type=str, default='mask',
                    help="运行模式：mask,compound 对应：生成蒙版图片、合成视频")



parser.add_argument("--danmu", type=str, default='danmu_real.txt',
                    help="弹幕文本文件")   

args = parser.parse_args()



mode = args.mode

vp = VideoProcess(args.video)

if mode == 'mask':
    
    vp.video2Masks()

if  mode == 'compound':

    vp.compound_video(args.danmu)