"""
! author: enpei
! Date: 2021-12-23
封装常用工具，降低Demo复杂度
"""
# 导入PIL
from PIL import Image, ImageDraw, ImageFont
# 导入OpenCV
import cv2
from matplotlib.pyplot import xlabel
import numpy as np
import time
import os
import glob

from sys import platform as _platform


class Utils:
    def __init__(self):
        pass
    # 添加中文
    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def getFaceXY(self,face):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        return x1,y1,x2,y2

    # 人脸框框
    def draw_face_box(self,face,frame,zh_name,is_watch,distance):

        color = (255, 0, 255) if is_watch =='是' else (0, 255, 0)
        x1,y1,x2,y2 = self.getFaceXY(face)
        
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        m_str = '' if distance =='' else 'm'
        frame = self.cv2AddChineseText(frame, "{name} {distance}{m_str}".format(name = zh_name,distance=distance,m_str=m_str),  (x1, y1-60), textColor=color, textSize=50)

        return frame
    
    
    
    # 保存人脸照片
    def save_face(self,face,frame,face_label_index):

        path = './face_imgs/'+str(face_label_index)
        if not os.path.exists(path):
            os.makedirs(path)

        x1,y1,x2,y2 = self.getFaceXY(face)
        face_img = frame[y1:y2,x1:x2]

        filename = path+'/'+str(face_label_index)+'-'+str(time.time())+'.jpg'
        
        cv2.imwrite(filename,face_img)

    # 人脸点
    def draw_face_points(self,landmarks,frame):
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)
    
    # 获取人脸和label数据
    def getFacesLabels(self):
        # 遍历所有文件
        label_list = []
        img_list = []
        for file_path in glob.glob('./face_imgs/*/*'):

            dir_str = '/'         
            if _platform == "linux" or _platform == "linux2":
                # linux
                pass
            elif _platform == "darwin":
                # MAC OS X
                pass
            elif _platform == "win32":
                # Windows
                dir_str = '\\'
            elif _platform == "win64":
                # Windows 64-bit
                dir_str = '\\'

            label_list.append(int( file_path.split(dir_str)[-1].split('-')[0] ))


            img = cv2.imread(file_path)
            # img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_list.append(img)

        return np.array(label_list),img_list

    # 加载label 对应中文
    def loadLablZh(self):
        with open('./face_model/label.txt',encoding='utf-8') as f:

            back_dict = {}
            for line in f.readlines():
                label_index = line.split(',')[0]
                label_zh = line.split(',')[1].split('\n')[0]
                back_dict[label_index] = label_zh


            return back_dict