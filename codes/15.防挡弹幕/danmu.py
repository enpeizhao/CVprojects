"""
!Date: 2022-03-02
!Author: Enpei

弹幕层管理类、弹幕轨道管理类

"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import random


class Danmu_track:
    """
    弹幕轨道管理类，从弹幕层管理处获取文本列表，合成在特定帧位置的弹幕图片
    """
    def __init__(self,track_w,track_h,initial_distance,speed,font_size= 40, font_color=(255, 0, 255, 255)):
        """
        初始化
        
        @param track_w int 轨道宽度
        @param track_h int 轨道高度
        @param initial_distance int 弹幕文字初始距离左侧距离
        @param speed int 运动速度（像素/帧）
        @param font_size int 弹幕字体大小
        @paraam font_color tuple  弹幕文字颜色(RGBA)
        """
        
        # 轨道宽度和高度
        self.track_w = track_w
        self.track_h = track_h  
        # 背景透明图
        self.trans_blank = self.create_blank(self.track_w,self.track_h)
        # 文字间距
        self.text_margin = 100
        # 字体
        self.font = ImageFont.truetype("./fonts/MSYH.ttc", font_size, encoding="utf-8")
        
        #初始距离
        self.initial_distance = initial_distance
        # 运动速度（像素/帧）
        self.speed = speed
        # 字体颜色
        self.font_color = font_color
        

    
    def create_blank(self,w,h):
        """
        绘制透明底图

        @param w int 宽度
        @param h int 高度
        
        @return Image
        """
        return Image.new("RGBA", (w,h), (255, 255, 255, 0))
    
    def draw_text(self,text_list,frame_index):
        """
        在底图上绘制文字

        @param text_list list 弹幕文字列表
        @param frame_index int 帧数索引
        
        @return Image
        """
        # 长文
        long_text =( " "*20).join(text_list)
        # 复制原底图
        trans_blank_copy = self.trans_blank.copy()
        # 绘制
        d = ImageDraw.Draw(trans_blank_copy)
        # 偏移
        distance = self.initial_distance - int(self.speed * frame_index)
        # 绘制字体
        d.text((distance, 0), long_text , font=self.font, fill=self.font_color)
        
        return trans_blank_copy
    



class Danmu_layer:
    """
    定义一个弹幕层管理类
    1、计算轨道数量
    2、分配每条轨道弹幕文字列表
    3、合成轨道图像
    4、添加MASK
    5、返回给视频处理类

    """
    def __init__(self,frame_w,frame_h,danmu_txt):
        """
        初始化

        @param frame_w int 视频宽度
        @param frame_h int 视频高度

        """
        # 画面宽度
        self.frame_w = frame_w
        # 画面高度
        self.frame_h = frame_h
        # 轨道高度
        self.track_h = 100
        # 轨道数量
        self.track_num = int( self.frame_h / self.track_h )
        # 弹幕层背景图
        self.background = self.create_trans_background()

        # 轨道颜色
        self.tracks_color = [(255,255,255,255),(255,0,0,255),(0,255,0,255),(0,0,255,255),(255,255,0,255),(0,255,255,255),(255,0,255,255)]
        # 轨道速度
        self.tracks_speed = [2,4,6,8]
        
        # 轨道对象列表
        self.track_obj_list = self.create_track_objs()

        # 弹幕文本文件
        self.danmu_txt = danmu_txt


    def create_trans_background(self):
        """
        绘制透明背景

        @return Image
        """
        return Image.new("RGBA",(self.frame_w,self.frame_h))
    
    def create_track_objs(self):
        """
        生成多个轨道对象

        @return Danmu_track  obj list 

        """

        track_obj_list = []
        for i in range(self.track_num):

            color = random.choice(self.tracks_color)
            speed = random.choice(self.tracks_speed)

            new_track = Danmu_track(self.frame_w ,self.track_h,1000,speed,80,color)
            track_obj_list.append(new_track)
        return track_obj_list
    

    def danmu_text2list(self):
        """
        从文本文件加载弹幕文字形成list

        @return list
        """

        text_list = None
        # 读取弹幕文本
        with open(self.danmu_txt,'r') as f:
            text_list = f.readlines()
            #清除尾部换行符
            text_list = [t.strip() for t in text_list]
        
        return text_list

    def distri_danmu(self):
        """
        分配弹幕

        @return numpy array
        """
        # 按轨道次序分配，分配后存在list中，[['弹幕1','弹幕4'],['弹幕2','弹幕5'],['弹幕3','弹幕6']]
        danmu_text_list = []
        #弹幕轨道数量
        track_num = self.track_num
        
        #弹幕文字数量
        danmu_text = self.danmu_text2list()
        # 列数，向上取整
        len_danmu_text = len(danmu_text)
        cols_num = math.ceil( len_danmu_text/  track_num)
        
        # 将弹幕列表补空
        blank_arr = ['' for i in range(track_num*cols_num - len_danmu_text) ]
        fixed_danmu_text = np.concatenate((danmu_text, blank_arr))
        
        #文字索引矩阵
        indexes_arr = np.arange(track_num*cols_num).reshape((cols_num,track_num))
        
        #最终分配的弹幕
        return fixed_danmu_text[indexes_arr.T]
    

    def compound_tracks(self,frame_index):
        """
        合成所有轨道图像在某一帧的画面

        @param frame_index int 帧数索引
        @return Image
        """

        # 复制背景
        background = self.background.copy()
        # 分配的弹幕文字
        text_list = self.distri_danmu()
        
        # 多个弹幕轨道绘制
        danmu_img = []
        for track_index in range(self.track_num):
            danmu_img.append( self.track_obj_list[track_index].draw_text(text_list[track_index].tolist(),frame_index) )
            
        # 背景上拼接
        for index,img in enumerate(danmu_img):
            background.paste(img,(0, self.track_h * index))
        return background