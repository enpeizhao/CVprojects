"""
加载训练好的YOLOv5n模型，并做简单识别应用
"""
from tkinter import Frame
import cv2
import numpy as np
import torch
import time
import pandas as pd

class PPE_detect:

    def __init__(self):
        # 加载模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./weights/ppe_yolo_n.pt',source='local')  # local repo
        # 置信度阈值
        self.model.conf = 0.3
        # 加载摄像头
        self.cap = cv2.VideoCapture(0)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(self.frame_w,self.frame_h)
        # 画面宽度和高度
        #self.frame_w = 1440
        #self.frame_h = 1080

        
        
        # 标签
        labels =  ['person','vest','blue helmet','red helmet','white helmet','yellow helmet'] 

        # 加载浮层
        self.overlay_person = self.getPng('./icons/person.png')
        self.overlay_vest = [
            self.getPng('./icons/vest_on.png'),
            self.getPng('./icons/vest_off.png')
        ]
        # 
        self.overlay_hat = [
            self.getPng('./icons/hat_blue.png'),
            self.getPng('./icons/hat_red.png'),
            self.getPng('./icons/hat_white.png'),
            self.getPng('./icons/hat_yellow.png'),
            self.getPng('./icons/hat_off.png'), # 最后一个不戴帽子
        ]
        self.color_hat = [(255,0,0),(0,0,255),(255,255,255),(0,255,255)]

    def getPng(self,fileName):
        """
        获取PNG图像
        @return numpy array 
        """
        overlay = cv2.imread(fileName)
        # overlay = cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)
        overlay = cv2.resize(overlay,(0,0), fx=0.3, fy=0.3)

        return overlay


    def get_iou(self,boxA, boxB):
        """
        计算两个框的IOU

        @param: boxA,boxB list形式的框坐标
        @return: iou float 
        """
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou


    def get_person_info_list(self,person_list,hat_list,vest_list):
        """
        获取每个人的完整信息
        
        @param: person_list,hat_list,vest_list numpy array
        @return  person_info_list list
        """
        hat_iou_thresh = 0
        vest_iou_thresh = 0

        person_info_list = []

        for person in person_list:
            person_info_item = [[],[],[]]
            # 人体框
            person_box = person[:5]
            
            person_info_item[0]= person_box
            # 依次与帽子计算IOU
            for hat in hat_list:
                hat_box = hat[:6]
                hat_iou = self.get_iou(person_box, hat_box)
                
                if hat_iou > hat_iou_thresh:
                    person_info_item[1] = hat_box
                    break
                    
            # 依次与防护服计算IOU
            for vest in vest_list:
                vest_box = vest[:5]
                vest_iou = self.get_iou(person_box, vest_box)

                
                if vest_iou > vest_iou_thresh:
                    person_info_item[2] = vest_box
                    break

            person_info_list.append(person_info_item)
        
        return person_info_list

    
    def render_frame(self,frame,person_info_list):
        """
        渲染每个人的信息

        @param: frame numpy img 渲染底图
        @param: person_info_list list 人员信息

        """

        for person in person_info_list:
            person_box = person[0]
            hat_box = person[1]
            vest_box = person[2]

            if len(person_box)>0:
                p_l,p_t,p_r,p_b = person_box[:4].astype('int')
                conf = person_box[4]

                conf_txt =str( round(conf*100,1) ) + '%'
                

                cv2.rectangle(frame,(p_l,p_t),(p_r,p_b),(0,255,0),5)
                cv2.putText(frame,conf_txt,(p_l,p_t-35),cv2.FONT_ITALIC,1,(0,255,0),2)                
            
            if len(hat_box) > 0:
                l,t,r,b = hat_box[:4].astype('int')
                label_id = hat_box[5]

                cv2.rectangle(frame,(l,t),(r,b),self.color_hat[label_id-2],8)

                hat_overlay = self.overlay_hat[label_id-2]
                
            else:
                hat_overlay = self.overlay_hat[-1]
        

            hat_overlay_h,hat_overlay_w = hat_overlay.shape[:2]
            
            overlay_l,overlay_t = p_l+150,p_t-85

            overlay_r,overlay_b = (overlay_l + hat_overlay_w),(overlay_t+hat_overlay_h)

            if overlay_t > 0 and overlay_r < self.frame_w:

                # 覆盖图层
                overlay_copy=cv2.addWeighted(frame[overlay_t:overlay_b, overlay_l:overlay_r ],1,hat_overlay,1,0)
                frame[overlay_t:overlay_b, overlay_l:overlay_r ] = overlay_copy


            if len(vest_box)>0:
                l,t,r,b = vest_box[:4].astype('int')
                vest_overlay = self.overlay_vest[0]
                conf = vest_box[4]

                conf_txt =str( round(conf*100,1) ) + '%'

                cv2.rectangle(frame,(l,t),(r,b),(255,0,0),5)

                cv2.putText(frame,conf_txt,(l,t-35),cv2.FONT_ITALIC,1,(255,0,0),2)                

            else:
                vest_overlay = self.overlay_vest[1]

            vest_overlay_h,vest_overlay_w = vest_overlay.shape[:2]
            
            overlay_l,overlay_t = p_l+220,p_t-85

            overlay_r,overlay_b = (overlay_l + vest_overlay_w),(overlay_t+vest_overlay_h)

            if overlay_t > 0 and overlay_r < self.frame_w:

                # 覆盖图层
                overlay_copy=cv2.addWeighted(frame[overlay_t:overlay_b, overlay_l:overlay_r ],1,vest_overlay,1,0)
                frame[overlay_t:overlay_b, overlay_l:overlay_r ] = overlay_copy



    def detect(self):
        """
        检测识别
        """

        while True:

            ret,frame = self.cap.read()

            frame = cv2.flip(frame,1)
            # 转为RGB
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 记录推理耗时
            start_time = time.time()
            # 推理
            results = self.model(img_cvt)
            pd = results.pandas().xyxy[0]
            

            person_list = pd[pd['name']=='person'].to_numpy()
            vest_list = pd[pd['name']=='vest'].to_numpy()
            hat_list = pd[pd['name'].str.contains('helmet')].to_numpy()

            #获取人员信息
            person_info_list = self.get_person_info_list(person_list,hat_list,vest_list)

            #遍历每个人，渲染相应数据
            self.render_frame(frame,person_info_list)



            end_time = time.time()
            fps_text = 1/(end_time - start_time)
            

            cv2.putText(frame,'FPS: '+ str(round(fps_text,2)),(30,50),cv2.FONT_ITALIC,1,(0,255,0),2)
            cv2.putText(frame,'Person: '+ str(len(person_info_list)),(30,100),cv2.FONT_ITALIC,1,(0,255,0),2)


            cv2.imshow('demo',frame)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


ppe = PPE_detect()            
ppe.detect()


