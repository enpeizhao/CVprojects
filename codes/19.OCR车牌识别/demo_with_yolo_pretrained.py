import cv2
import numpy as np
import torch
import time
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tools.infer.utility as utility
from tools.infer.predict_system import TextSystem,predict_rec



class Plate_detect:

    def __init__(self):

        args = utility.parse_args()
        
        args.det_model_dir="./weights/det/ch_db_mv3_inference/"
        args.rec_model_dir="./weights/rec/ch_ppocr_server_v2.0_rec_infer/"
        args.rec_char_dict_path="./ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls=False 
        args.use_gpu=True
        # 检测加识别
        self.text_sys = TextSystem(args)
        # OCR热身
        if 1:
            print('Warm up ocr model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)

        
        # 加载模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./weights/yolov5s.pt',source='local')  # local repo
        # 置信度阈值
        self.model.conf = 0.4
        # 加载摄像头
        self.cap = cv2.VideoCapture('./test_imgs/car_plate.mp4')

        # 画面宽度和高度
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
    
    # 添加中文
    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/MSYH.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


    def detect(self):

        while True:
            ret,frame = self.cap.read()

            if frame is None:
                break
            # 转为RGB
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 推理
            results = self.model(img_cvt)

            pd = results.pandas().xyxy[0]
            car_list = pd[pd['name']=='car'].to_numpy()
            # 遍历每辆车
            for car in car_list:
                l,t,r,b = car[:4].astype('int')
                
                cv2.rectangle(frame, (l,t), (r,b), (0,255,0),5)
                # cv2.putText(frame, str(r-l), (l,t-10), cv2.FONT_HERSHEY_PLAIN, 10, (0,255,0),5)
                if r-l > 500:
                    crop = frame[t:b,l:r]
                    dt_boxes, rec_res = self.text_sys(crop)
                
                    for index,box in enumerate(dt_boxes) :
                        box = np.array(box).astype(np.int32)
                        box[:,0] = box[:,0] + l
                        box[:,1] = box[:,1] + t

                        cv2.polylines(frame, [box], True, color=(255, 0, 255), thickness=5)
                        box_l,box_t = box[0][0],box[0][1]
                        text, score = rec_res[index]
                        frame = self.cv2AddChineseText(frame, text, (box_l,box_t-100),(0,255,0),80) 
                        print(text, score)     

     
            frame= cv2.resize(frame, (608,1080))
            cv2.imshow('demo',frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


plate = Plate_detect()            
plate.detect()


