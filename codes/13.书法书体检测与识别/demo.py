import cv2
import numpy as np
from utils import Utils

import time
utils = Utils()

cap = cv2.VideoCapture('./videos/raw.mp4')

width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_index = 0

fpsTime = time.time()

videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), 15, (1080,1920))

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(1080,1920))

    img_copy = frame.copy()
    
        
    # 遮罩手部：可以用手部检测器实现，
    frame[1550:1920,700:1080]  = 255
    

    if 154 >frame_index> 50:
        # cv2.rectangle(frame,(0,1300),(300,1920),(255,0,255),10)
        # cv2.rectangle(frame,(300,1400),(1080,1920),(255,0,255),10)

        frame[1300:1920,0:300]  = 255
        frame[1400:1920,300:1080]  = 255

    if 46 >frame_index> 38:
        cv2.rectangle(frame,(300,400),(600,1920),(255,0,255),10)

        frame[400:1920,300:600]  = 255
        # frame[1400:1920,300:1080]  = 255
    
    
    # frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
    # 灰度
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 二值化
    retval, black_img = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

    # 腐蚀
    kernel = np.ones((3,3),dtype=np.int8)
    erosion = cv2.erode(black_img,kernel,iterations = 2)

    # 再膨胀，连接主体
    kernel = np.ones((10,10),dtype=np.int8)
    dialation = cv2.dilate(erosion,kernel,iterations = 2)

    # # 在膨胀基础上闭合

    kernel = np.ones((10,10),dtype=np.int8)
    closing = cv2.morphologyEx(dialation,cv2.MORPH_CLOSE,kernel)

    # # 复制底图
    
    # 再次取边缘、轮廓

    edged = cv2.Canny(closing.copy(),30,200)
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    


    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # 过滤
        if( 300> w > 100 ) and ( 300>h > 100):
            cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),10)


    frame_index+=1

    # time.sleep(0.1)

    cTime = time.time()
    fps_text = 1/(cTime-fpsTime)
    fpsTime = cTime
    
    # img_copy = utils.cv2AddChineseText(img_copy, "帧率: " + str(int(fps_text)) ,  (20, 100), textColor=(255, 0, 255), textSize=100)

    videoWriter.write(img_copy)
    cv2.imshow('demo',img_copy)
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()