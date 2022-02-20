"""
人脸考勤系统（可以运行在CPU）
1、人脸检测：
2、注册人脸，将人脸特征存储进feature.csv
3、识别人脸，将考勤数据存进attendance.csv
"""

# 导入包
import cv2
import numpy as np
import dlib
import time
import csv
from argparse import ArgumentParser
# 导入PIL
from PIL import Image, ImageDraw, ImageFont



# 加载人脸检测器
hog_face_detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')
haar_face_detector  = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

# 加载关键点检测器
points_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
# 加载resnet模型
face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

# 绘制中文
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./fonts/songti.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 绘制左侧信息
def drawLeftInfo(frame,fpsText,mode="Reg",detector='haar',person=1, count=1):
    
    # 帧率
    cv2.putText(frame, "FPS:  " + str(round(fpsText,2)), (30, 50), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

    # 模式：注册、识别
    cv2.putText(frame, "Mode:  " + str(mode), (30, 80), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

    if mode == 'Recog':
        # 检测器
        cv2.putText(frame, "Detector:  " + detector, (30, 110), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

        # 人数
        cv2.putText(frame, "Person:  " + str(person), (30, 140), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

        # 总人数
        cv2.putText(frame, "Count:  " + str(count), (30, 170), cv2.FONT_ITALIC,0.8, (0, 255, 0), 2)


# 注册人脸
def faceRegiser(faceId=1,userName='default',interval=3,faceCount=3,resize_w=0,resize_h=0):


    # 计数
    count = 0
    # 开始注册时间
    startTime = time.time()

    # 视频时间
    frameTime = startTime

    # 控制显示打卡成功的时长
    show_time = (startTime - 10)

    # 打开文件
    f =  open('./data/feature.csv','a',newline='')
    csv_writer = csv.writer(f)

    cap = cv2.VideoCapture(0)

    if resize_w == 0 or resize_h == 0:
        resize_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
        resize_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) //2

    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame,(resize_w,resize_h))
        frame = cv2.flip(frame,1)

        # 检测
        face_detetion = hog_face_detector(frame,1)

        for face in face_detetion:
            # 识别68个关键点
            points = points_detector(frame,face)
            # 绘制人脸关键点
            for point in points.parts():
                cv2.circle(frame,(point.x,point.y),2,(255,0,255),1)
            # 绘制框框
            l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
            
            now = time.time()

            if  (now - show_time) < 0.5:
                frame = cv2AddChineseText(frame, "注册成功 {count}/{faceCount}".format(count=(count+1),faceCount=faceCount) ,  (l, b+30), textColor=(255, 0, 255), textSize=40)


            # 检查次数
            if count < faceCount:

                # 检查时间
                
                if now - startTime > interval:

                    # 特征描述符
                    face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)
                    
                    face_descriptor = [f for f in face_descriptor ]

                    # 描述符增加进data文件
                    line = [faceId,userName,face_descriptor]
                    # 写入
                    csv_writer.writerow(line)
                    
                    # 保存照片样本

                    print('人脸注册成功 {count}/{faceCount}，faceId:{faceId}，userName:{userName}'.format(count=(count+1),faceCount=faceCount,faceId=faceId,userName=userName))

                    frame = cv2AddChineseText(frame, "注册成功 {count}/{faceCount}".format(count=(count+1),faceCount=faceCount) ,  (l, b+30), textColor=(255, 0, 255), textSize=40)
                    show_time = time.time()


                    # 时间重置
                    startTime = now
                    # 次数加一
                    count +=1
                
                
            else:
                print('人脸注册完毕')
                return

            
            
            # 只取其中一张脸
            break

        now = time.time()
        fpsText = 1 / (now - frameTime)
        frameTime = now
        # 绘制
        drawLeftInfo(frame,fpsText,'Register')

        cv2.imshow('Face Attendance Demo: Register',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

            
    f.close()
    cap.release()
    cv2.destroyAllWindows()

# 刷新右侧考勤信息
def updateRightInfo(frame,face_info_list,face_img_list):
    
    # 重新绘制逻辑：从列表中每隔3个取一批显示，新增人脸放在最前面
    
    # 如果有更新，重新绘制

    # 如果没有，定时往后移动

    left_x = 30
    left_y = 20
    resize_w = 80

    offset_y = 120
    index = 0

    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    

    for face in face_info_list[:3]:
        name = face[0]
        time = face[1]
        face_img = face_img_list[index]

        # print(face_img.shape)
        
        face_img = cv2.resize(face_img,(resize_w,resize_w))
        
        offset_y_value = offset_y * index
        frame[(left_y+offset_y_value):(left_y+resize_w+ offset_y_value), -(left_x+resize_w):-left_x] = face_img
        
        cv2.putText(frame, name, ((frame_w-(left_x+resize_w)), (left_y+resize_w)+15 + offset_y_value), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, time, ((frame_w-(left_x+resize_w)), (left_y+resize_w)+30 + offset_y_value), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)

        index+=1

    return frame

# 返回DLIB格式的face
def getDlibRect(detector='hog',face=None):
   
    l,t,r,b = None,None,None,None

    if detector == 'hog':
        l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
        

    if detector == 'cnn':
        l = face.rect.left()
        t = face.rect.top()
        r = face.rect.right()
        b = face.rect.bottom()

        
    if detector == 'haar':
        l = face[0]
        t = face[1]
        r = face[0] + face[2]
        b = face[1] + face[3]
        
    nonnegative = lambda x : x if x >= 0 else 0  
    return map(nonnegative,(l,t,r,b ))

# 获取CSV中信息
def getFeatList():
    print('加载注册的人脸特征')

    feature_list = None
    label_list = []
    name_list = []

    # 加载保存的特征样本
    with open('./data/feature.csv','r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            # 重新加载数据
            faceId = line[0]
            userName = line[1]
            face_descriptor = eval(line[2])

            label_list.append(faceId)
            name_list.append(userName)
            
            # 转为numpy格式
            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)
            # 转为二维矩阵，拼接
            face_descriptor = np.reshape(face_descriptor,(1,-1))
            # 初始化
            if feature_list is None:
                feature_list = face_descriptor
            else:
                # 拼接
                feature_list = np.concatenate((feature_list,face_descriptor),axis=0)
    print("特征加载完毕")
    return feature_list,label_list,name_list

# 人脸识别
def faceRecognize(detector='haar',threshold=0.5, write_video = False,resize_w=0,resize_h=0):

    

   
    # 视频时间
    frameTime = time.time()

    # 加载特征
    feature_list,label_list,name_list = getFeatList()

    face_time_dict = {}
    # 保存name,time人脸信息
    face_info_list = []
    # numpy格式人脸图像数据
    face_img_list = []

    # 侦测人数
    person_detect = 0

    # 统计人脸数
    face_count = 0

    # 控制显示打卡成功的时长
    show_time = (frameTime - 10)

    # 考勤记录
    f = open('./data/attendance.csv','a')
    csv_writer = csv.writer(f)

    cap = cv2.VideoCapture(0)


    if resize_w == 0 or resize_h == 0:
        resize_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
        resize_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) //2

    videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (resize_w,resize_h))

    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame,(resize_w,resize_h))
        frame = cv2.flip(frame,1)
        
        
        # 切换人脸检测器
        if detector == 'hog':
            face_detetion = hog_face_detector(frame,1)
        if detector == 'cnn':
            face_detetion = cnn_detector(frame,1)
        if detector == 'haar':   
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
            face_detetion = haar_face_detector.detectMultiScale(frame_gray,minNeighbors=7,minSize=(100,100))

        person_detect = len(face_detetion)

        for face in face_detetion:
            
           
            l,t,r,b = getDlibRect(detector,face)

            face = dlib.rectangle(l,t,r,b) 

            # 识别68个关键点
            points = points_detector(frame,face)

            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)

            # 人脸区域
            face_crop = frame[t:b,l:r]
            

            #特征
            face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)   
            face_descriptor = [f for f in face_descriptor ]
            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)

            # 计算距离
            distance = np.linalg.norm((face_descriptor-feature_list),axis = 1)
            # 最小距离索引
            min_index = np.argmin(distance)
            # 最小距离
            min_distance = distance[min_index]

            predict_name = "Not recog"

            if min_distance < threshold:
                # 距离小于阈值，表示匹配
                predict_id = label_list[min_index]
                predict_name = name_list[min_index]

                 # 判断是否新增记录：如果一个人距上次检测时间>3秒，或者换了一个人，将这条记录插入
                need_insert = False
                now = time.time()
                if predict_name in face_time_dict:
                    if (now - face_time_dict[predict_name]) > 3:
                        # 刷新时间
                        face_time_dict[predict_name] = now
                        need_insert = True
                    else:
                        # 还是上次人脸
                        need_insert = False

                else:
                    # 新增数据记录
                    face_time_dict[predict_name] = now
                    need_insert = True
                    
                if  (now - show_time) < 1:
                    frame = cv2AddChineseText(frame, "打卡成功" ,  (l, b+30), textColor=(0, 255, 0), textSize=40)    
                
                
                if need_insert:

                    # 连续显示打卡成功1s
                    frame = cv2AddChineseText(frame, "打卡成功" ,  (l, b+30), textColor=(0, 255, 0), textSize=40)    
                    show_time = time.time()


                    time_local = time.localtime(face_time_dict[predict_name])
                    #转换成新的时间格式(2016-05-05 20:28:54)
                    face_time = time.strftime("%H:%M:%S",time_local)
                    face_time_full = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

                    # 开始位置增加

                    face_info_list.insert(0,[predict_name,face_time])
                    face_img_list.insert( 0, face_crop )

                    # 写入考勤表
                    line = [predict_id,predict_name,min_distance,face_time_full]
                    csv_writer.writerow(line)

                    face_count+=1


            # 绘制人脸点
            cv2.putText(frame, predict_name + " " + str(round(min_distance,2)) , (l, b+30), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)
            
            # 处理下一张脸



        now = time.time()
        fpsText = 1 / (now - frameTime)
        frameTime = now
        # 绘制
        drawLeftInfo(frame,fpsText,'Recog',detector=detector,person=person_detect, count=face_count)

        # 舍弃face_img_list、face_info_list后部分，节约内存
        if len(face_info_list) > 10:
            face_info_list = face_info_list[:9]
            face_img_list = face_img_list[:9]

        frame = updateRightInfo(frame,face_info_list,face_img_list)

        if write_video:
            videoWriter.write(frame)

        cv2.imshow('Face Attendance Demo: Recognition',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

            
    f.close()
    videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()


# 参数
parser = ArgumentParser()
parser.add_argument("--mode", type=str, default='reg',
                    help="运行模式：reg,recog 对应：注册人脸、识别人脸")
parser.add_argument("--id", type=int, default=1,
                    help="人脸ID，正整数不可以重复")   

parser.add_argument("--name", type=str, default='enpei',
                    help="人脸姓名，英文格式")                       

parser.add_argument("--interval", type=int, default=3,
                    help="注册人脸每张间隔时间")    

parser.add_argument("--count", type=int, default=3,
                    help="注册人脸照片数量")        
                   

parser.add_argument("--w", type=int, default=0,
                    help="画面缩放宽度")   
parser.add_argument("--h", type=int, default=0,
                    help="画面缩放高度")  

parser.add_argument("--detector", type=str, default='haar',
                    help="人脸识别使用的检测器，可选haar、hog、cnn")     

parser.add_argument("--threshold", type=float, default=0.5,
                    help="人脸识别距离阈值，越低越准确")                                               

parser.add_argument("--record", type=bool, default=False,
                    help="人脸识别是否录制")                                               

args = parser.parse_args()



mode = args.mode

if mode == 'reg':
    faceRegiser(faceId=args.id,userName=args.name,interval=args.interval,faceCount=args.count,resize_w=args.w,resize_h=args.h)    

if  mode == 'recog':
    faceRecognize(detector=args.detector,threshold=args.threshold, write_video = args.record,resize_w=args.w,resize_h=args.h)