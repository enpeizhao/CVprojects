"""
! author: enpei
! date: 2021-12-23
主要功能：检测孩子是否在看电视，看了多久，距离多远
使用技术点：人脸检测、人脸识别（采集照片、训练、识别）、姿态估计
"""
import cv2,time
from pose_estimator import PoseEstimator
import numpy as np
import dlib
from utils import Utils
import os
from argparse import ArgumentParser


class MonitorBabay:
    def __init__(self):
        # 人脸检测
        self.face_detector = dlib.get_frontal_face_detector()
        # 人脸识别模型：pip uninstall opencv-python，pip install opencv-contrib-python
        self.face_model = cv2.face.LBPHFaceRecognizer_create()

        # 人脸68个关键点
        self.landmark_predictor = dlib.shape_predictor("./assets/shape_predictor_68_face_landmarks.dat")

        # 站在1.5M远处，左眼最左边距离右眼最右边的像素距离(请使用getEyePixelDist方法校准，然后修改这里的值)
        self.eyeBaseDistance = 65

        # pose_estimator.show_3d_model()

        self.utils = Utils()


    # 采集照片用于训练
    # 参数
    # label_index: label的索引
    # save_interval：隔几秒存储照片
    # save_num：存储总量
    def collectFacesFromCamera(self,label_index,save_interval,save_num):
        cap = cv2.VideoCapture(0)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fpsTime = time.time()
        last_save_time = fpsTime
        saved_num = 0
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector(gray)
            
            for face in faces:

                if saved_num < save_num:
                    if (time.time() - last_save_time) > save_interval:
                        self.utils.save_face(face,frame,label_index)
                        saved_num +=1
                        last_save_time = time.time()

                        print('label_index:{index}，成功采集第{num}张照片'.format(index = label_index,num = saved_num))
                else:
                    print('照片采集完毕！')
                    exit()

                self.utils.draw_face_box(face,frame,'','','')    

            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime
            
            frame = self.utils.cv2AddChineseText(frame, "帧率: " + str(int(fps_text)),  (10, 30), textColor=(0, 255, 0), textSize=50)
            frame = cv2.resize(frame, (int(width)//2, int(height)//2) )
            cv2.imshow('Collect data', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


    # 训练人脸模型
    def train(self):
        print('训练开始！')
        label_list,img_list = self.utils.getFacesLabels()
        self.face_model.train(img_list, label_list)
        self.face_model.save("./face_model/model.yml")
        print('训练完毕！')
    

    
    # 获取两个眼角像素距离
    def getEyePixelDist(self):
        
        cap = cv2.VideoCapture(0)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 姿态估计
        self.pose_estimator = PoseEstimator(img_size=(height, width))
        
        fpsTime = time.time()

        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector(gray)
           
            pixel_dist = 0

            for face in faces:
                
                # 关键点
                landmarks = self.landmark_predictor(gray, face)
                image_points = self.pose_estimator.get_image_points(landmarks)

                left_x = int(image_points[36][0])
                left_y = int(image_points[36][1])
                right_x = int(image_points[45][0])
                right_y = int(image_points[45][1])

                pixel_dist = abs(right_x-left_x)

                cv2.circle(frame, (left_x, left_y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (right_x, right_y), 8, (255, 0, 255), -1)

                # 人脸框
                frame = self.utils.draw_face_box(face,frame,'','','')
              

            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime
            
            frame = self.utils.cv2AddChineseText(frame, "帧率: " + str(int(fps_text)),  (20, 30), textColor=(0, 255, 0), textSize=50)
            frame = self.utils.cv2AddChineseText(frame, "像素距离: " + str(int(pixel_dist)),  (20, 100), textColor=(0, 255, 0), textSize=50)
           
            # frame = cv2.resize(frame, (int(width)//2, int(height)//2) )
            cv2.imshow('Baby wathching TV', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

    # 运行主程序
    def run(self,w,h,display):

        model_path = "./face_model/model.yml"
        if not os.path.exists(model_path):
            print('人脸识别模型文件不存在，请先采集训练')
            exit()

        label_zh = self.utils.loadLablZh()

        self.face_model.read(model_path)

        cap = cv2.VideoCapture(0)

        width = w
        height = h

        print(width,height)

        # 姿态估计
        self.pose_estimator = PoseEstimator(img_size=(height, width))

        fpsTime = time.time()

        zh_name = ''
        x_label = ''
        z_label = ''
        is_watch = ''
        angles = [0,0,0]
        person_distance = 0
        watch_start_time = fpsTime
        watch_duration = 0

        # fps = 12
        # videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (width,height))

        while True:
            _, frame = cap.read()
            frame = cv2.resize(frame,(width,height))
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector(gray)
            
            

            for face in faces:
                
                x1,y1,x2,y2 = self.utils.getFaceXY(face)
                face_img = gray[y1:y2,x1:x2]

                try:
                    # 人脸识别
                    idx, confidence = self.face_model.predict(face_img)
                    zh_name = label_zh[str(idx)]
                except cv2.error:
                    print('cv2.error')

                # 关键点
                landmarks = self.landmark_predictor(gray, face)
                # 计算旋转矢量
                rotation_vector, translation_vector = self.pose_estimator.solve_pose_by_68_points(landmarks)

                # 计算距离
                person_distance = round(self.pose_estimator.get_distance(self.eyeBaseDistance),2)


                # 计算角度
                rmat, jac = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                
                
                if angles[1] < -15:
                    x_label = '左'
                elif angles[1] > 15:
                    x_label = '右'
                else:
                    x_label = '前'


                if angles[0] < -15:
                    z_label = "下"
                elif angles[0] > 15:
                    z_label = "上"
                else:
                    z_label = "中"

                is_watch = '是' if( x_label =='前' and z_label == '中') else '否'

                if is_watch == '是':
                    now = time.time()
                    watch_duration += ( now - watch_start_time)
                
                watch_start_time= time.time()
                
                if display == 1:
                    # 人脸框
                    frame = self.utils.draw_face_box(face,frame,zh_name,is_watch,person_distance)
                if display == 2:
                    # 68个关键点
                    self.utils.draw_face_points(landmarks,frame)
                if display == 3:
                    # 梯形方向
                    self.pose_estimator.draw_annotation_box(
                        frame, rotation_vector, translation_vector,is_watch)
                if display == 4:
                    # 指针
                    self.pose_estimator.draw_pointer(frame, rotation_vector, translation_vector)
                if display == 5:
                    # 三维坐标系
                    self.pose_estimator.draw_axes(frame, rotation_vector, translation_vector)

                # 仅测试单人
                break

            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime
            
            frame = self.utils.cv2AddChineseText(frame, "帧率: " + str(int(fps_text)),  (20, 30), textColor=(0, 255, 0), textSize=50)

            color = (255, 0, 255) if person_distance <=1 else (0, 255, 0)

            frame = self.utils.cv2AddChineseText(frame, "距离: " + str(person_distance ) +"m",  (20, 100), textColor=color, textSize=50)

            color = (255, 0, 255) if is_watch =='是' else (0, 255, 0)


            frame = self.utils.cv2AddChineseText(frame, "观看: " + str(is_watch ),  (20, 170), textColor=color, textSize=50)

            # 
            duration_str = str(round((watch_duration/60),2)) +"min"

            frame = self.utils.cv2AddChineseText(frame, "时长: " + duration_str, (20, 240), textColor= (0, 255, 0), textSize=50)



            color = (255, 0, 255) if x_label =='前' else (0, 255, 0)
            
            frame = self.utils.cv2AddChineseText(frame, "X轴: {degree}° {x_label} ".format(x_label=str(x_label ),degree = str(int(angles[1]))) ,  (20, height-220), textColor=color, textSize=40)

            color = (255, 0, 255) if z_label =='中' else (0, 255, 0)

            frame = self.utils.cv2AddChineseText(frame, "Z轴: {degree}° {z_label}".format(z_label=str(z_label ),degree = str(int(angles[0]))) ,  (20, height-160), textColor=color, textSize=40)


            frame = self.utils.cv2AddChineseText(frame, "Y轴: {degree}°".format(degree = str(int(angles[2]) )),(20, height-100), textColor=(0, 255, 0), textSize=40)


            # videoWriter.write(frame) 
            # frame = cv2.resize(frame, (int(width)//2, int(height)//2) )
            cv2.imshow('Baby wathching TV', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

m = MonitorBabay()


# 参数
parser = ArgumentParser()
parser.add_argument("--mode", type=str, default='run',
                    help="运行模式：collect,train,distance,run对应：采集、训练、评估距离、主程序")
parser.add_argument("--label_id", type=int, default=1,
                    help="采集照片标签id.")
parser.add_argument("--img_count", type=int, default=3,
                    help="采集照片数量")        
parser.add_argument("--img_interval", type=int, default=3,
                    help="采集照片间隔时间")            
                    
parser.add_argument("--display", type=int, default=1,
                    help="显示模式，取值1-5")     
                     
parser.add_argument("--w", type=int, default=960,
                    help="画面宽度")   
parser.add_argument("--h", type=int, default=720,
                    help="画面高度")                           
args = parser.parse_args()


mode = args.mode

if mode == 'collect':
    print("即将采集照片.")
    if args.label_id and args.img_count and args.img_interval:
        m.collectFacesFromCamera(args.label_id,args.img_interval,args.img_count)

if  mode == 'train':
    m.train()

if  mode == 'distance':
    m.getEyePixelDist()

if mode == 'run':
    m.run(args.w,args.h,args.display)