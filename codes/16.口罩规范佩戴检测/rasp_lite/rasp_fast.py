# 导入相关包
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
# from scipy.special import softmax
import time
import sys
from fdlite.face_detection import FaceDetection, FaceDetectionModel


class MaskDetection:

    """
    口罩检测：正常、未佩戴、不规范（漏鼻子）
    可运行在树莓派
    """

    def __init__(self,mode='rasp'):
        """
        构造函数
        """
        # 加载人脸检测模型
        
        # 加载口罩模型

        self.interpreter = tflite.Interpreter(model_path="./data/face_mask_model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 标签
        self.labels = ['正常','未佩戴','不规范']
        # 颜色，BGR顺序，绿色、红色、黄色
        self.colors = [(0,255,0),(0,0,255),(0,255,255)]

        # 中文label图像
        self.zh_label_img_list = self.getPngList()


    def getPngList(self):
        """
        获取PNG图像列表

        @return numpy array list
        """
        overlay_list = []
        # 遍历文件
        for i in range(3):
            fileName = './label_img/%s.png' % (i)
            overlay = cv2.imread(fileName,cv2.COLOR_RGB2BGR)
            overlay = cv2.resize(overlay,(0,0), fx=0.3, fy=0.3)
            overlay_list.append(overlay)

        return overlay_list


    
    def imageProcess(self,face_region):
        """
        将图像转为blob

        @param: face_region numpy arr 输入的numpy图像
        @return: blob或None 
        """
        
        if face_region is not None:
            # blob处理
            blob = cv2.dnn.blobFromImage(face_region,1,(100,100),(104,117,123),swapRB=True)
            blob_squeeze = np.squeeze(blob).T
            blob_rotate = cv2.rotate(blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
            blob_flip = cv2.flip(blob_rotate,1)
            blob_norm = np.maximum(blob_flip,0) / blob_flip.max()
            # face_resize = cv2.resize(face_region,(100,100))
            return blob_norm
        else:
            return None

    def detect(self):
        """
        识别
        """

        # 人脸检测器
        detect_faces = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)
        
        # 获取视频流
        cap = cv2.VideoCapture(0)

        # 视频宽度和高度
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



        # 记录帧率时间
        frameTime = time.time()


        

        while True:
            # 读取
            ret,frame = cap.read()
            # 翻转
            frame = cv2.flip(frame,1)
            # 检测
            faces = detect_faces(frame)
            # 记录人数
            person_count = 0

            if not len(faces):
                print('no faces detected :(')
            else:
                
                # 遍历多个人脸
                for face in faces:

                    person_count+=1

                    l,t,r,b = ([face.bbox.xmin,face.bbox.ymin,face.bbox.xmax,face.bbox.ymax] * np.array([frame_w,frame_h,frame_w,frame_h])).astype(int)
                    
                    
                    t -= 20
                    b += 20
                    # 越界处理
                    if l<= 0 or t <=0 or r >= frame_w or b >= frame_h :
                        continue

                    # 人脸区域
                    face_region = frame[t:b,l:r]
                    # 转为blob
                    blob_norm = self.imageProcess(face_region)

                    if blob_norm is not None:
                        # 预测
                        img_input = blob_norm.reshape(1,100,100,3)

                        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)

                        self.interpreter.invoke()

                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        result = output_data
                        # 最大值索引
                        # softmax处理
                        # result = softmax(result)[0]
                        # 最大值索引
                        max_index = result[0].argmax()
                        # 最大值
                        # max_value = result[0][max_index]
                        # 标签
                        label = self.labels[max_index]

                        # 中文标签
                        overlay = self.zh_label_img_list[max_index]
                        overlay_h,overlay_w = overlay.shape[:2]

                        # 覆盖范围
                        overlay_l,overlay_t = l,(t - overlay_h-20)
                        overlay_r,overlay_b = (l + overlay_w),(overlay_t+overlay_h)

                        # 判断边界
                        if overlay_t > 0 and overlay_r < frame_w:
                            
                            overlay_copy=cv2.addWeighted(frame[overlay_t:overlay_b, overlay_l:overlay_r ],1,overlay,20,0)
                            frame[overlay_t:overlay_b, overlay_l:overlay_r ] = overlay_copy

                            # cv2.putText(frame, str(round(max_value*100,2))+"%", (overlay_r+20, overlay_t+40), cv2.FONT_ITALIC, 0.8, self.colors[max_index], 2)

                    # 人脸框
                    cv2.rectangle(frame,(l,t),(r,b),self.colors[max_index],5)


            now = time.time()
            fpsText = 1 / (now - frameTime)
            frameTime = now

            cv2.putText(frame, "FPS:  " + str(round(fpsText,2)), (50, 60), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "Person:  " + str(person_count), (50, 110), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 3)


            cv2.imshow('demo',frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        


mask_detection = MaskDetection()
mask_detection.detect()