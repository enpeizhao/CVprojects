# 导入相关包
import cv2
import numpy as np
import tensorflow as tf
# from scipy.special import softmax
import time



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
        self.mask_model = tf.keras.models.load_model('./data/face_mask_model.h5')
        print(self.mask_model.summary())
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
            # 对于图像一般不用附属，所以移除
            # 为了后面算法更好的收敛学习，并且归一化处理
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
        face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt','./weights/res10_300x300_ssd_iter_140000.caffemodel')
        
        # 获取视频流
        cap = cv2.VideoCapture(0)

        # 视频宽度和高度
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        # 记录帧率时间
        frameTime = time.time()

        videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), 10, (960,720))

        

        while True:
            # 读取
            ret,frame = cap.read()
            # 翻转
            frame = cv2.flip(frame,1)
            # 识别前缩放
            frame_resize = cv2.resize(frame,(300,300))

            # 人脸检测
            img_blob = cv2.dnn.blobFromImage(frame_resize,1.0,(300,300),(104.0, 177.0, 123.0),swapRB=True)
            # 输入
            face_detector.setInput(img_blob)
            # 推理
            detections = face_detector.forward()
            # 人数
            num_of_detections = detections.shape[2]

            # 记录人数
            person_count = 0

            # 遍历多个人
            for index in range(num_of_detections):
                # 置信度
                detection_confidence = detections[0,0,index,2]
                # 挑选置信度
                if detection_confidence>0.5:

                    person_count+=1

                    # 位置
                    locations = detections[0,0,index,3:7] * np.array([frame_w,frame_h,frame_w,frame_h])
                    l,t,r,b  = locations.astype('int')
                    # 人脸区域
                    face_region = frame[t:b,l:r]
                    # 转为blob
                    blob_norm = self.imageProcess(face_region)

                    if blob_norm is not None:
                        # 预测
                        img_input = blob_norm.reshape(1,100,100,3)

                        result = self.mask_model.predict(img_input)
                        # softmax处理
                        result = tf.nn.softmax(result[0]).numpy()

                        # 最大值索引
                        max_index = result.argmax()
                        # 最大值
                        max_value = result[max_index]
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

                            cv2.putText(frame, str(round(max_value*100,2))+"%", (overlay_r+20, overlay_t+40), cv2.FONT_ITALIC, 0.8, self.colors[max_index], 2)

                    # 人脸框
                    cv2.rectangle(frame,(l,t),(r,b),self.colors[max_index],5)


            now = time.time()
            fpsText = 1 / (now - frameTime)
            frameTime = now

            cv2.putText(frame, "FPS:  " + str(round(fpsText,2)), (50, 60), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "Person:  " + str(person_count), (50, 110), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 3)

            videoWriter.write(frame)

            cv2.imshow('demo',frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        videoWriter.release()
        cap.release()
        cv2.destroyAllWindows()
        


mask_detection = MaskDetection()
mask_detection.detect()