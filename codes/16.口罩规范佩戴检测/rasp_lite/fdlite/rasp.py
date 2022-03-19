# 导入相关包
import cv2
import numpy as np
# 导入dlib
import dlib
# import tensorflow as tf
from scipy.special import softmax
import os
import time
# 加载模型
from face_detection import FaceDetection, FaceDetectionModel


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./data/face_mask.tflite")
# Load the TFLite model and allocate tensors.
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


    
def imageProcess(face_region):
    
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

cap = cv2.VideoCapture(0)
detect_faces = FaceDetection(model_type=FaceDetectionModel.FULL_SPARSE)

# 标签
labels = ['yes','no','nose']

print(labels)
# labels =  ['未佩戴','正常']

frameTime = time.time()
while True:

    ret,frame = cap.read()

    frame = cv2.flip(frame,1)
    
    
    
    
    faces = detect_faces(frame)
    if not len(faces):
        print('no faces detected :(')
    else:
        h,w = frame.shape[:2]
        for face in faces:

            l,t,r,b = ([face.bbox.xmin,face.bbox.ymin,face.bbox.xmax,face.bbox.ymax] * np.array([w,h,w,h])).astype(int)

            
            face_region = frame[t:b,l:r]
            blob_norm = imageProcess(face_region)

            if blob_norm is not None:
                # 预测
                img_input = blob_norm.reshape(1,100,100,3)

    
                interpreter.set_tensor(input_details[0]['index'], img_input)

                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                result = softmax(output_data)
                # 最大值索引
                max_index = result[0].argmax()
                # 最大值
                max_value = result[0][max_index]
                # 标签
                label = labels[max_index]
                print(label,max_value)

                # cv2.imshow('demo blob',blob_norm)

            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),5)
    
    
    now = time.time()
    fpsText = 1 / (now - frameTime)
    frameTime = now

    cv2.putText(frame, "FPS:  " + str(round(fpsText,2)), (30, 50), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

    cv2.imshow('demo',frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


