from face_detection import FaceDetection, FaceDetectionModel

import cv2
import numpy as np
import time

        
cap = cv2.VideoCapture(0)

detect_faces = FaceDetection(model_type=FaceDetectionModel.FULL_SPARSE)


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


