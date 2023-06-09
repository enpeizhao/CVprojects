import ctypes
import cv2
import numpy as np
import time
import os

# 多进程
from multiprocessing import Process, Value, Queue, Array

# import pyrealsense2 as rs
import pyrealsense2.pyrealsense2 as rs
import cv2.aruco as aruco
from serial.tools import list_ports
from pydobot import Dobot
import torch


def realsense_video(center_p_queue, dobot_status):
    # start another thread
    
    # initialize realsense
    # Create a context object. This object owns the handles to all connected realsense devices
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 分别是宽、高、数据格式、帧率
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)

    # Start streaming
    pipeline.start(config)
        
    # define range of green color in HSV
    lower_green = np.array([150, 0, 0])
    upper_green = np.array([170, 255, 255])
    
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    video_writer = cv2.VideoWriter('./output/9.recog_by_yolo.mp4', fourcc, 15, ( 640, 480*2))
    # set quality
    video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100) # 1

    alpha_val = 0.219
    print("alpha_val: ", alpha_val)
    

    model = torch.hub.load('./yolov5', 'custom', path='./weights/fruits_best.pt',source='local')  # local repo
    model.conf = 0.3
        
    try:
        while True:
            start_time = time.time()
            # clear queue
            while not center_p_queue.empty():
                center_p_queue.get()
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            # get depth frame
            depth = frames.get_depth_frame()

            # display color frame
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            # 获取intelrealsense参数
            intr = color_frame.profile.as_video_stream_profile().intrinsics

          

            img_cvt = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)

            # Inference
            results = model(img_cvt)
            result_np = results.pandas().xyxy[0].to_numpy()

            for box in result_np:
                l,t,r,b = box[:4].astype('int')
                label = box[5]
                label_text = ""
                box_color = (0, 255, 0)
                if label == 1:
                    label_text = "apple"
                    box_color = (0, 0, 255)
                    # get center of box
                    x =  (l + r) / 2
                    y =  (t + b) / 2 

                elif label == 3:
                    label_text = "banana"
                    box_color = (0, 255, 0)
                    x =  (l + r) / 2
                    y =  (t + b) / 2 
                elif label == 6:
                    label_text = "mongo"
                    box_color = (255, 0, 0)
                    x =  (l + r) / 2
                    y =  (t + b) / 2 

                cv2.rectangle(color_image,(l,t),(r,b),box_color,2)

                
                cv2.circle(color_image, (int(x), int(y)), 3, (0, 0, 255), -1)

                # get middle pixel distance
                dist_to_center = depth.get_distance(int(x), int(y))
                # realsense提供的方法，将像素坐标转换为相机坐标系下的坐标
                x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(
                    intr, [x, y], dist_to_center
                )
                # put text under box
                cv2.putText(color_image, label_text, (l, b+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                cv2.putText(color_image, "x: {:.3f}".format(x_cam), (l, b+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                cv2.putText(color_image, "y: {:.3f}".format(y_cam), (l, b+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                cv2.putText(color_image, "z: {:.3f}".format(z_cam), (l, b+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                # add to queue
                center_p_queue.put([x_cam, y_cam, z_cam])

                

            # depth frame
            depth_img = np.asanyarray(depth.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=alpha_val), cv2.COLORMAP_JET
            )

            # fps
            end_time = time.time()
            fps_text = "FPS:  {:.2f}".format(1/(end_time - start_time))
            cv2.putText(color_image, fps_text, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # put robot status
            if dobot_status.value == 0:
                status_text = "Status: Searching"
                cv2.putText( color_image, status_text, (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,)
            else:
                status_text = "Status: Running"
                cv2.putText( color_image, status_text, (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,)

            # stack color frame and depth frame
            images = np.vstack((color_image, depth_colormap))
            # return images, center
            cv2.imshow("image", images)
            video_writer.write(images)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()


def dobot_grasp(center_p_queue, dobot_status):
    if os.path.exists("./save_parms/image_to_arm.npy"):
        image_to_arm = np.load("./save_parms/image_to_arm.npy")
    else:
        print("image_to_arm.npy not exist")
        return
    # initialize dobot
    port = list_ports.comports()[0].device
    device = Dobot(port=port, verbose=True)
    (x, y, z, r, j1, j2, j3, j4) = device.pose()
    print("#################dobot pose")
    print(
        "x: {}, y: {}, z: {}, r: {}, j1: {}, j2: {}, j3: {}, j4: {}".format(
            x, y, z, r, j1, j2, j3, j4
        )
    )
    # move to calibration position
    device.move_to(x=0, y=-261, z=116, r=0, wait=True)
    device.suck(enable=False)
    time.sleep(3)

    dobot_status.value = 0
    
    while True:
        
        # images, center = get_aruco_center( calib = False)
        if not dobot_status.value:
            # set to running status
            dobot_status.value = 1

            # get value from queue
            center_p = center_p_queue.get()
            center = center_p[:]
            
            img_pos = np.ones(4)
            img_pos[0:3] = center
            arm_pos = np.dot(image_to_arm, np.array(img_pos))
            print(arm_pos)
            if np.sqrt(arm_pos[0] * arm_pos[0] + arm_pos[1] * arm_pos[1]) > 320:
                print("Can not reach!!!!!!!!!!!!!!!, distance: {}".format(np.sqrt(arm_pos[0] * arm_pos[0] + arm_pos[1] * arm_pos[1])))

                time.sleep(3)
                continue
            device.speed(100, 100)
            device.suck(enable=True)
            device.move_to(177, -150, 164, 0, wait=False) # corner
            device.move_to(arm_pos[0], arm_pos[1], arm_pos[2]+20, 0, wait=False) # 20mm above the object
            device.speed(50, 50) # slow down
            device.move_to(arm_pos[0], arm_pos[1], arm_pos[2]-2, 0, wait=False) # 1mm above the object
            device.speed(100, 100) # speed up
            device.move_to(arm_pos[0], arm_pos[1], arm_pos[2]+20, 0, wait=False) # 20mm above the object
            # x range: 140 - 300,
            device.move_to(177, -150, 164, 0, wait=False) # corner
            device.move_to(x=34, y=-228, z=152, r=0, wait=True) 
            device.suck(enable=False) 
            print("another one")
            # time.sleep(1)

            # reset to search status
            dobot_status.value = 0

            
        else:
            print("no marker detected")
            time.sleep(3)
            # reset to search status
            dobot_status.value = 0
            continue


if __name__ == "__main__":
    # process1
    center_arr = Array(ctypes.c_double, [0, 0, 0])
    dobot_status = Value(ctypes.c_int8, 0) # 0: stop, 1: running
    center_p_queue = Queue() # 用于进程间通信, 传递center_p
    process1 = Process(target=realsense_video, args=(center_p_queue, dobot_status))
    process2 = Process(target=dobot_grasp, args=(center_p_queue, dobot_status))
    process1.start()
    # process2.start()
    process1.join()
    # process2.join()

