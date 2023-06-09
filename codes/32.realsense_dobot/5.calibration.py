# realsense and dobot calibration

import pyrealsense2 as rs
# import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
from serial.tools import list_ports
from pydobot import Dobot
import os
import time

class Calibration:
    def __init__(self):
        # initialize dobot
        port = list_ports.comports()[0].device
        self.device = Dobot(port=port, verbose=True)
        (x, y, z, r, j1, j2, j3, j4) = self.device.pose()
        print("#################dobot pose")
        print("x: {}, y: {}, z: {}, r: {}, j1: {}, j2: {}, j3: {}, j4: {}".format(x, y, z, r, j1, j2, j3, j4))
        # move to calibration position
        self.device.move_to(x=200, y=0, z=0, r=0, wait=True)
        

        # initialize realsense
        # Create a context object. This object owns the handles to all connected realsense devices
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 分别是宽、高、数据格式、帧率
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        
        # Start streaming
        self.pipeline.start(config)

        # dictionary used in ArUco markers
        self.dictionary = aruco.Dictionary_get(aruco.DICT_5X5_50)
        # create parameters object
        self.parameters = aruco.DetectorParameters_create()

    def get_aruco_center(self, calib = True):
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        # get depth frame
        depth = frames.get_depth_frame()
        
        # display color frame
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        cv2.putText(color_image, "Status: Calibrating", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,  255, 0), 2)

        # 获取intelrealsense参数
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        # 内参矩阵，转ndarray方便后续opencv直接使用
        intr_matrix = np.array([
            [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
        ]) 
        intr_coeffs = np.array(intr.coeffs)
        # 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
        corners, ids, rejected_img_points = aruco.detectMarkers(color_image, self.dictionary, parameters=self.parameters)

        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, intr_matrix, intr_coeffs)


        center = None
        # if markers are detected
        if ids is not None:
            # draw borders around markers
            aruco.drawDetectedMarkers(color_image, corners)
            # draw axis around markers, parameters: image, camera internal parameters, distortion parameters, rotation vector, translation vector, length of axis line
            cv2.drawFrameAxes(color_image, intr_matrix, intr_coeffs, rvec, tvec, 0.05) 
            # print ids and corners of detected markers
            for i, corner in zip(ids, corners):
                # get aruco center coordinate
                if calib:
                    x = (corner[0][0][0] + corner[0][3][0]) / 2
                    y = (corner[0][0][1] + corner[0][3][1]) / 2
                else:
                    x = (corner[0][0][0] + corner[0][2][0]) / 2
                    y = (corner[0][0][1] + corner[0][2][1]) / 2
                    
                cv2.circle(color_image, (int(x), int(y)), 3, (0, 0, 255), -1)

                # get middle pixel distance
                dist_to_center = depth.get_distance(int(x), int(y))
                # realsense提供的方法，将像素坐标转换为相机坐标系下的坐标
                x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(intr, [x, y], dist_to_center)
                # display_txt = "x: {:.3f}, y: {:.3f}, z: {:.3f}".format(x_cam, y_cam, z_cam)
                cv2.putText(color_image, "x: {:.3f}m".format(x_cam), (int(x) + 50, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,  255, 0), 2)
                cv2.putText(color_image, "y: {:.3f}m".format(y_cam), (int(x) + 50, int(y) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,  255, 0), 2)
                cv2.putText(color_image, "z: {:.3f}m".format(z_cam), (int(x) + 50, int(y) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,  255, 0), 2)
                # print(display_txt)


                center = [x_cam, y_cam, z_cam]
                
                # just need one marker
                break

        # depth frame
        depth_img = np.asanyarray(depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.219), cv2.COLORMAP_JET)

        # stack color frame and depth frame
        images = np.vstack((color_image, depth_colormap))
        return images, center
        
    def run_calibration(self):
        # dobot move to different position
        # get the position of the aruco marker
        # calculate the transform matrix
        # save the transform matrix
        # test the transform matrix
        self.device.suck(enable=True)
        print("#################please put the aruco marker on the dobot end effector")
        time.sleep(20)

        # define move points, x, y, z, r
        default_cali_points = [
                           [180, -150, -60, 0], [220, -80, -60, 0],
                           [230, 0, -60, 0],
                           [285, 0, -60, 0],
                           [260, -100, -60, 0],  [220, -180, -60, 0],
                           [250, -190, -60, 0], [290, -110, -60, 0],
                           [305, 0, -60, 0],

                           [230, 0, -40, 0],
                           [220, -80, -40, 0],  [180, -150, -40, 0],
                           [220, -180, -40, 0], [260, -100, -40, 0],
                           [285, 0, -40, 0],
                           [305, 0, -40, 0],
                           [290, -110, -40, 0],  [250, -190, -40, 0],

                           [180, -150, -20, 0], [220, -80, -20, 0],
                           [230, 0, -20, 0],
                           [285, 0, -20, 0],
                           [260, -100, -20, 0],  [220, -180, -20, 0],
                           [250, -190, -20, 0], [290, -110, -20, 0],
                           [305, 0, -20, 0],

                           [230, 0, 5, 0],
                           [220, -80, 0, 0],  [180, -150, 0, 0],
                           [220, -180, 0, 0], [260, -100, 0, 0],
                           [285, 0, 0, 0],
                           [305, 0, 0, 0],
                           [290, -110, 0, 0],  [250, -190, 0, 0],

                           [180, -150, 20, 0], [220, -80, 20, 0],
                           [230, 0, 20, 0],
                           [285, 0, 20, 0],
                           [260, -100, 20, 0],  [220, -180, 20, 0],
                           [250, -190, 20, 0], [290, -110, 20, 0],
                           [305, 0, 20, 0],

                           [230, 0, 40, 0],
                           [220, -80, 40, 0],  [180, -150, 40, 0],
                           [220, -180, 40, 0], [260, -100, 40, 0],
                           [285, 0, 40, 0],
                           [305, 0, 40, 0],
                           [290, -110, 40, 0],  [250, -190, 40, 0],]
                           
        np_cali_points = np.array(default_cali_points)
        arm_cord = np.column_stack(
            (np_cali_points[:, 0:3], np.ones(np_cali_points.shape[0]).T)).T 
        centers = np.ones(arm_cord.shape)
        
        img_to_arm_file = "./save_parms/image_to_arm.npy"
        arm_to_img_file = "./save_parms/arm_to_image.npy"

        if os.path.exists(img_to_arm_file) and os.path.exists(arm_to_img_file):
            image_to_arm = np.load(img_to_arm_file)
            arm_to_image = np.load(arm_to_img_file)
            print("load image to arm and arm to image transform matrix")
        else:
            print("need to calibrate the camera and dobot")
            for index, point in enumerate(default_cali_points):
                print("#################dobot move to point {}, x: {}, y: {}, z: {}, r: {}".format(index, point[0], point[1], point[2], point[3]))
                self.device.speed(100, 100 )
                # move to the point
                self.device.move_to(point[0], point[1], point[2], point[3], wait=True)
                # add x offset
                arm_cord.T[index][0] = arm_cord.T[index][0] + 40 # +40 因为 aruco marker 的中心点距离end effector 30mm

                # get the position of the aruco marker
                images, center = self.get_aruco_center()
                if center is not None:
                    # save the center
                    centers[0:3, index] = center                
                    # display the image
                    cv2.imshow("image", images)
                    cv2.waitKey(1)
                else:
                    print("no aruco marker detected")
                    continue

                time.sleep(1)

        # calculate the transform matrix
        image_to_arm = np.dot(arm_cord, np.linalg.pinv(centers))
        arm_to_image = np.linalg.pinv(image_to_arm)
        print("Finished calibration!")

        print("Image to arm transform:\n", image_to_arm)
        print("Arm to Image transform:\n", arm_to_image)
        # write to file
        np.save(img_to_arm_file, image_to_arm)
        np.save(arm_to_img_file, arm_to_image)

        print("Sanity Test:")

        print("-------------------")
        print("Image_to_Arm")
        print("-------------------")
        for ind, pt in enumerate(centers.T):
            print("Expected:", arm_cord.T[ind][0:3])
            print("Result:", np.dot(image_to_arm, np.array(pt))[0:3])

        print("-------------------")
        print("Arm_to_Image")
        print("-------------------")
        for ind, pt in enumerate(arm_cord.T):
            print("Expected:", centers.T[ind][0:3])
            pt[3] = 1
            print("Result:", np.dot(arm_to_image, np.array(pt))[0:3])

    def run_recog(self):
        if os.path.exists("./save_parms/image_to_arm.npy"):
            image_to_arm = np.load("./save_parms/image_to_arm.npy")
        self.device.suck(enable=False)
        time.sleep(3)
        while True:
            images, center = self.get_aruco_center( calib = False)
            if center is not None:
                cv2.imwrite("save.jpg", images)
                cv2.imshow("image", images)
                cv2.waitKey(1)
                img_pos = np.ones(4)
                img_pos[0:3] = center
                arm_pos = np.dot(image_to_arm, np.array(img_pos))
                print(arm_pos)
                if (np.sqrt(arm_pos[0]*arm_pos[0] + arm_pos[1]*arm_pos[1]) > 300):
                    print("Can not reach!!!!!!!!!!!!!!!")
                    time.sleep(3)
                    continue
                self.device.speed(100, 100)
                self.device.suck(enable=True)
                self.device.move_to(arm_pos[0], arm_pos[1], arm_pos[2]+20, 0, wait=True)
                self.device.speed(50, 50)
                self.device.move_to(arm_pos[0], arm_pos[1], arm_pos[2]-3, 0, wait=True)
                self.device.speed(100, 100)
                # x range: 140 - 300,  
                self.device.move_to(140, -180, 90, 0, wait=True)
                self.device.suck(enable=False)
                time.sleep(5)
                print("another one")
if __name__ == "__main__":
    cali = Calibration()
    if not os.path.exists("./save_parms/image_to_arm.npy") or not os.path.exists("./save_parms/arm_to_image.npy"):
        cali.run_calibration()
    cali.run_recog()