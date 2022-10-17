'''
@author: enpei
@date:2022-09-29

mediapipe关键点提取
'''
import mediapipe as mp  # pip install mediapipe
import cv2


class Mpkeypoints:
    """
    获取人体Pose关键点
    """

    def __init__(self):
        
        # 实例化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def getFramePose(self, image):
        """
        获取每一帧画面的关键点
        """
        
        # 推理
        results = self.pose.process(image)

        return results.pose_landmarks
