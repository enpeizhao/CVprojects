"""
手势识别操作APPLE TV
"""
# 导入opencv
import cv2
# 导入mediapipe
import mediapipe as mp
# 导入time
import time
import numpy as np
import random
from paho.mqtt import client as mqtt_client

import threading
from playsound import playsound


class handPlay():
    def __init__(self):
        # 设定高度、宽度
        self.videoW, self.videoH = 960, 540

        self.cap = cv2.VideoCapture(0)
        # 输入图像大小
        self.cap.set(3, self.videoW)
        self.cap.set(4, self.videoH)

        # 0，1分别表示控制指令未发送、已发送
        self.command_sent = 0
        # 动作计时比对
        self.action_start_time = time.time()

        # 动作检测间隔时间
        self.action_interval = 0.5

        # 记录FPS
        self.fps_time = time.time()

        # 记录指令次数
        self.command_index = 0
        # 记录指令间隔
        self.command_interval = 3
        self.command_start_time = time.time()

        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # mqtt
        self.broker = '192.168.1.165'
        self.mqtt_port = 1883
        self.mqtt_topic = "jetson/atv/pause"
        # generate client ID with pub prefix randomly
        self.client_id = f'python-mqtt-{random.randint(0, 1000)}'
        self.username = 'homeassistant'
        self.password = 'wae8fu1uKaeChooZaeweiyeiGhae8aigon7ooratheeghieDadae6lei1eeso7ae'

    # 连接MQTT
    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT self.broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        client = mqtt_client.Client(self.client_id)
        client.username_pw_set(self.username, self.password)
        client.on_connect = on_connect
        client.connect(self.broker, self.mqtt_port)
        return client

    # 检测是否出现暂停姿势
    def checkHandAction(self, landmark_list):
        # 暂停的动作，指尖在上：landmark_list中第4，8，12，16，20的y坐标大于3，7，11，15，19
        if (landmark_list[4][2] < landmark_list[3][2]
                and landmark_list[8][2] < landmark_list[7][2]
                and landmark_list[12][2] < landmark_list[11][2]
                and landmark_list[16][2] < landmark_list[15][2]
                and landmark_list[20][2] < landmark_list[19][2]):
            return "pause_play"

    def playVoice(self, fileName, mode):
        playsound(fileName)

    # 发送到home asssisant 的指令
    def sendRemoteCommand(self, mqtt_client):
        cTime = time.time()
        if cTime - self.command_start_time > self.command_interval:

            msg = f"hello"
            result = mqtt_client.publish(self.mqtt_topic, msg)
            status = result[0]
            if status == 0:
                t = threading.Thread(target=self.playVoice,
                                     args=("./hand_succ.wav", 'play_pause'))
                t.start()
                print('发送指令: ' + str(self.command_index))
                self.command_index += 1
                self.command_start_time = cTime
            else:
                print(f"Failed to send message to topic ")

        else:
            print("指令频繁")
            t = threading.Thread(target=self.playVoice,
                                 args=("./hand_error.wav", 'error'))
            t.start()

    # 描线，jetson 上默认方法不支持，参考代码：https://github.com/Kazuhito00/mediapipe-python-sample
    def draw_landmarks(self, image, cx, cy, landmarks, handedness):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # キーポイント
        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append((landmark_x, landmark_y))

            if index == 0:  # 手首1
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 1:  # 手首2
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 2:  # 親指：付け根
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 3:  # 親指：第1関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 4:  # 親指：指先
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 5:  # 人差指：付け根
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 6:  # 人差指：第2関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 7:  # 人差指：第1関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 8:  # 人差指：指先
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 9:  # 中指：付け根
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 10:  # 中指：第2関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 11:  # 中指：第1関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 12:  # 中指：指先
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 13:  # 薬指：付け根
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 14:  # 薬指：第2関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 15:  # 薬指：第1関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 16:  # 薬指：指先
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 17:  # 小指：付け根
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 18:  # 小指：第2関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 19:  # 小指：第1関節
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 20:  # 小指：指先
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

        # 接続線
        if len(landmark_point) > 0:
            # 親指
            cv2.line(image, landmark_point[2], landmark_point[3], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[3], landmark_point[4], (0, 255, 0),
                     2)

            # 人差指
            cv2.line(image, landmark_point[5], landmark_point[6], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[6], landmark_point[7], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[7], landmark_point[8], (0, 255, 0),
                     2)

            # 中指
            cv2.line(image, landmark_point[9], landmark_point[10], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[10], landmark_point[11],
                     (0, 255, 0), 2)
            cv2.line(image, landmark_point[11], landmark_point[12],
                     (0, 255, 0), 2)

            # 薬指
            cv2.line(image, landmark_point[13], landmark_point[14],
                     (0, 255, 0), 2)
            cv2.line(image, landmark_point[14], landmark_point[15],
                     (0, 255, 0), 2)
            cv2.line(image, landmark_point[15], landmark_point[16],
                     (0, 255, 0), 2)

            # 小指
            cv2.line(image, landmark_point[17], landmark_point[18],
                     (0, 255, 0), 2)
            cv2.line(image, landmark_point[18], landmark_point[19],
                     (0, 255, 0), 2)
            cv2.line(image, landmark_point[19], landmark_point[20],
                     (0, 255, 0), 2)

            # 手の平
            cv2.line(image, landmark_point[0], landmark_point[1], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[1], landmark_point[2], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[2], landmark_point[5], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[5], landmark_point[9], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[9], landmark_point[13], (0, 255, 0),
                     2)
            cv2.line(image, landmark_point[13], landmark_point[17],
                     (0, 255, 0), 2)
            cv2.line(image, landmark_point[17], landmark_point[0], (0, 255, 0),
                     2)

        # 重心 + 左右
        if len(landmark_point) > 0:
            # handedness.classification[0].index
            # handedness.classification[0].score

            cv2.circle(image, (cx, cy), 12, (0, 255, 0), 2)
            cv2.putText(image, handedness.classification[0].label[0],
                        (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)  # label[0]:一文字目だけ

        return image

    def calc_palm_moment(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        palm_array = np.empty((0, 2), int)

        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            if index == 0:  # 手首1
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:  # 手首2
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:  # 人差指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 9:  # 中指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 13:  # 薬指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 17:  # 小指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
        M = cv2.moments(palm_array)
        cx, cy = 0, 0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        return cx, cy

    # 处理视频
    def processVideo(self):
        mqtt_client = self.connect_mqtt()
        mqtt_client.loop_start()

        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:

            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("请检查摄像头")
                    continue

                # 镜像，转为RGB
                # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 提高性能
                image.flags.writeable = False

                # 传给模型分析
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 判断是否有手掌
                if results.multi_hand_landmarks:

                    # 解析每个手掌
                    for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness):

                        # 解析手指，存入关节坐标
                        landmark_list = []
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])

                        # 为每个手掌画线，这个是默认方法，可能再JETSON中调用不聊
                        # self.mp_drawing.draw_landmarks(
                        #     image,
                        #     hand_landmarks,
                        #     self.mp_hands.HAND_CONNECTIONS,
                        #     self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        #     self.mp_drawing_styles.get_default_hand_connections_style())

                        cx, cy = self.calc_palm_moment(image, hand_landmarks)
                        image = self.draw_landmarks(image, cx, cy,
                                                    hand_landmarks, handedness)

                        cTime = time.time()

                        if (self.checkHandAction(landmark_list) == "pause_play"
                            ):

                            cv2.putText(image, "Action: Pause", (10, 170),
                                        cv2.FONT_HERSHEY_PLAIN, 3,
                                        (255, 0, 255), 3)

                            if ((cTime - self.action_start_time >
                                 self.action_interval)
                                    and (self.command_sent == 0)):

                                # 发送指令
                                self.sendRemoteCommand(mqtt_client)
                                # 设置已发送
                                self.command_sent = 1

                            # 无需处理下一双手
                            break

                        else:
                            # 重置
                            self.command_sent = 0
                            self.action_start_time = cTime

                # 计算FPS
                cTime = time.time()
                fps_text = 1 / (cTime - self.fps_time)
                self.fps_time = cTime
                # 放在视频上
                cv2.putText(image, "FPS: " + str(int(fps_text)), (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                # 展示手掌数量
                if results.multi_hand_landmarks:
                    cv2.putText(
                        image,
                        "Hands Num: " + str(len(results.multi_hand_landmarks)),
                        (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                # 缩放显示
                # out.write(image)
                image = cv2.resize(image, (480, 270))
                cv2.imshow('MediaPipe Hands', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        self.cap.release()
        # out.release()

        cv2.destroyAllWindows()


remote = handPlay()
remote.processVideo()
