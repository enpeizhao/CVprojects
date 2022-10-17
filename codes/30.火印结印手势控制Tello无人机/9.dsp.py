"""
@author: enpei

tello无人机视觉控制

"""

import ctypes
import cv2
import numpy as np
import time

# 多进程
from multiprocessing import Process, Value
# tello
from djitellopy import Tello
# PID
from PID import simple_PID
# mediapipe
from KEYPOINTS import Mpkeypoints

import torch

import math

# 实例化关键点提取
mp_keypoints = Mpkeypoints()


class Ai_tello:
    def __init__(self):
        # tello初始化
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()

        # ************************************ 绘制 相关 *********************************
        self.png_dict = {}
        # 获取
        self.getPngList()


        # ************************************ 结印动作相关变量*********************************

        # 加载 yolov5模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./weights/pose.pt',source='local') 
        # 置信度阈值
        self.model.conf = 0.5

        # 起飞指令激活时间
        self.take_off_time = None
        # 结印动作顺序
        self.yolo_action_seq = ['ani_1','ani_2','ani_3','ani_4','ani_5','ani_6']
        # 状态机，1表示当前动作已做完（击中）
        self.yolo_action_status = [0,0,0,0,0,0]


        # ************************************ 骨骼点动作相关变量*********************************

        # 当前指令
        self.this_status_id = 1
        # 帧数计数器
        self.frame_count = {
            'action_6':0,
            'action_7':0,
            'action_8':0
        }
        # 指令6的上次激活时间
        self.state_6_last_act_time = time.time()

        # ************************************ 测量误差 相关 *********************************

        self.dist_up_down_error = Value(ctypes.c_int,0)             # 镜头中心与双肩中心点上下误差距离
        self.dist_left_right_error = Value(ctypes.c_int,0)          # 镜头中心与双肩中心点左右误差距离
        self.dist_forward_backward_error = Value(ctypes.c_int,0)    # 前后误差距离


        # ************************************ PID 相关 *********************************
        
        # 设置PID参数，这些参数需要根据电脑处理速度自行调节
        self.up_down_pid_params = [0.32, 0.01, 0.03]                   # 上下调整用的PID参数
        self.yaw_pid_params = [0.12, 0, 0.05]                       # 偏航旋转调整用的PID参数
        self.forward_backward_pid_params = [0.35, 0, 0.03]       # 前后调整用的PID参数

        # 实例化PID
        self.up_down_pid = simple_PID(self.up_down_pid_params)                       # 上下方向PID
        self.yaw_rotate_pid = simple_PID(self.yaw_pid_params)                        # yaw偏航旋转角度PID
        self.forward_backward_pid = simple_PID(self.forward_backward_pid_params)     # 前后方向PID

        # PID调节量（进程间共享）
        self.pid_up_down_output = Value(ctypes.c_int,0)         # 上下
        self.pid_yaw_output = Value(ctypes.c_int,0)             # 偏航旋转
        self.pid_forward_backward_output = Value(ctypes.c_int,0)      # 前后

        # pid 曲线Y坐标列表
        self.pid_y_list = []

        # ************************************ 飞控指令 相关 *********************************
        
        # 飞机电量
        self.tello_drone_battery = Value(ctypes.c_int,0)

        self.flight_control_command = Value(ctypes.c_int,-1)      # 飞行控制指令：-1 在地面

        # 无人机与人前后保持距离，当前希望200CM
        self.drone_forward_target_distance = Value(ctypes.c_int,200)

        # 飞机信息查询间隔
        self.flight_infor_retrieve_interval = 1


        

    def updatePid(self):
        '''
        PID更新
        '''
        # 前后调整值
        self.pid_forward_backward_output.value = - int(self.forward_backward_pid.update(self.dist_forward_backward_error.value,-50,50))
        # 上下调整值
        self.pid_up_down_output.value = - int(self.up_down_pid.update(self.dist_up_down_error.value,-100,100))
        # yaw旋转调整值
        self.pid_yaw_output.value = - int(self.yaw_rotate_pid.update(self.dist_left_right_error.value,-100,100))
        

        return True

    def pidDrawLine(self,bg,which_error='up_down'):
        '''
        绘制曲线
        '''
        errorVal = None
        if which_error == 'up_down':
            errorVal = self.dist_up_down_error.value
        elif which_error == 'yaw_rotate':
            errorVal = self.dist_left_right_error.value
        else:
            errorVal = self.dist_forward_backward_error.value

        # 绘制中心线
        line1_left_p = (0,400)
        line1_right_p = (800,400)

        line2_left_p = (400,0)
        line2_right_p = (400,800)

        cv2.line(bg,line1_left_p,line1_right_p,(0,255,0),2)
        cv2.line(bg,line2_left_p,line2_right_p,(0,255,0),2)

        # 连续线的纵轴坐标
        self.pid_y_list.append(errorVal)

        # 截断数组，使用后100个坐标
        self.pid_y_list = self.pid_y_list[-200:]

        all_points = []
        # 添加X坐标
        for i,y in enumerate( self.pid_y_list[::-1] ):
            # 第一个点坐标为0，第二个是-4，然后是-8等，直到-400
            x = 400 + (-2 * i)
            y = 400 + (y)

            p = (x,y)

            # 第一个点画出来
            if x == 400:
                cv2.circle(bg,(x,y),10,(255,0,255),-1)
                cv2.putText(bg, str(y-400), (x+20, y), cv2.FONT_ITALIC, 1, (255, 0, 255), 2)
            
            # 点坐标
            all_points.append(p)
        
        pts = np.asarray(all_points,np.int32)
        
        pts = pts.reshape((-1, 1, 2)) 
        
        # 画连续线
        cv2.polylines(bg, [pts], False, (255,0,255), 2) 

        return True

    def flightControl(self):
        '''
        飞行控制进程调用函数
        '''

        # 启动另一个进程
        Process(target=ai_tello.cameraProcess).start()

        # 初始化当前OBJ
        this_obj = self.up_down_pid
        this_obj_name = 'up_down'
        display_pid_paras_list = self.up_down_pid_params

        retrieve_time = time.time()

        # 飞行时间
        tello_takeoff_time = None

        

        while True:

            # 查询电量
            now = time.time()
            if now - retrieve_time > self.flight_infor_retrieve_interval:
                self.tello_drone_battery.value = self.tello.get_battery()
                retrieve_time = now

            # 更新PID
            self.updatePid()

            # 调节参数的背景图片，调节PID参数时需要鼠标在窗口上
            
            bg = np.zeros((800,800,3))

            # 绘制左上角文字
            cv2.putText(bg, 'PID: {}'.format(this_obj_name), (30, 100), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

            cv2.putText(bg, 'Value: {} {} {}'.format(display_pid_paras_list[0], display_pid_paras_list[1], display_pid_paras_list[2]), (30, 150), cv2.FONT_ITALIC, 1, (255, 0, 255), 2)

            # 绘制PID动态曲线
            self.pidDrawLine(bg,this_obj_name)

           
            # 调节参数用的背景图
            cv2.imshow("Control panel", bg)
            
            # 暂停1ms
            key = cv2.waitKey(1) & 0xff


            # PID调试界面

            # 切换PID实例
            if key == ord('z'):
                this_obj = self.forward_backward_pid
                this_obj_name  = 'forward_backward'
                display_pid_paras_list = self.forward_backward_pid_params
            if key == ord('x'):
                this_obj = self.up_down_pid
                this_obj_name  = 'up_down'
                display_pid_paras_list = self.up_down_pid_params
            if key == ord('c'):
                this_obj = self.yaw_rotate_pid
                this_obj_name  = 'yaw_rotate'
                display_pid_paras_list = self.yaw_pid_params

            # 微调三个值
            if key == ord('q'):
                # 增加p
                display_pid_paras_list = this_obj.setParas('p', 0.01)

            elif key == ord('a'):
                # 减少p
                display_pid_paras_list = this_obj.setParas('p', -0.01)

            elif key == ord('w'):
                # 增加i
                display_pid_paras_list = this_obj.setParas('i', 0.01)

            elif key == ord('s'):
                # 减少i
                display_pid_paras_list = this_obj.setParas('i', -0.01)

            elif key == ord('e'):
                # 增加d
                display_pid_paras_list = this_obj.setParas('d', 0.01)
            elif key == ord('d'):
                # 减少d
                display_pid_paras_list = this_obj.setParas('d', -0.01)


            


            # 飞控状态判断
            if self.flight_control_command.value > -1:

                if self.flight_control_command.value == 0:
                    # 起飞指令
                    print('take off')
                    self.flight_control_command.value = 1
                    # 计时
                    tello_takeoff_time = time.time()

                    self.tello.takeoff()
                    self.tello.move_up(50)
                    
                else:
                    # 起飞，起飞后要等待10S再执行其他命令
                    if time.time() - tello_takeoff_time > 10:
                            
                        # 对应动作（1-8）
                        action_id = self.flight_control_command.value
                        
                        if action_id == 1:
                            # 高度、yaw角度锁定
                            self.tello.send_rc_control(0, 0, self.pid_up_down_output.value, self.pid_yaw_output.value)
                            
                            # pass

                        if action_id == 2:
                            # 高度、yaw角度锁定，向前移动
                            self.tello.send_rc_control(0, 20, 0, self.pid_yaw_output.value)

                        if action_id == 3:
                            # 高度、yaw角度锁定，向后移动
                            self.tello.send_rc_control(0, -20, 0, self.pid_yaw_output.value)

                        if action_id == 4:
                            # 高度、yaw角度锁定，向左移动
                            self.tello.send_rc_control(-20, 0, 0, self.pid_yaw_output.value)

                        if action_id == 5:
                            # 高度、yaw角度锁定，向右移动
                            self.tello.send_rc_control(20, 0, 0, self.pid_yaw_output.value)
                        if action_id == 6:
                            # 高度、yaw角度锁定，前后移动
                            self.tello.send_rc_control(0, self.pid_forward_backward_output.value, self.pid_up_down_output.value, self.pid_yaw_output.value)

                        if action_id == 7:
                            # 设置目标距离为100cm处降落
                            self.drone_forward_target_distance.value = 100

                            # 判断距离是否达到
                            if self.dist_forward_backward_error.value >= 0:
                                # 允许降落
                                time.sleep(3)
                                self.tello.land() 
                                self.flight_control_command.value =-1
                            else:
                                # 继续调整    
                                # 高度、yaw角度锁定，前后移动
                                self.tello.send_rc_control(0, self.pid_forward_backward_output.value, self.pid_up_down_output.value, self.pid_yaw_output.value)


                        if action_id == 8:   
                            # 原地降落
                            self.tello.land()
                            self.flight_control_command.value =-1
    
    def getPngList(self):
        '''
        读取PNG图片，追加进png_dict
        '''
        palm_action = {'ani_1':'巳','ani_2':'未','ani_3':'申','ani_4':'亥','ani_5':'午','ani_6':'寅'}
        char_list = ['电量','状态','上下差','偏航差','前后差',]
        state_list ={-1:'准备中',0:'起飞',1:'高度偏航锁定',2:'向前',3:'向后',4:'向左',5:'向右',6:'追踪',7:'手掌降落',8:'立即降落'}

        for name in palm_action.values():
            filename = './png_label/'+name+'.png'
            png_img = self.readPngFile(filename,0.9)
            self.png_dict[name]= png_img

        for name in char_list:
            filename = './png_label/'+name+'.png'
            png_img = self.readPngFile(filename)
            self.png_dict[name]= png_img

        for name in state_list.values():
            filename = './png_label/'+name+'.png'
            png_img = self.readPngFile(filename)
            self.png_dict[name]= png_img

        print('PNG文字标签加载完毕')
    
    def readPngFile(self,fileName,scale=0.5):
        '''
        读取PNG图片
        '''
        # 解决中文路径问题
        png_img = cv2.imdecode(np.fromfile(fileName,dtype=np.uint8),-1)
        # 转为BGR，变成3通道
        png_img = cv2.cvtColor(png_img,cv2.COLOR_RGB2BGR)
        png_img = cv2.resize(png_img,(0,0), fx=scale, fy=scale)
        return png_img   

    def addOverylay(self,frame,overlay,l,t):

        '''
        添加标签png覆盖
        '''
        # 解决l、t超界
        l = max(l,0)
        t = max(t,0)
        # 覆盖显示
        overlay_h,overlay_w = overlay.shape[:2]
        # 覆盖范围
        overlay_l,overlay_t = l,t
        overlay_r,overlay_b = (l + overlay_w),(overlay_t+overlay_h)
        # 遮罩
        overlay_copy=cv2.addWeighted(frame[overlay_t:overlay_b, overlay_l:overlay_r ],1,overlay,20,0)
        frame[overlay_t:overlay_b, overlay_l:overlay_r ] =  overlay_copy


    def drawInfo(self, frame, duration,):
        '''
        绘制左上角信息，被cameraProcess进程调用
        '''

        # FPS信息
        fps = round(1 / duration, 1)
        cv2.putText(frame, 'FPS: {}'.format(fps), (50, 80),cv2.FONT_ITALIC, 1.5, (0, 255, 0), 2)
        
        # 电量
        overlay = self.png_dict['电量']
        self.addOverylay(frame,overlay,50,100)
        cv2.putText(frame, '{}%'.format(self.tello_drone_battery.value), (150, 142), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 2)

        # 状态
        state_list ={-1:'准备中',0:'起飞',1:'高度偏航锁定',2:'向前',3:'向后',4:'向左',5:'向右',6:'追踪',7:'手掌降落',8:'立即降落'}

        state_char =state_list[self.flight_control_command.value]

        overlay = self.png_dict['状态']
        self.addOverylay(frame,overlay,50,160)
        
        overlay = self.png_dict[state_char]
        self.addOverylay(frame,overlay,150,160)

        # 上下差
        overlay = self.png_dict['上下差']
        self.addOverylay(frame,overlay,50,220)
        cv2.putText(frame, '{}'.format(self.dist_up_down_error.value), (190, 260), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 2)

        # 偏航差
        overlay = self.png_dict['偏航差']
        self.addOverylay(frame,overlay,50,280)
        cv2.putText(frame, '{}'.format(self.dist_left_right_error.value), (190, 320), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 2)

        # 前后差
        overlay = self.png_dict['前后差']
        self.addOverylay(frame,overlay,50,340)
        cv2.putText(frame, '{}'.format(self.dist_forward_backward_error.value), (190, 380), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 2)
       


    def cameraProcess(self):
        '''
        视频流处理：动作识别、绘制等
        '''
        
        # cap = cv2.VideoCapture(0)
        # 动作
        palm_action = {'ani_1':'巳','ani_2':'未','ani_3':'申','ani_4':'亥','ani_5':'午','ani_6':'寅'}
    

        while True:
            start_time = time.time()

            # 读取视频帧
            # ret,frame = cap.read()
            frame = self.tello.get_frame_read().frame

            frame = cv2.flip(frame, 1)

            # 计算视频高度和宽度
            frame_h, frame_w = frame.shape[:2]

            # 计算画面中心点坐标
            center_x = frame_w // 2
            center_y = frame_h // 2

            # 转为RGB
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 已经起飞
            if self.take_off_time != None:
                

                if time.time() -  self.take_off_time > 2:
                    # 关键点检测
                    pose_landmarks = mp_keypoints.getFramePose(img_cvt)

                    # 判断画面中是否有关键点：
                    if  pose_landmarks:

                        # 解析关键点，并且恢复为像素值
                        p_list = [[int(landmark.x * frame_w),int(landmark.y * frame_h )] for landmark in pose_landmarks.landmark]
                        # 转为numpy，才能广播计算
                        p_list = np.asarray(p_list)

                        # 原图上绘制关节点及、连线
                        self.processKeypoints(frame,p_list)
                        self.action2command(p_list)

                        # 选择双肩中心点坐标
                        shoulder_center_x = int(p_list[12][0] + int(p_list[11][0] - p_list[12][0]) / 2)
                        shoulder_center_y = int(p_list[12][1] + int(p_list[11][1] - p_list[12][1]) / 2)
                        # 绘制中心点
                        cv2.circle(frame, (shoulder_center_x,shoulder_center_y), 10, (0, 0, 255), -1)

                        # 绘制与画面中心点的连线
                        cv2.line(frame, (shoulder_center_x, shoulder_center_y),(center_x, center_y), (255, 0, 255), 2)

                        # 计算画面中心与鼻尖距离误差
                        self.dist_left_right_error.value = shoulder_center_x - center_x
                        self.dist_up_down_error.value = shoulder_center_y - center_y

                        # 计算双肩距离
                        shoulder_distance = int(math.sqrt((p_list[11][0]-p_list[12][0])**2 + (p_list[11][1]-p_list[12][1])**2))
                        
                        # 防止为0
                        shoulder_distance = 1 if shoulder_distance == 0 else shoulder_distance

                        # 绘制像素距离
                        cv2.putText(frame,'L:{}'.format(shoulder_distance),(p_list[11][0]+20,p_list[11][1]),cv2.FONT_ITALIC,1.5,(0,255,0),2 )

                        # 估算距离人的距离 (如在150cm处双肩距离为155个像素)
                        forawrd_distance =  int( (155 / shoulder_distance) * 150  )
                        # 距离误差
                        self.dist_forward_backward_error.value = self.drone_forward_target_distance.value - forawrd_distance
            else:
                # 未起飞

                # 目标检测推理
                results = self.model(img_cvt)
                results_arr = results.pandas().xyxy[0].to_numpy()

                # 解析目标检测结果
                for item in results_arr:

                    # 标签ID
                    ret_label_id = item[-2]
                    # 标签名称
                    ret_label_text = item[-1]
                    # 置信度
                    ret_conf = item[-3]
                
                    # 只绘制人
                    if ret_label_text =='person':
                        l,t,r,b = item[:4].astype('int')
                        cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
                        cv2.putText(frame,'{}%'.format(round(ret_conf*100,2)),(l,t-10),cv2.FONT_ITALIC,1.5,(0,255,0),2)

                    
                    # 结印动作，且置信度要求高一些
                    if 'ani_' in  ret_label_text and ret_conf >= 0.7:

                        l,t,r,b = item[:4].astype('int')
                        # 绘制
                        cv2.rectangle(frame,(l,t),(r,b),(0,255,20),2)
                        # 绘制动作中文png                        
                        label_zh = palm_action[ret_label_text]
                        overlay = self.png_dict[label_zh]
                        self.addOverylay(frame,overlay,l,t-100)
                        cv2.putText(frame,'{}%'.format(round(ret_conf*100,2)),(l+80,t-20),cv2.FONT_ITALIC,1.5,(255,0,255),2)


                        # 状态机列表中第一个0的索引
                        first_0_index = next(i for i,x in enumerate(self.yolo_action_status) if x == 0 )
                        # 对应动作名
                        check_action_name = self.yolo_action_seq[first_0_index]

                        # 动作匹配
                        if ret_label_text == check_action_name:
                            # 赋值1
                            self.yolo_action_status[first_0_index] = 1
                            # 检查是否完毕
                            if self.yolo_action_status == [1,1,1,1,1,1]:
                                # 发送起飞指令，设为此时
                                self.take_off_time = time.time()
                                print('动作全部匹配完毕，发送起飞指令')
                                # 允许起飞
                                self.flight_control_command.value = 0

                            else:
                                print('击中一个动作，当前列表为'+str(self.yolo_action_status))
                        else:
                            print('未击中动作，当前列表为'+str(self.yolo_action_status))
                
                    
            # 绘制左上角信息
            duration = time.time() - start_time
            # 绘制信息
            self.drawInfo(frame, duration)

            cv2.imshow('demo', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()


    def processKeypoints(self,frame,p_list):
        '''
        关键点画线
        识别出对应动作
        '''
        # 关键点连接关系
        connectoions = [
            [0,1],
            [1,2],
            [2,3],
            [3,7],
            [0,4],
            [4,5],
            [5,6],
            [6,8],
            [9,10],
            [11,12],
            [12,14],
            [14,16],
            [16,18],
            [16,20],
            [16,22],
            [11,13],
            [13,15],
            [15,17],
            [15,19],
            [15,21],
            [12,24],
            [24,26],
            [26,28],
            [28,30],
            [28,32],
            [11,23],
            [23,25],
            [25,27],
            [27,29],
            [27,31],
            [23,24],
        ]
        # 画线        
        for conn in connectoions:
            cv2.line(frame,p_list[conn[0]],p_list[conn[1]],(0,255,0),2)

        # 画点
        for p_i,(x,y) in enumerate(p_list):
            # 鼻尖中心点
            cv2.circle(frame,(x,y),5,(0,255,0),-1)
        
        
        


        return None

    def action2command(self,p_list):
        '''
        静态动作识别，以及状态修正
        '''
        

        # 本帧静态动作识别
        frame_action = 1

        # 双手上举(超过鼻尖)
        if (p_list[15][1] < p_list[11][1]) and (p_list[16][1]<p_list[12][1]):
            frame_action = 6
        # 双手交叉

        elif (p_list[15][0] < p_list[16][0]):

            if (p_list[15][1] < p_list[13][1]) and (p_list[16][1] < p_list[14][1]) :
                # 双手向上交叉
                frame_action = 7
            elif (p_list[15][1] > p_list[13][1]) and (p_list[16][1] > p_list[14][1]) :
                # 双手向下交叉
                frame_action = 8
        else:

            # 右手-内收
            if (p_list[15][0] < p_list[13][0]) and (p_list[15][1] < p_list[13][1]):
                frame_action = 2

            # 右手-外放
            if (p_list[15][0] > p_list[13][0]) and (p_list[15][1] < p_list[13][1]):
                frame_action = 3

            # 左手-内收
            if (p_list[16][0] > p_list[14][0]) and (p_list[16][1] < p_list[14][1]):
                frame_action = 4

            # 左手-外放
            if (p_list[16][0] < p_list[14][0]) and (p_list[16][1] < p_list[14][1]):
                frame_action = 5

        # print( str(frame_action) + ' {}'.format(time.time()))


        # 指令处理

        # 不在指令7、8
        if (self.this_status_id != 7) and (self.this_status_id != 8):
            # 指令6锁定状态
            if (self.this_status_id == 6):
                #激活时间超过3S
                if (time.time() - self.state_6_last_act_time > 3):
                    # 动作6
                    if frame_action ==6:
                        
                        # 帧数超过30帧
                        if self.frame_count['action_6'] > 30:
                            # 退出该指令
                            self.this_status_id = 1
                            # 累加清零
                            self.frame_count['action_6'] = 0
                            print('指令6退出，帧数清零，回到指令1') 
                        else:
                            # 累加增加1
                            self.frame_count['action_6'] = self.frame_count['action_6'] + 1

                            print('指令6中，即将退出，累加帧数: {}'.format(self.frame_count['action_6'])) 
                            

                    else:
                        # 累加清零
                        self.frame_count['action_6'] = 0
                        print('指令6执行中，未检测到退出动作，帧数清零') 


            else:
                
                # 不在指令6、7、8中

                # 设置当前指令1，清空之前指令状态
                self.this_status_id = 1

                # 当前静态动作是6
                if frame_action == 6:
                    # 清空7、8累加
                    self.frame_count['action_7'] = 0
                    self.frame_count['action_8'] = 0

                    
                    # 动作6累计帧超过30
                    if self.frame_count['action_6'] > 30:
                        # 当前指令6
                        self.this_status_id = 6
                        # 激活时间为当前
                        self.state_6_last_act_time = time.time()
                        # 清空6累加
                        self.frame_count['action_6'] = 0

                        print('指令6进入') 

                    else:
                        # 累加增加1
                        self.frame_count['action_6'] = self.frame_count['action_6'] + 1

                        print('指令6即将进入，累加帧数: {}'.format(self.frame_count['action_6'])) 

                # 当前静态动作7
                elif frame_action == 7:
                    # 清空6、8累加
                    self.frame_count['action_6'] = 0
                    self.frame_count['action_8'] = 0

                    # 动作7累计帧超过30
                    if self.frame_count['action_7'] > 10:
                        # 当前指令7
                        self.this_status_id = 7
                        print('指令7进入') 

                    else:
                        # 累加增加1
                        self.frame_count['action_7'] = self.frame_count['action_7'] + 1

                        print('指令7即将进入，累加帧数: {}'.format(self.frame_count['action_7'])) 
                
                # 当前静态动作8
                elif frame_action == 8:
                    # 清空6、7累加
                    self.frame_count['action_6'] = 0
                    self.frame_count['action_7'] = 0

                    # 动作8累计帧超过30
                    if self.frame_count['action_8'] > 10:
                        # 当前指令8
                        self.this_status_id = 8
                        print('指令8进入') 

                    else:
                        # 累加增加1
                        self.frame_count['action_8'] = self.frame_count['action_8'] + 1

                        print('指令8即将进入，累加帧数: {}'.format(self.frame_count['action_8'])) 
                
                else:
                    # 当前静态动作1、2、3、4、5

                    # 清空6、7、8累加
                    self.frame_count['action_6'] = 0
                    self.frame_count['action_7'] = 0
                    self.frame_count['action_8'] = 0
                    
                    # 设置当前指令为动作ID
                    self.this_status_id = frame_action
                    
                    print('指令{}进入'.format(self.this_status_id)) 

            
        else:
            # 处于指令7、8中
            print('最终指令{}，即将降落'.format(self.this_status_id)) 

        
        print('处理后指令状态：{}，时间:{}'.format(self.this_status_id,time.time()))

        
        # 发送飞行指令给飞控进程
        if self.flight_control_command.value  >0 :
            # 发送飞行指令
            self.flight_control_command.value = self.this_status_id



if __name__ == '__main__':

    # 实例化
    ai_tello = Ai_tello()
    # 开启两个进程（防止飞机运动时，画面静止）
    ai_tello.flightControl()
    