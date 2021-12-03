
import sys
import os
import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import dgl
from dgl.nn import GraphConv
import mediapipe as mp
import glob
import math


sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d
from utils.vis import vis_keypoints, vis_3d_keypoints


# 图卷积神经网络模型
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


# 输入一个手部图片，返回3D坐标
class HandPose:
    def __init__(self):
        
        cfg.set_args('0')
        cudnn.benchmark = True
        # joint set information is in annotations/skeleton.txt
        self.joint_num = 21 # single hand
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}

        # snapshot load
        model_path = './snapshot_20.pth.tar'
        assert osp.exists(model_path), 'Cannot find self.hand_pose_model at ' + model_path
        print('Load checkpoint from {}'.format(model_path))
        self.hand_pose_model = get_model('test', self.joint_num)
        self.hand_pose_model = DataParallel(self.hand_pose_model).cuda()
        ckpt = torch.load(model_path)
        self.hand_pose_model.load_state_dict(ckpt['network'], strict=False)
        self.hand_pose_model.eval()


        # prepare input image
        self.transform = transforms.ToTensor()

    def get3Dpoint(self,x_t_l, y_t_l, cam_w, cam_h,original_img):
        bbox = [x_t_l, y_t_l, cam_w, cam_h] # xmin, ymin, width, height

        original_img_height, original_img_width = original_img.shape[:2]
        bbox = process_bbox(bbox, (original_img_height, original_img_width, original_img_height))
        img, trans, inv_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, cfg.input_img_shape)
        img = self.transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        
        # forward
        inputs = {'img': img}
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = self.hand_pose_model(inputs, targets, meta_info, 'test')
        img = img[0].cpu().numpy().transpose(1,2,0) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
        joint_coord = out['joint_coord'][0].cpu().numpy() # x,y pixel, z root-relative discretized depth
        
        
        rel_root_depth = out['rel_root_depth'][0].cpu().numpy() # discretized depth
        hand_type = out['hand_type'][0].cpu().numpy() # handedness probability

        # restore joint coord to original image space and continuous depth space
        joint_coord[:,0] = joint_coord[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        joint_coord[:,1] = joint_coord[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        joint_coord[:,:2] = np.dot(inv_trans, np.concatenate((joint_coord[:,:2], np.ones_like(joint_coord[:,:1])),1).transpose(1,0)).transpose(1,0)
        joint_coord[:,2] = (joint_coord[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

        # restore right hand-relative left hand depth to continuous depth space
        rel_root_depth = (rel_root_depth/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

        # right hand root depth == 0, left hand root depth == rel_root_depth
        joint_coord[self.joint_type['left'],2] += rel_root_depth
      
        # 3D节点信息
        return joint_coord
        


# 动作识别类
class HandRecognize:
    def __init__(self):
        self.modelGCN = GCN(3, 16, 6)
        self.modelGCN.load_state_dict(torch.load('./saveModel/handsModel.pth'))
        self.modelGCN.eval()
        self.handPose = HandPose()
        self.mp_hands = mp.solutions.hands

        
        # 中指与矩形左上角点的距离
        self.L1 = 0
        self.L2 = 0

        # image实例，以便另一个类调用
        self.image=None

        self.overlay_list = self.init_overlay_list()
        self.overlay_list_last_type = 0


    # 初始化，获取动作对应图片
    def init_overlay_list(self):
        overlay_list = []
        img_list = glob.glob('./actionImage/*')
        for img_file in img_list:
            overlay = cv2.imread(img_file,cv2.COLOR_RGB2BGR)
            overlay = cv2.resize(overlay,(0,0), fx=0.5, fy=0.5)
            overlay_list.append(overlay)
            
        return  overlay_list   
    # 计算相对坐标
    def relativeMiddleCor(self,x_list, y_list,z_list):
        # 计算相对于几何中心的坐标

        # 计算几何中心坐标
        min_x = min(x_list)
        max_x = max(x_list)

        min_y = min(y_list)
        max_y = max(y_list)

        min_z = min(z_list)
        max_z = max(z_list)

        middle_p_x = min_x+ 0.5*(max_x-min_x)
        middle_p_y = min_y+ 0.5*(max_y-min_y)
        middle_p_z = min_z+ 0.5*(max_z-min_z)

        # p(相对) = (x原始 -  Px(重心), y原始 -  Py(重心))
        x_list = np.array(x_list) - middle_p_x
        y_list = np.array(y_list) - middle_p_y
        z_list = np.array(z_list) - middle_p_z

        x_y_z_column = np.column_stack((x_list, y_list,z_list))

        return x_y_z_column
    # 预测动作
    def predictAction(self,joint_coord):
        # 验证模式
        x_list = joint_coord[:,0].tolist()
        y_list = joint_coord[:,1].tolist()
        z_list = joint_coord[:,2].tolist()

        # 构造图以及特征
        u,v = torch.tensor([[0,0,0,0,0,4,3,2,8,7,6,12,11,10,16,15,14,20,19,18,0,21,21,21,21,21,25,24,23,29,28,27,33,32,31,37,36,35,41,40,39],
            [4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17,21,25,29,33,37,41,24,23,22,28,27,26,32,31,30,36,35,34,40,39,38]])
        g = dgl.graph((u,v))
        
        # 无向处理
        bg = dgl.to_bidirected(g)
        
        x_y_z_column = self.relativeMiddleCor(x_list, y_list,z_list)
        # 添加特征
        bg.ndata['feat'] =torch.tensor( x_y_z_column ) # x,y,z坐标

        # 测试模型
            
        device = torch.device("cuda:0")
        bg = bg.to(device)
        self.modelGCN = self.modelGCN.to(device)
        pred = self.modelGCN(bg, bg.ndata['feat'].float())
        pred_type =pred.argmax(1).item()

        return pred_type
    
    # 采集训练数据
    def getTrainningData(self,task_type = '-1',type_num = 100):

        start_time=time.time()
        # 从摄像头采集：
        cap = cv2.VideoCapture(0)
        # 计算刷新率
        fpsTime = time.time()

        while cap.isOpened():

            success,original_img = cap.read()
            original_img  = cv2.flip(original_img, 1)
            if not success:
                print("空帧.")
                continue
            
            # prepare bbox
            x_t_l = 200
            y_t_l = 150
            cam_w = 300
            cam_h = 300
            joint_coord = self.handPose.get3Dpoint(x_t_l, y_t_l, cam_w, cam_h,original_img)


            
            duration = time.time() -start_time
            cv2.imshow('data',original_img)
            # 存储训练数据
            if task_type != '-1':
                if  duration < 30:
                    print('等等')
                    continue
                
                action_dir = './trainingData/'+task_type
                if not os.path.exists(action_dir):
                    os.makedirs(action_dir)
                    # 文件夹不存在的话创建文件夹

                path, dirs, files = next(os.walk(action_dir))
                file_count = len(files)
                # 判断数据采集是否达标
                if file_count > int(type_num):

                    print('采集完毕')
                    break
                # Data to be written
                dictionary ={
                    "action_type" : task_type,
                    "x_list" : joint_coord[:,0].tolist(),
                    "y_list" : joint_coord[:,1].tolist(),
                    "z_list" : joint_coord[:,2].tolist()
                }
                # Serializing json 
                json_object = json.dumps(dictionary, indent = 4)
                
                json_fileName = action_dir +'./'+task_type+'-'+str(time.time()) +'.json'
                # Writing to .json
                with open(json_fileName, "w") as outfile:
                    outfile.write(json_object)
                    print(str(file_count)+'-采集并写入:'+json_fileName )
                # 文件名：action_type + time.time()
            

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
    

    # 主函数
    def recognize(self):
        # 计算刷新率
        fpsTime = time.time()
        
        # OpenCV读取视频流
        cap = cv2.VideoCapture(0)
        # 视频分辨率
        resize_w = 960
        resize_h = 720
        fps = cap.get(cv2.CAP_PROP_FPS)
        videoWriter = cv2.VideoWriter('./video/oto_other.mp4', cv2.VideoWriter_fourcc(*'H264'), 10, (resize_w,resize_h))

        # load the overlay image. size should be smaller than video frame size
        overlay = cv2.imread('./actionImage/text_0.png',cv2.COLOR_RGB2BGR)
        overlay = cv2.resize(overlay,(0,0), fx=0.5, fy=0.5)
        overlay_rows,overlay_cols,channels = overlay.shape


        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            while cap.isOpened():

                # 初始化矩形
                success, self.image = cap.read()
                self.image = cv2.resize(self.image, (resize_w, resize_h))

                if not success:
                    print("空帧.")
                    continue
                

                # 提高性能
                self.image.flags.writeable = False
                # 转为RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                # 镜像
                self.image = cv2.flip(self.image, 1)
                # mediapipe模型处理
                results = hands.process(self.image)

                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                # 判断是否有手掌
                if results.multi_hand_landmarks:
                    # 遍历每个手掌

                    # 用来存储手掌范围的矩形坐标
                    paw_x_list = []
                    paw_y_list = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        # 在画面标注手指
                        # self.mp_drawing.draw_landmarks(
                        #     self.image,
                        #     hand_landmarks,
                        #     self.mp_hands.HAND_CONNECTIONS,
                        #     self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        #     self.mp_drawing_styles.get_default_hand_connections_style())


                        # 解析手指，存入各个手指坐标
                        landmark_list = []

                        
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)

                    if len(paw_x_list) > 0:

                        # 比例缩放到像素
                        ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
                        ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)
                        
                        # 设计手掌左上角、右下角坐标
                        paw_left_top_x,paw_right_bottom_x = map(ratio_x_to_pixel,[min(paw_x_list),max(paw_x_list)])
                        paw_left_top_y,paw_right_bottom_y = map(ratio_y_to_pixel,[min(paw_y_list),max(paw_y_list)])

                        # 计算模型
                        # prepare bbox
                        x_t_l = paw_left_top_x-100
                        y_t_l = paw_left_top_y-100
                        cam_w = (paw_right_bottom_x-paw_left_top_x)+200
                        cam_h =  (paw_right_bottom_y -paw_left_top_y )+200

                        # cv2.rectangle(self.image, (x_t_l, y_t_l), ((x_t_l+cam_w), (y_t_l+cam_h)), (255, 0, 255), 2)

                        joint_coord = self.handPose.get3Dpoint(x_t_l, y_t_l, cam_w, cam_h,self.image)

                        pred_type = self.predictAction(joint_coord)
                        print("action: " + str(pred_type))

                        # 给手掌画框框
                        cv2.rectangle(self.image,(paw_left_top_x-50,paw_left_top_y-50),(paw_right_bottom_x+50,paw_right_bottom_y+50),(0, 255,0),2)
                        

                        # 模型计算后的动作
                        action_type = int(pred_type)


                        overlay = self.overlay_list[action_type]
                        overlay_rows,overlay_cols,channels = overlay.shape
                        action_text_lx = paw_left_top_x-overlay_cols
                        action_text_ly = paw_left_top_y-overlay_rows

                        self.overlay_list_last_type = action_type

                        
                        if (action_text_ly )> 0 and (action_text_lx > 0):
                            
                            
                            overlay_copy=cv2.addWeighted(self.image[action_text_ly:paw_left_top_y, action_text_lx:paw_left_top_x ],1,overlay,20,0)

                            self.image[action_text_ly:paw_left_top_y, action_text_lx:paw_left_top_x ] = overlay_copy

             
                # 显示刷新率FPS
                cTime = time.time()
                fps_text = 1/(cTime-fpsTime)
                fpsTime = cTime
                cv2.putText(self.image, "FPS: " + str(int(fps_text)), (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                cv2.putText(self.image, "Action: "+str(self.overlay_list_last_type) , (10, 120),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                            
                # 显示画面
                # self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
                cv2.imshow('Enpei test', self.image)
                videoWriter.write(self.image) 
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()
            videoWriter.release()


handRecognize  = HandRecognize()
handRecognize.recognize()