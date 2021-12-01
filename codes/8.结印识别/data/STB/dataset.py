# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, load_skeleton, process_bbox, get_aug_config, augmentation, transform_input_to_output_space, generate_patch_image, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        self.mode = mode
        self.root_path = '../data/STB/data'
        self.rootnet_output_path = '../data/STB/rootnet_output/rootnet_stb_output.json'
        self.original_img_shape = (480, 640) # height, width
        self.transform = transform
        self.joint_num = 21 # single hand
        self.joint_type = {'right': np.arange(self.joint_num,self.joint_num*2), 'left': np.arange(0,self.joint_num)}
        self.root_joint_idx = {'right': self.joint_num, 'left': 0}
        self.skeleton = load_skeleton(osp.join(self.root_path, 'skeleton.txt'), self.joint_num*2)
        
        self.datalist = [];
        self.annot_path = osp.join(self.root_path, 'STB_' + self.mode + '.json')
        db = COCO(self.annot_path)

        if self.mode == 'test' and cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            
            seq_name = img['seq_name']
            img_path = osp.join(self.root_path, 'images', seq_name, img['file_name'])
            img_width, img_height = img['width'], img['height']
            cam_param = img['cam_param']
            focal, princpt = np.array(cam_param['focal'],dtype=np.float32), np.array(cam_param['princpt'],dtype=np.float32)
            
            joint_img = np.array(ann['joint_img'],dtype=np.float32)
            joint_cam = np.array(ann['joint_cam'],dtype=np.float32)
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32)
            
            # transform single hand data to double hand data structure
            hand_type = ann['hand_type']
            joint_img_dh = np.zeros((self.joint_num*2,2),dtype=np.float32)
            joint_cam_dh = np.zeros((self.joint_num*2,3),dtype=np.float32)
            joint_valid_dh = np.zeros((self.joint_num*2),dtype=np.float32)
            joint_img_dh[self.joint_type[hand_type]] = joint_img
            joint_cam_dh[self.joint_type[hand_type]] = joint_cam
            joint_valid_dh[self.joint_type[hand_type]] = joint_valid
            joint_img = joint_img_dh; joint_cam = joint_cam_dh; joint_valid = joint_valid_dh;

            if self.mode == 'test' and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                abs_depth = rootnet_result[str(aid)]['abs_depth']
            else:
                bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = joint_cam[self.root_joint_idx[hand_type],2] # single hand abs depth
            
            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type, 'abs_depth': abs_depth}
            self.datalist.append(data)
     
    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
  
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type = data['img_path'], data['bbox'], data['joint'], data['hand_type']
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)

        # image load
        img = load_img(img_path)
        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
        img = self.transform(img.astype(np.float32))/255.
        rel_root_depth = np.zeros((1),dtype=np.float32)
        root_valid = np.zeros((1),dtype=np.float32)
        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)
        
        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'inv_trans': inv_trans, 'hand_type_valid': 1}
        return inputs, targets, meta_info

    def evaluate(self, preds):

        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = preds['joint_coord'], preds['rel_root_depth'], preds['hand_type'], preds['inv_trans']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe = [[] for _ in range(self.joint_num)] # treat right and left hand identical
        acc_hand_cls = 0
        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type = data['bbox'], data['cam_param'], data['joint'], data['hand_type']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']

            # restore coordinates to original space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num*2):
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)
            pred_joint_coord_img[:,2] = pred_joint_coord_img[:,2] + data['abs_depth']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            pred_joint_coord_cam[self.joint_type['right']] = pred_joint_coord_cam[self.joint_type['right']] - pred_joint_coord_cam[self.root_joint_idx['right'],None,:]
            pred_joint_coord_cam[self.joint_type['left']] = pred_joint_coord_cam[self.joint_type['left']] - pred_joint_coord_cam[self.root_joint_idx['left'],None,:]
            gt_joint_coord[self.joint_type['right']] = gt_joint_coord[self.joint_type['right']] - gt_joint_coord[self.root_joint_idx['right'],None,:]
            gt_joint_coord[self.joint_type['left']] = gt_joint_coord[self.joint_type['left']] - gt_joint_coord[self.root_joint_idx['left'],None,:]
            
            # select right or left hand using groundtruth hand type
            pred_joint_coord_cam = pred_joint_coord_cam[self.joint_type[gt_hand_type]]
            gt_joint_coord = gt_joint_coord[self.joint_type[gt_hand_type]]
            joint_valid = joint_valid[self.joint_type[gt_hand_type]]

            # mpjpe save
            for j in range(self.joint_num):
                if joint_valid[j]:
                    mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
            
            if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                acc_hand_cls += 1
            elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                acc_hand_cls += 1

            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_valid = joint_valid.copy()
                filename = 'out_' + str(n) + '.jpg'
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton[:self.joint_num], filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.png'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton[:self.joint_num], filename)

        print('Handedness accuracy: ' + str(acc_hand_cls / sample_num))

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num):
            mpjpe[j] = np.mean(np.stack(mpjpe[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe[j])
        print(eval_summary)
        print('MPJPE: %.2f' % (np.mean(mpjpe)))

