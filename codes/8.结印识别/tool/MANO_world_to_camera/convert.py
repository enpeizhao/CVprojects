# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import json
from glob import glob
import os.path as osp
import smplx
import torch
import cv2

def get_fitting_error(mesh, regressor, joint, joint_valid, hand_type):
    # ih26m joint coordinates from MANO mesh
    joint_from_mesh = np.dot(regressor, mesh)

    # choose one of right and left hands
    if hand_type == 'right':
        joint = joint[np.arange(0,21),:]
        joint_valid = joint_valid[np.arange(0,21),:]
    else:
        joint = joint[np.arange(21,21*2),:]
        joint_valid = joint_valid[np.arange(21,21*2),:]

    # coordinate masking for error calculation
    joint_from_mesh = joint_from_mesh[np.tile(joint_valid==1, (1,3))].reshape(-1,3)
    joint = joint[np.tile(joint_valid==1, (1,3))].reshape(-1,3)

    error = np.sqrt(np.sum((joint_from_mesh - joint)**2,1)).mean()
    return error

# mano layer
smplx_path = 'SMPLX_PATH'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
joint_regressor = np.load('J_regressor_mano_ih26m.npy')
root_joint_idx = 20

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1
            
root_path = '../../data/InterHand2.6M/'
img_root_path = osp.join(root_path, 'images')
annot_root_path = osp.join(root_path, 'annotations')
subset = 'all'
split = 'train'
capture_idx = '13'
seq_name = '0266_dh_pray'
cam_idx = '400030'

with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_MANO_NeuralAnnot.json')) as f:
    mano_params = json.load(f)
with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_camera.json')) as f:
    cam_params = json.load(f)
with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_joint_3d.json')) as f:
    joints = json.load(f)

img_path_list = glob(osp.join(img_root_path, split, 'Capture' + capture_idx, seq_name, 'cam' + cam_idx, '*.jpg'))
for img_path in img_path_list:
    frame_idx = img_path.split('/')[-1][5:-4]
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    
    for hand_type in ('right', 'left'):
        # camera extrinsic parameters
        cam_param = cam_params[capture_idx]
        t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3,3)
        t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t

        # gt 3D joint coordinate
        joint = np.array(joints[str(capture_idx)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3) # world coordinate
        joint = np.dot(R, joint.transpose(1,0)).transpose(1,0) + t.reshape(1,3) # apply camera extrinsic to convert world coordinates to camera coordinates
        joint_valid = np.array(joints[str(capture_idx)][str(frame_idx)]['joint_valid'], dtype=np.float32).reshape(-1,1)

        # mano parameter
        try:
            mano_param = mano_params[capture_idx][frame_idx][hand_type]
            if mano_param is None:
                continue
        except KeyError:
            continue

        ####################
        # First world to camera coordinate conversion method: apply camera extrinsics to world coordinates
        # get MANO 3D mesh coordinates (world coordinate)
        mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
        root_pose = mano_pose[0].view(1,3)
        hand_pose = mano_pose[1:,:].view(1,-1)
        shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
        trans = torch.FloatTensor(mano_param['trans']).view(1,3)
        output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
        mesh = output.vertices[0].numpy() * 1000 # meter to milimeter (world coordinate)
        mesh = np.dot(R, mesh.transpose(1,0)).transpose(1,0) + t.reshape(1,3) # apply camera extrinsic to convert world coordinates to camera coordinates
        
        # fitting error
        fit_err = get_fitting_error(mesh, joint_regressor, joint, joint_valid, hand_type) # error between GT 3D joint coordinates in camera coordinate system
        print('Fitting error of the first method: ' + str(fit_err) + ' mm')

        ####################
        # Second world to camera coordinate conversion method: apply camera extrinsics to MANO parameters
        # get MANO 3D mesh coordinates (world coordinate)
        mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
        root_pose = mano_pose[0].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose)) # multiply camera rotation to MANO root pose
        root_pose = torch.from_numpy(root_pose).view(1,3)
        hand_pose = mano_pose[1:,:].view(1,-1)
        shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
        output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape) # this is rotation-aligned, but not translation-aligned with the camera coordinates
        mesh = output.vertices[0].detach().numpy() 
        joint_from_mesh = np.dot(joint_regressor, mesh)
        
        
        # compenstate rotation (translation from origin to root joint was not cancled)
        root_joint = joint_from_mesh[root_joint_idx,None,:]
        trans = np.array(mano_param['trans'])
        trans = np.dot(R, trans.reshape(3,1)).reshape(1,3) - root_joint + np.dot(R, root_joint.transpose(1,0)).transpose(1,0) + t / 1000 # change translation vector
        trans = torch.from_numpy(trans).view(1,3)
        output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans) 
        mesh = output.vertices[0].numpy() * 1000 # meter to milimeter (camera coordinate)
       
        # fitting error
        fit_err = get_fitting_error(mesh, joint_regressor, joint, joint_valid, hand_type) # error between GT 3D joint coordinates in camera coordinate system
        print('Fitting error of the second method: ' + str(fit_err) + ' mm')


 
