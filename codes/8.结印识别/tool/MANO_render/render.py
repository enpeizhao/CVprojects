# Copyright (c) Facebook, Inc. and its affiliates.

import os
import numpy as np
import cv2
import json
from glob import glob
import os.path as osp
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
import smplx
import torch

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()
    
# mano layer
smplx_path = 'SMPLX_PATH'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}

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

save_path = osp.join(subset, split, capture_idx, seq_name, cam_idx)
os.makedirs(save_path, exist_ok=True)

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
    
    prev_depth = None
    for hand_type in ('right', 'left'):
        # get mesh coordinate
        try:
            mano_param = mano_params[capture_idx][frame_idx][hand_type]
            if mano_param is None:
                continue
        except KeyError:
            continue
        
        # get MANO 3D mesh coordinates (world coordinate)
        mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
        root_pose = mano_pose[0].view(1,3)
        hand_pose = mano_pose[1:,:].view(1,-1)
        shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
        trans = torch.FloatTensor(mano_param['trans']).view(1,3)
        output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
        mesh = output.vertices[0].numpy() * 1000 # meter to milimeter
        
        # apply camera extrinsics
        cam_param = cam_params[capture_idx]
        t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3,3)
        t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
        mesh = np.dot(R, mesh.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
        
        # save mesh to obj files
        save_obj(mesh, mano_layer[hand_type].faces, osp.join(save_path, img_path.split('/')[-1][:-4] + '_' + hand_type + '.obj'))
        
        # mesh
        mesh = mesh / 1000 # milimeter to meter
        mesh = trimesh.Trimesh(mesh, mano_layer[hand_type].faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        # add camera intrinsics
        focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
        princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        scene.add(camera)
        
        # renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height, point_size=1.0)
       
        # light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        # render
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:,:,:3].astype(np.float32)
        depth = depth[:,:,None]
        valid_mask = (depth > 0)
        if prev_depth is None:
            render_mask = valid_mask
            img = rgb * render_mask + img * (1 - render_mask)
            prev_depth = depth
        else:
            render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth==0)
            img = rgb * render_mask + img * (1 - render_mask)
            prev_depth = depth * render_mask + prev_depth * (1 - render_mask)

    # save image
    cv2.imwrite(osp.join(save_path, img_path.split('/')[-1]), img)
