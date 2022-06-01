import options.options as option
import utils.util as util
import data.util as data_util
from models import create_model

import cv2
import torch
import numpy as np

opt = option.parse('options/dped/test_dped.yml', is_train=False)
opt = option.dict_to_nonedict(opt)

model = create_model(opt)

def getImg( LR_path):

    # get LR image
    img_LR = data_util.read_img(None, LR_path)
    H, W, C = img_LR.shape

    # BGR to RGB, HWC to CHW, numpy to tensor
    if img_LR.shape[2] == 3:
        img_LR = img_LR[:, :, [2, 1, 0]]
    img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

    return {'LQ': torch.unsqueeze(img_LR, 0) , 'LQ_path': LR_path}


need_GT = False
img_file = './image_test/00002.png'
data = getImg(img_file)
model.feed_data(data, need_GT=need_GT)

if opt['model'] == 'sr':
    model.test_x8()
elif opt['large'] is not None:
    model.test_chop()
else:
    model.test()
if opt['back_projection'] is not None and opt['back_projection'] is True:
    model.back_projection()
visuals = model.get_current_visuals(need_GT=need_GT)

sr_img = util.tensor2img(visuals['SR'])  # uint8

cv2.imwrite('./output/res.jpg', sr_img)

        