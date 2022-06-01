import sys
import os
sys.path.append("./DeOldify")

# 导入deoldify相关组件
from deoldify.filters import IFilter, MasterFilter, ColorizerFilter
from deoldify.generators import gen_inference_deep, gen_inference_wide
from pathlib import Path
from PIL import Image

# 导入RealSR相关组件
sys.path.append("./RealSR")
import options.options as option
import utils.util as util
from models import create_model

import cv2
import torch
import numpy as np

# 导入FOM
sys.path.append("./first-order-model")
import yaml
from tqdm import tqdm
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

class GanApp:
    def __init__(self):
        # 初始化 deoldify
        learn_stable = gen_inference_wide(root_folder=Path('./DeOldify'), weights_name='ColorizeStable_gen')
        self.deoldify_model = MasterFilter([ColorizerFilter(learn=learn_stable)], render_factor=10)
        print('初始化deoldify ok')
        # 初始化超分
        self.sr_opt = option.parse('./config/test_dped.yml', is_train=False)
        self.sr_opt = option.dict_to_nonedict(self.sr_opt)
        
        self.sr_model = create_model(self.sr_opt)
        print('初始化RealSR ok')

        # 加载FOM模型
        checkpoint = './first-order-model/weights/vox-adv-cpk.pth.tar'
        config = './first-order-model/config/vox-adv-256.yaml'

        self.fom_generator, self.fom_kp_detector = self.load_fom_checkpoints(config_path=config, checkpoint_path=checkpoint, cpu=False)
        print('初始化FOM ok')

    def deoldify(self,img,render_factor=35):
        """
        风格化
        """
        # 转换通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        # 渲染彩图
        color_img = self.deoldify_model.filter(
            pil_img, pil_img, render_factor=render_factor,post_process=True
        )
        color_img = np.asarray(color_img)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        # 转为numpy图
        print('deoldify 转换成功')
        return np.asarray(color_img)

 

    def srGetImg( self,img):
        """
        RealSR预处理图片
        """
        # 归一化
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        
        H, W, C = img.shape
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        return {'LQ': torch.unsqueeze(img, 0) , 'LQ_path': ''}

    def realSR(self,img):
        """
        超分辨率
        """
        need_GT = False
        # 处理图片
        data = self.srGetImg(img)
        self.sr_model.feed_data(data, need_GT=need_GT)

        if self.sr_opt['model'] == 'sr':
            self.sr_model.test_x8()
        elif self.sr_opt['large'] is not None:
            self.sr_model.test_chop()
        else:
            self.sr_model.test()
        if self.sr_opt['back_projection'] is not None and self.sr_opt['back_projection'] is True:
            self.sr_model.back_projection()
        visuals = self.sr_model.get_current_visuals(need_GT=need_GT)

        sr_img = util.tensor2img(visuals['SR'])  # uint8
        print('realSR 转换成功')
        return sr_img

    def load_fom_checkpoints(self,config_path, checkpoint_path, cpu=False):
        """
        加载FOM模型
        """

        with open(config_path) as f:
            config = yaml.load(f)

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()
        
        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
    
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        
        if not cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()
        
        return generator, kp_detector

    def make_animation(self, source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
        """
        制作动画
        """
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                    use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

    def FOM_video(self,driving_video,source_image,result_video):

        # 读取图片
        source_image = imageio.imread(source_image)
        # 读取视频
        reader = imageio.get_reader(driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        # 预处理
        source_image = resize(source_image, (255, 255))[..., :3]
        driving_video = [resize(frame, (255, 255))[..., :3] for frame in driving_video]
        
        # 推理
        predictions = self.make_animation(source_image, driving_video, self.fom_generator, self.fom_kp_detector, relative=True, adapt_movement_scale=True, cpu=False)
        # 保存
        imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


# 实例化
gan = GanApp()
# 读取图片
test_img = cv2.imread('./images/4.jpg')
# 风格化
deold_img = gan.deoldify(test_img)   
cv2.imwrite('./images/out1.jpg',deold_img)     
# 超分
sr_img = gan.realSR(deold_img)            
cv2.imwrite('./images/out2.jpg',sr_img)   

driving_video = './images/xc.mp4'
source_image = './images/4.jpg'
result_video = './images/result1.mp4'
# 图像动起来
gan.FOM_video(driving_video, source_image,result_video)

