"""
! author : enpei
! data: 2021-12-08
封装百度Paddle detection 和 OCR

"""
import sys
import os
import cv2
# detection相关包(windows路径格式，注意调整)
sys.path.append("baidu_pp_detection\\python")

from infer import Config,Detector
from visualize import visualize_box_mask, lmk2out
import numpy as np


# OCR相关包(windows路径格式，注意调整)
sys.path.append("baidu_pp_ocr\\tools\\infer")
sys.path.append("baidu_pp_ocr\\")
import utility as utility
from predict_system import TextSystem
from ppocr.utils.logging import get_logger
logger = get_logger()

# 物体识别
class Baidu_PP_Detection:
    def __init__(self):
        # 初始化detection模型和OCR模型
        self.model_dir = './baidu_pp_detection/models/cascade_rcnn_dcn_r101_vd_fpn_gen_server_side'
        config = Config(self.model_dir)
        self.labels_en = config.labels
        self.labels_zh = self.get_label_zh()
        self.ob_detector = Detector(
            config,
            self.model_dir,
            device="GPU",
            run_mode='fluid',
            trt_calib_mode=False)
         # 热身
        if 1:
            print('Warm up detection model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                im,results = self.detect_img(img)

    # 获取对应中文label        
    def get_label_zh(self):
        file_path = self.model_dir+'/generic_det_label_list_zh.txt'
        back_list = []
        with open(file_path,'r',encoding='utf-8') as label_text:
            for label in  label_text.readlines():
                back_list.append(label.replace('\n',''))
        return back_list
                
    #侦测图片    
    def detect_img(self,img):
        results = self.ob_detector.predict(img, 0.5)
        im = visualize_box_mask(
            img,
            results,
            self.ob_detector.config.labels,
            mask_resolution=self.ob_detector.config.mask_resolution,
            threshold=0.5)
        im = np.array(im)
        return im,results


    # 测试识别物体
    def test_predict_video(self,camera_id):
        capture = cv2.VideoCapture(camera_id)
    
        index = 1
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            print('detect frame:%d' % (index))
            index += 1
            
            im,results = self.detect_img(frame)
            for box in results['boxes']:
                # 类别，英文，中文
                label_id = box[0].astype(int)
                print('##',label_id,self.labels_en[label_id],self.labels_zh[label_id-1])

            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# OCR
class Baidu_PP_OCR:
    def __init__(self):

        args = utility.parse_args()
        
        args.det_model_dir="./baidu_pp_ocr/models/ch_PP-OCRv2_det_infer/"
        args.rec_model_dir="./baidu_pp_ocr/models/ch_PP-OCRv2_rec_infer/"
        args.rec_char_dict_path="./baidu_pp_ocr/ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls=False 
        args.use_gpu=True

        self.text_sys = TextSystem(args)

        # 热身
        if 1:
            print('Warm up ocr model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)
    
    def ocr_image(self,img):
        
        dt_boxes, rec_res = self.text_sys(img)
        text_list = []
        for text, score in rec_res:
                # logger.info("{}, {:.3f}".format(text, score))
                text_list.append(text)
        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        return src_im,text_list

    def test_ocr(self):
        image_dir="./fapiao.png" 
        img = cv2.imread(image_dir)
        src_im,text_list = self.ocr_image(img)
        print(text_list)
        cv2.imwrite('./output.jpg',src_im)

# ocr = Baidu_PP_OCR()
# ocr.test_ocr()

# dete = Baidu_PP_Detection()
# dete.test_predict_video(0)