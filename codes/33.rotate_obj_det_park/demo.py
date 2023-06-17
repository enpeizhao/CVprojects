# -*- coding : utf-8 -*-
# @Time     : 2023/5/16 - 16:01
# @Author   : enpei
# @FileName : demo.py
#
import os
import numpy as np
import cv2
import paddle
import shutil

from infer import Detector
from parking import load_park_data, compute_one_image_car_park_infos, update_infos
# 导入PIL
from PIL import Image, ImageDraw, ImageFont

class ParkingLot(object):
    def __init__(self) -> None:
        # 公共内容
        self.device = "GPU" if "gpu" in paddle.get_device() else "CPU"
        self.model_dir = './output_inference_vel/ppyoloe_r_crn_m_3x_dota/'
        park_info_file = '车位框数据/space1.json'
        self.park_data, self.num_of_park = load_park_data(park_info_file)
        

    # 绘制中文
    def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/MSYH.TTC", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def visualize(self, image, park_info, single_match_data, stat_info):

        h, w = image.shape[:2]
        # draw park boundary
        for park_id in range(self.num_of_park):
            pts = np.asarray(self.park_data[park_id]['points'], dtype=np.int32)
            if park_info[park_id]['occupied']:
                # occupied, red
                color = (0, 0, 255)
            else:
                # unoccupied, green
                color = (0, 255, 0)
                cv2.fillPoly(image, [pts], (0, 255, 0))
            
            cv2.polylines(image, [pts], True, color=color, thickness=4)
            


        for parked_car in single_match_data['parked']:
            # parked car, blue
            pts = np.asarray(parked_car['points'], dtype=np.int32)
            cv2.polylines(image, [pts], True, color=(255, 0, 0), thickness=2)
        for not_parked_car in single_match_data['not_parked']:
            # not parked car, yellow
            pts = np.asarray(not_parked_car['points'], dtype=np.int32)
            cv2.polylines(image, [pts], True, color=(0, 255, 255), thickness=4)

  
        # set transparent backgroud
        color = (0,0,0)
        alpha = 0.2
        l,t = int(0.37*w -15 ), int(0.2*h - 10)
        r,b = l+300,t+240
        image[t:b,l:r,0] = image[t:b,l:r,0] * alpha + color[0] * (1-alpha)
        image[t:b,l:r,1] = image[t:b,l:r,1] * alpha + color[1] * (1-alpha)
        image[t:b,l:r,2] = image[t:b,l:r,2] * alpha + color[2] * (1-alpha)


        image = self.cv2AddChineseText(image, "车位总数：" + str(stat_info['park_nums']), (int(0.37*w), int(0.20*h)), textColor=(255, 255, 255), textSize=40)
        image = self.cv2AddChineseText(image, "空闲车位：" + str(stat_info['free_parks']), (int(0.37*w), int(0.225*h)), textColor=(0, 255, 0), textSize=40)
        image = self.cv2AddChineseText(image, "占用车位：" + str(stat_info['occupied_parks']), (int(0.37*w), int(0.25*h)), textColor=(255, 0, 255), textSize=40)
        image = self.cv2AddChineseText(image, "未停车辆：" + str(stat_info['no_parked_cars']), (int(0.37*w), int(0.275*h)), textColor=(255, 255, 0), textSize=40)
        return image

    def run_on_video(self, video_file,  output_dir='output'):
        # init detector
        detector = Detector(self.model_dir, device=self.device, threshold=0.5)

        video_out_name = os.path.split(video_file)[-1]
        capture = cv2.VideoCapture(video_file)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('output.mp4', fourcc, fps, (height, width))
        index = 0

        # aligned park h and w, used to compute the park boundary
        aligned_park_h = 2160
        aligned_park_w = 3840
        # model input h and w
        input_h = 576
        input_w = 1024

        while (1):
       
            ret, frame = capture.read()
            if not ret:
                break
            print('dectect frame: %d' % (index))
            index += 1

        
            # another size
            render_img = np.zeros((aligned_park_h, aligned_park_w, 3), dtype=np.uint8)

            # letterbox
            scale = min(aligned_park_w / img_w, aligned_park_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            resized_image = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            render_img[(aligned_park_h - new_h) // 2:(aligned_park_h - new_h) // 2 + new_h,
                      (aligned_park_w - new_w) // 2:(aligned_park_w - new_w) // 2 + new_w, :] = resized_image

            input_img = cv2.resize(render_img, ( input_w, input_h), interpolation=cv2.INTER_CUBIC

            # inference
            results = detector.predict_image(
                [input_img[:, :, ::-1]], visual=False, save_results=False)
            detector.det_times.info()

            match_data = compute_one_image_car_park_infos(
                self.park_data, results, conf_threshold=0.8, iou_threshold=0.25)
            stat_info, park_info, single_match_data = update_infos(
                self.num_of_park, self.park_data, match_data)

            # visualize
            vis_img = self.visualize(render_img, park_info,
                                     single_match_data, stat_info)
            writer.write(vis_img)
            
            vis_img = cv2.resize(vis_img, ( input_w, input_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("./tmp.jpg", vis_img)
            
        writer.release()


if __name__ == '__main__':

    # delete all the files under output
    shutil.rmtree('./output')
    # init class
    pp = ParkingLot()
    video_file = 'videos/test1.mp4'
    pp.run_on_video(video_file )
    print('Done...')