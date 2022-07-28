'''
demo演示
'''
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import torch
import mmcv

class RoadSeg:
    def __init__(self, config_path ,model_path):
        '''
        @param model_path: 模型路径
        @param config_path: 配置文件路径
        '''
        self.config_path = config_path
        self.model_path = model_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        self.model = init_detector(self.config_path, self.model_path, device=self.device)
        # 参数        
        self.CLASSES = ('arrow', 'car', 'dashed', 'line')
        self.colors  = [(255,0,255), (0,255,255), (0,255,0), (0,0,255)]
        self.alpha_list = [0.3, 0.5, 0.3, 0.3]
    
    def convertResult(self, result):
        '''
        @param result: 检测结果
        @return: bboxes, labels, segms 结果
        '''
        bbox_result, segm_result = result
        # 将结果转换为numpy数组
        bboxes = np.vstack(bbox_result)
        # 制作标签id
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # 制作分割mask
        segms = mmcv.concat_list(segm_result)
        segms = np.stack(segms, axis=0)

        return bboxes, labels, segms

    def inference(self, video_path,threshold=0.3):
        '''
        @param video_path: 视频路径
        '''
        cap = cv2.VideoCapture(video_path)
        # videoWriter保存为mp4视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 创建视频写入对象
        # 获取视频文件名
        video_name = video_path.split('/')[-1].split('.')[0]
        # 拼接
        record_video_path = './record_video/' + video_name + '.mp4'
        videoWriter = cv2.VideoWriter(record_video_path, fourcc, fps, (width, height))
        

        while True:
            ret, frame = cap.read()
            # 缩放为一半

            if not ret:
                break
            result = inference_detector(self.model, frame)
            bboxes, labels, segms = self.convertResult(result)
          
            # 遍历每一个检测结果
            for i,box in enumerate(bboxes):
                # 如果检测结果的置信度大于阈值
                conf =  box[-1]
                if conf > threshold:
                    # 检测到的框
                    l,t,r,b = box[:4].astype('int')
                    # 对应mask
                    seg = segms[i]
                    # 若mask为true，则将frame中的像素置为半透明
                    alpha = self.alpha_list[labels[i]]
                    color = self.colors[labels[i]]
                    frame[seg > 0,0] = frame[seg > 0,0] * alpha + color[0] * (1-alpha)
                    frame[seg > 0,1] = frame[seg > 0,1] * alpha + color[1] * (1-alpha)
                    frame[seg > 0,2] = frame[seg > 0,2] * alpha + color[2] * (1-alpha)
              
                    # 绘制检测框
                    # cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
                    # 绘制类别
                    # cv2.putText(frame,str(self.CLASSES[labels[i]]),(l,t-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
           
            # 写入视频
            videoWriter.write(frame)
            # 显示结果
            # resize
            frame = cv2.resize(frame, (int(width/2), int(height/2)))
            cv2.imshow('result',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 实例化
    road_seg = RoadSeg( './weights/25-swin-transfomer-tiny.py' , './weights/25-swin-transfomer-tiny.pth')
    # 视频路径
    video_path = 'imgs/25-swin-transfomer-test-video.mov'
    # 检测结果
    road_seg.inference(video_path)

