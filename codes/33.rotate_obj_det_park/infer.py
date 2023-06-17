# -*- coding : utf-8 -*-
# @Time     : 2023/5/16 - 14:35
# @Author   : enpei
# @FileName : infer.py
#
import os
import yaml
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, Pad, decode_image
from visualize import visualize_box_mask
from utils import Timer


class Detector(object):
    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 cpu_threads=1,
                 output_dir='output',
                 threshold=0.5):
        self.pred_config = self.set_config(model_dir)
        self.predictor, self.config = load_predictor(
            model_dir,
            self.pred_config.arch,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            cpu_threads=cpu_threads,
        )
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_list = []
        input_im_info_list = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_list.append(im)
            input_im_info_list.append(im_info)
        inputs = create_inputs(input_im_list, input_im_info_list)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        np_boxes_num = result['boxes_num']
        assert isinstance(np_boxes_num, np.ndarray), '`np_boxes_num` should be a `numpy.ndarray`'
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def predict(self, repeats=1):
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor 'bbox_num'
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ['masks', 'segm']:
                results[k] = np.concatenate(v)
        return results

    def get_timer(self):
        return self.det_times

    def predict_image(self,
                      image_list,
                      visual=True,
                      save_results=False):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]

            self.det_times.preprocess_time_s.start()
            inputs = self.preprocess(batch_image_list)
            self.det_times.preprocess_time_s.end()

            # model prediction
            self.det_times.inference_time_s.start()
            result = self.predict()
            self.det_times.inference_time_s.end()

            # postprocess
            self.det_times.postprocess_time_s.start()
            result = self.postprocess(inputs, result)
            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += len(batch_image_list)
            if visual:
                visualize(
                    batch_image_list,
                    result,
                    self.pred_config.labels,
                    output_dir='output',
                    threshold=self.threshold)
            results.append(result)
            print('Test iter {}'.format(i))

        results = self.merge_batch_result(results)
        if save_results:
            Path(self.output_dir).mkdir(exist_ok=True)
            self.save_coco_results(image_list, results, use_coco_category=False)
        return results

    def predict_video(self, video_file, camera_id):
        video_out_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 1
        while (1):
            ret, frame = capture.read()
            if not ret: break
            print('dectect frame: %d' % (index))
            index += 1
            results = self.predict_image([frame[:, :, ::-1]], visual=False)
            im = visualize_box_mask(
                frame,
                results,
                self.pred_config.labels,
                color=(255, 0, 0),
                threshold=self.threshold)
            im = np.array(im)
            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()

    def save_coco_results(self, image_list, results, use_coco_category=False):
        bbox_results = []
        idx = 0
        print("Start saving coco json files...")
        for i, box_num in enumerate(results['boxes_num']):
            file_name = os.path.split(image_list[i])[-1]
            if use_coco_category:
                img_id = int(os.path.splitext(file_name)[0])
            else:
                img_id = i

            if 'boxes' in results:
                boxes = results['boxes'][idx:idx + box_num].tolist()
                bbox_results.extend([{
                    'image_id': img_id,
                    'category_id': int(box[0]),
                    'file_name': file_name,
                    'bbox': [box[2:]],  # xyxy -> xywh
                    'score': box[1]} for box in boxes])

            idx += box_num

        if bbox_results:
            bbox_file = os.path.join(self.output_dir, "bbox.json")
            with open(bbox_file, 'w') as f:
                json.dump(bbox_results, f)
            print(f"The bbox result is saved to {bbox_file}")


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print('The RCNN export model is used for ONNX and it only supports batch_size = 1')
        self.print_config()

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def create_inputs(imgs, im_info):
    inputs = {}
    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array((im_info[0]['im_shape'],)).astype('float32')
        inputs['scale_factor'] = np.array((im_info[0]['scale_factor'],)).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'],)).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'],)).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros((im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


def load_predictor(model_dir,
                   arch,
                   run_mode='paddle',
                   batch_size=2,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   cpu_threads=1,
                   ):
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError("Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(run_mode, device))
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    if not os.path.exists(infer_model):
        infer_model = os.path.join(model_dir, 'inference.pdmodel')
        infer_params = os.path.join(model_dir, 'inference.pdiparams')
        if not os.path.exists(infer_model):
            raise ValueError("Cannot find any inference model in dir: {},".format(model_dir))
    config = Config(infer_model, infer_params)
    if device == 'GPU':
        config.enable_use_gpu(200, 0)
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)

    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config


def visualize(image_list, result, labels, output_dir='output/', threshold=0.5):
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = result['boxes_num'][idx]
        im_results = {}
        if 'boxes' in result:
            im_results['boxes'] = result['boxes'][start_idx:start_idx + im_bboxes_num, :]
        start_idx += im_bboxes_num
        im = visualize_box_mask(
            image_file, im_results, labels, color=(255, 0, 0), threshold=threshold)
        img_name = os.path.split(image_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        im.save(out_path, quality=95)
        print("save result to: " + out_path)
