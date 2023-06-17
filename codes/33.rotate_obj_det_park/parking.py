# -*- coding : utf-8 -*-
# @Time     : 2023/5/17 - 15:09
# @Author   : enpei
# @FileName : parking.py
#
import os
import json
import numpy as np
from shapely.geometry import Polygon
from copy import deepcopy


def load_park_data(json_file):
    with open(json_file, 'r', encoding='utf8') as fd:
        park_json = json.load(fd)

    shapes = sorted(park_json['shapes'], key=lambda x: x['points'][1])
    number_of_parks = len(shapes)
    park_data = {}
    for idx, shape in enumerate(shapes):
        type = shape['shape_type']
        points = np.array(shape['points']).reshape(-1).tolist()
        if type == 'rectangle':
            xmin, ymin, xmax, ymax = points
            points = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        points = np.array(points, dtype=np.float32).reshape(-1, 2)

        xmin = min(points[:, 0])
        xmax = max(points[:, 0])
        ymin = min(points[:, 1])
        ymax = max(points[:, 1])
        park_data[idx] = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'points': points,
            'occupied': False,
            'idle_frames': 0,
            'occupied_frame_idx': 0
        }
    return park_data, number_of_parks


def compute_iou(bbox1, bbox2):
    bbox1_poly = Polygon(bbox1)
    bbox2_poly = Polygon(bbox2)
    bbox2_poly.buffer(0)
    bbox1_poly.buffer(0)
    if not bbox1_poly.is_valid or not bbox2_poly.is_valid:
        return 0.0
    inter_area = bbox1_poly.intersection(bbox2_poly).area
    union_area = bbox1_poly.area + bbox2_poly.area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_one_image_car_park_infos(park_data, car_det_res, conf_threshold=0.4, iou_threshold=0.3):
    aligned_park_h = 2160
    aligned_park_w = 3840
    input_h = 576
    input_w = 1024

    car_num = car_det_res['boxes_num']
    # car_det_res = sorted(car_det_res, key=lambda x: x['boxes'][3])
    if isinstance(car_num, list):
        car_num = int(car_num[0])
    else:
        car_num = int(car_num)

    matched_res = {
        'parked': [],
        'not_parked': []
    }
    car_parked = np.zeros(car_num)
    for idx in range(car_num):
        label_id, score, rbox = car_det_res['boxes'][idx][0], car_det_res['boxes'][idx][1], car_det_res['boxes'][idx][2:]
        # resize rbox to align with park
        
        rbox = np.array(rbox).reshape(-1, 2)
        rbox[:, 0] = rbox[:, 0] * aligned_park_w / input_w
        rbox[:, 1] = rbox[:, 1] * aligned_park_h / input_h
        
        if score < conf_threshold:
            continue
        rbox_np = np.array(rbox).reshape(-1, 2)
        for park_id in park_data.keys():
            park_info = park_data[park_id]
            if max(rbox_np[:, 0]) < park_info['xmin'] or min(rbox_np[:, 0]) > park_info['xmax'] or \
                max(rbox_np[:, 1]) < park_info['ymin'] or min(rbox_np[:, 1]) > park_info['ymax']:
                continue
            park_pts = park_info['points']
            iou = compute_iou(rbox_np, park_pts)
            if iou > iou_threshold:
                car_parked[idx] = 1
                matched_res['parked'].append({
                    'park_id': park_id,
                    'points': rbox_np,
                    'car_id': idx
                })
    not_parked_cars = car_det_res['boxes'][car_parked == 0]
    print(f'not parked car : {len(not_parked_cars)}')
    for not_parked in not_parked_cars:
        # print(f'not not_parked : {not_parked}')
        if not_parked[1] < conf_threshold:
            continue
        # resize rbox to align with park
        rbox = np.array(not_parked[2:]).reshape(-1, 2)
        rbox[:, 0] = rbox[:, 0] * aligned_park_w / input_w
        rbox[:, 1] = rbox[:, 1] * aligned_park_h / input_h

        # matched_res['not_parked'].append({
        #     'points': np.array(not_parked[2:]).reshape(-1, 2)
        # })
        matched_res['not_parked'].append({
            'points': rbox 
        })

    return matched_res


def update_infos(park_nums, park_data, single_matched_data):
    stat_infos = {
        "park_nums": park_nums,
        "occupied_parks": 0,
        "free_parks": 0,
        "no_parked_cars": 0,
        "total_cars": 0
    }

    # update parks
    parked_cars_num = 0
    free_parks_np = np.zeros(park_nums)
    for parked_car in single_matched_data['parked']:
        parked_cars_num += 1
        park_id = parked_car['park_id']
        free_parks_np[park_id] = 1
        park_data[park_id]['occupied'] = True

    for idx, occupied in enumerate(free_parks_np):
        if occupied == 1:
            continue
        park_data[idx]['idle_frames'] += 1
        if park_data[idx]['occupied'] and park_data[idx]['idle_frames'] > 5:
            park_data[idx]['occupied'] = False

    stat_infos['total_cars'] = parked_cars_num + len(single_matched_data['not_parked'])
    stat_infos['no_parked_cars'] = len(single_matched_data['not_parked'])

    for park_id, park in park_data.items():
        if park['occupied']:
            stat_infos['occupied_parks'] += 1
        else:
            stat_infos['free_parks'] += 1

    return stat_infos, park_data, single_matched_data


if __name__ == '__main__':
    json_file = '车位框数据/车位框底图.json'
    res, num = load_park_data(json_file)
    print(num)
    print(res)
