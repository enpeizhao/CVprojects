# -*- coding : utf-8 -*-
# @Time     : 2023/5/16 - 14:43
# @Author   : enpei
# @FileName : visualize.py
#
from __future__ import division

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math


def visualize_box_mask(im, results, labels, color, threshold=0.5):
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    elif isinstance(im, np.ndarray):
        im = Image.fromarray(im)

    if 'boxes' in results and len(results['boxes']) > 0:
        im = draw_box(im, results['boxes'], labels, color, threshold=threshold)
    return im


def draw_box(image, np_boxes, labels, color, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        color (tuple): color
        threshold (float): threshold of box
    Returns:
        im (PIL.Image.Image): visualized image
    """
    draw_thickness = min(image.size) // 320
    draw = ImageDraw.Draw(image)
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                  'right_bottom:[{:.2f},{:.2f}]'.format(int(clsid), score, xmin, ymin, xmax, ymax))
            # draw bbox
            draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),(xmin, ymin)], width=draw_thickness, fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)], width=2, fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)

        text = "{} {:.4f}".format(labels[clsid], score)
        #tw, th = draw.textsize(text)
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text)
        tw, th = right - left,  bottom - top
        draw.rectangle([(x1 + 1, y1 - th), (x1 + tw + 1, y1)], fill=color)
        draw.text((x1 + 1, y1 - th), text, fill=(255, 255, 255))
    return image