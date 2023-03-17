#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh = 0.5);

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

void yolo_nms(std::vector<Detection>& res, int32_t* num_det, int32_t* cls, float* conf, float* bbox, float conf_thresh, float nms_thresh);

float iou(float lbox[4], float rbox[4]);

static bool cmp(const Detection& a, const Detection& b) {
  return a.conf > b.conf;
}
