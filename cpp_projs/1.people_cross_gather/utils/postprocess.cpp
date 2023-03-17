#include "postprocess.h"
#include "utils.h"

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {

    float scale = std::min(kInputH / float(img.cols), kInputW / float(img.rows));
    int offsetx = (kInputW - img.cols * scale) / 2; 
    int offsety = (kInputH - img.rows * scale) / 2; 

    size_t output_width = img.cols;
    size_t output_height = img.rows;

    float x1 = (bbox[0] - offsetx) / scale;
    float y1 = (bbox[1] - offsety) / scale;
    float x2 = (bbox[2] - offsetx) / scale;
    float y2 = (bbox[3] - offsety) / scale;

    x1 = clamp(x1, 0, output_width);
    y1 = clamp(y1, 0, output_height);
    x2 = clamp(x2, 0, output_width);
    y2 = clamp(y2, 0, output_height);

    auto left = x1;
    auto width = clamp(x2 - x1, 0, output_width);
    auto top = y1;
    auto height = clamp(y2 - y1, 0, output_height);

    return cv::Rect(left, top, width, height);
}

float iou(float lbox[4], float rbox[4]) {
  float interBox[] = {
    (std::max)(lbox[0], rbox[0]), //left
    (std::min)(lbox[2], rbox[2]), //right
    (std::max)(lbox[1], rbox[1]), //top
    (std::min)(lbox[3], rbox[3]), //bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
  return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3]-lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS);
}


void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
  int det_size = sizeof(Detection) / sizeof(float);
  std::map<float, std::vector<Detection>> m;
  for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; i++) {
    if (output[1 + det_size * i + 4] <= conf_thresh) continue;
    Detection det;
    memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
    if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
    m[det.class_id].push_back(det);
  }
  for (auto it = m.begin(); it != m.end(); it++) {
    auto& dets = it->second;
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m) {
      auto& item = dets[m];
      res.push_back(item);
      for (size_t n = m + 1; n < dets.size(); ++n) {
        if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
          dets.erase(dets.begin() + n);
          --n;
        }
      }
    }
  }
}

void batch_nms(std::vector<std::vector<Detection>>& res_batch, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh) {
  res_batch.resize(batch_size);
  for (int i = 0; i < batch_size; i++) {
    nms(res_batch[i], &output[i * output_size], conf_thresh, nms_thresh);
  }
}

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch) {
  for (size_t i = 0; i < img_batch.size(); i++) {
    auto& res = res_batch[i];
    cv::Mat img = img_batch[i];
    for (size_t j = 0; j < res.size(); j++) {
      cv::Rect r = get_rect(img, res[j].bbox);
      cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
  }
}

static cv::Rect get_downscale_rect(float bbox[4], float scale) {
  float left = bbox[0] - bbox[2] / 2;
  float top = bbox[1] - bbox[3] / 2;
  float right = bbox[0] + bbox[2] / 2;
  float bottom = bbox[1] + bbox[3] / 2;
  left /= scale;
  top /= scale;
  right /= scale;
  bottom /= scale;
  return cv::Rect(round(left), round(top), round(right - left), round(bottom - top));
}

void yolo_nms(std::vector<Detection>& res, int32_t* num_det, int32_t* cls, float* conf, float* bbox, float conf_thresh, float nms_thresh) {
  res.clear();
  std::map<int32_t, std::vector<Detection>> m;
  for (int i = 0; i < num_det[0]; i++) {
    if (conf[i] <= conf_thresh) continue;
    Detection det;
    det.bbox[0] = bbox[i * 4 + 0];
    det.bbox[1] = bbox[i * 4 + 1];
    det.bbox[2] = bbox[i * 4 + 2];
    det.bbox[3] = bbox[i * 4 + 3];
    det.conf = conf[i];
    det.class_id = cls[i];
    if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
    m[det.class_id].push_back(det);
  }
  for (auto it = m.begin(); it != m.end(); it++) {
    auto& dets = it->second;
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t i = 0; i < dets.size(); ++i) {
      auto& item = dets[i];
      res.push_back(item);
      for (size_t j = i + 1; j < dets.size(); ++j) {
        if (iou(item.bbox, dets[j].bbox) > nms_thresh) {
          dets.erase(dets.begin() + j);
          --j;
        }
      }
    }
  }
}