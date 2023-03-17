#include "preprocess.h"
#include "cuda_utils.h"
#include "config.h"

static uint8_t *img_buffer_device = nullptr;

struct AffineMatrix
{
  float value[6];
};

// 一个线程处理一个像素点
__global__ void preprocess_kernel(
    uint8_t *src, float *dst, int dst_width,
    int dst_height, int edge)
{
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge)
    return;

  int dx = position % dst_width; // 计算当前线程对应的目标图像的x坐标
  int dy = position / dst_width; // 计算当前线程对应的目标图像的y坐标

  // normalization（对原图中(x,y)坐标的像素点3个通道进行归一化）
  float c0 = src[dy * dst_width * 3 + dx * 3 + 0] / 255.0f;
  float c1 = src[dy * dst_width * 3 + dx * 3 + 1] / 255.0f;
  float c2 = src[dy * dst_width * 3 + dx * 3 + 2] / 255.0f;

  // bgr to rgb
  float t = c2;
  c2 = c0;
  c0 = t;

  // rgbrgbrgb to rrrgggbbb
  // NHWC to NCHW
  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

__global__ void warpaffine_kernel(
    uint8_t *src, int src_line_size, int src_width,
    int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge)
{
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge)
    return;

  // 从d2s中读取变换矩阵
  float m_x1 = d2s.value[0];
  float m_y1 = d2s.value[1];
  float m_z1 = d2s.value[2];
  float m_x2 = d2s.value[3];
  float m_y2 = d2s.value[4];
  float m_z2 = d2s.value[5];

  int dx = position % dst_width; // 计算当前线程对应的目标图像的x坐标
  int dy = position / dst_width; // 计算当前线程对应的目标图像的y坐标

  float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
  {
    // 超出边界的像素点用const_value_st填充
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  }
  else
  {
    // 双线性插值，实现图像的放大缩小
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;

    if (y_low >= 0)
    {
      if (x_low >= 0)
        v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width)
        v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height)
    {
      if (x_low >= 0)
        v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width)
        v4 = src + y_high * src_line_size + x_high * 3;
    }

    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

  // bgr to rgb
  float t = c2;
  c2 = c0;
  c0 = t;

  // normalization
  c0 = c0 / 255.0f;
  c1 = c1 / 255.0f;
  c2 = c2 / 255.0f;

  // rgbrgbrgb to rrrgggbbb
  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}
// GPU做归一化、BGR2RGB、NHWC to NCHW
void cuda_pure_preprocess(
    uint8_t *src, float *dst, int dst_width, int dst_height)
{

  int img_size = dst_width * dst_height * 3;
  CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

  // 一个线程处理一个像素点，一共需要 dst_height * dst_width 个线程
  int jobs = dst_height * dst_width;
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);

  preprocess_kernel<<<blocks, threads>>>(
      img_buffer_device, dst, dst_width, dst_height, jobs);
}

void cuda_preprocess(
    uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height)
{

  int img_size = src_width * src_height * 3;
  CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

  // 计算变换矩阵
  AffineMatrix s2d, d2s;
  float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

  s2d.value[0] = scale;
  s2d.value[1] = 0;
  s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
  s2d.value[3] = 0;
  s2d.value[4] = scale;
  s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

  cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
  cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
  cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

  memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

  // 一个线程处理一个像素点，一共需要 dst_height * dst_width 个线程
  int jobs = dst_height * dst_width;
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);
  // 调用kernel函数

  warpaffine_kernel<<<blocks, threads>>>(
      img_buffer_device, src_width * 3, src_width,
      src_height, dst, dst_width,
      dst_height, 128, d2s, jobs);
}

void cuda_batch_preprocess(std::vector<cv::Mat> &img_batch,
                           float *dst, int dst_width, int dst_height)
{
  int dst_size = dst_width * dst_height * 3;
  for (size_t i = 0; i < img_batch.size(); i++)
  {
    cuda_preprocess(img_batch[i].ptr(), img_batch[i].cols, img_batch[i].rows, &dst[dst_size * i], dst_width, dst_height);
  }
}

void cuda_preprocess_init(int max_image_size)
{
  // prepare input data in device memory
  CUDA_CHECK(cudaMalloc((void **)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy()
{
  CUDA_CHECK(cudaFree(img_buffer_device));
}

// 使用cuda预处理所有步骤
void process_input_gpu(cv::Mat &src, float *input_device_buffer)
{
  cuda_preprocess(src.ptr(), src.cols, src.rows, input_device_buffer, kInputW, kInputH);
}

// 对输入图片进行letterbox处理，即保持原图比例，将图片放在一个正方形的画布中，多余的部分用黑色填充
// 使用cv::warpAffine，将图片进行仿射变换，将图片放在画布中
// 这里使用inline关键字，表示该函数在编译时会被直接替换到调用处，不会生成函数调用，提高效率
inline cv::Mat letterbox(cv::Mat &src)
{
  float scale = std::min(kInputH / (float)src.rows, kInputW / (float)src.cols);

  int offsetx = (kInputW - src.cols * scale) / 2;
  int offsety = (kInputH - src.rows * scale) / 2;

  cv::Point2f srcTri[3]; // 计算原图的三个点：左上角、右上角、左下角
  srcTri[0] = cv::Point2f(0.f, 0.f);
  srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
  srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);
  cv::Point2f dstTri[3]; // 计算目标图的三个点：左上角、右上角、左下角
  dstTri[0] = cv::Point2f(offsetx, offsety);
  dstTri[1] = cv::Point2f(src.cols * scale - 1.f + offsetx, offsety);
  dstTri[2] = cv::Point2f(offsetx, src.rows * scale - 1.f + offsety);
  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);       // 计算仿射变换矩阵
  cv::Mat warp_dst = cv::Mat::zeros(kInputH, kInputW, src.type()); // 创建目标图
  cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());        // 进行仿射变换
  return warp_dst;
}

// 使用CPU做letterbox，GPU做归一化、BGR2RGB、NHWC to NCHW
void process_input_cv_affine(cv::Mat &src, float *input_device_buffer)
{
  auto warp_dst = letterbox(src);
  cuda_pure_preprocess(warp_dst.ptr(), input_device_buffer, kInputW, kInputH);
}

// 使用CPU做letterbox、归一化、BGR2RGB、NHWC to NCHW
void process_input_cpu(cv::Mat &src, float *input_device_buffer)
{

  auto warp_dst = letterbox(src);                      // letterbox
  warp_dst.convertTo(warp_dst, CV_32FC3, 1.0 / 255.0); // normalization
  cv::cvtColor(warp_dst, warp_dst, cv::COLOR_BGR2RGB); // BGR2RGB
  // NHWC to NCHW：rgbrgbrgb to rrrgggbbb：
  std::vector<cv::Mat> warp_dst_nchw_channels;
  cv::split(warp_dst, warp_dst_nchw_channels); // 将输入图像分解成三个单通道图像：rrrrr、ggggg、bbbbb

  // 将每个单通道图像进行reshape操作，变为1x1xHxW的四维矩阵
  for (auto &img : warp_dst_nchw_channels)
  {
    // reshape参数分别是cn：通道数，rows：行数
    // 类似[[r,r,r,r,r]]或[[g,g,g,g,g]]或[[b,b,b,b,b]]，每个有width * height个元素
    img = img.reshape(1, 1);
  }
  // 将三个单通道图像拼接成一个三通道图像，即rrrrr、ggggg、bbbbb拼接成rrrgggbbb
  cv::Mat warp_dst_nchw;
  cv::hconcat(warp_dst_nchw_channels, warp_dst_nchw);
  // 将处理后的图片数据拷贝到GPU
  CUDA_CHECK(cudaMemcpy(input_device_buffer, warp_dst_nchw.ptr(), kInputH * kInputW * 3 * sizeof(float), cudaMemcpyHostToDevice));
}