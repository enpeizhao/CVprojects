/*
multi thread runtime testj
thread 1: read video stream
thread 2: inference
thread 3: postprocess
thread 4: sreamer
*/
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include "utils/types.h"
#include "streamer/streamer.hpp"

#include <fstream>
#include "task/border_cross.h"
#include "task/gather.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
// 用于测试
#define PRINT_STEP_TIME 1
#define PRINT_ALL_TIME 1

// 定义数据结构
struct bufferItem
{
    cv::Mat frame;                // 原始图像
    std::vector<Detection> bboxs; // 检测结果
};
// 缓存大小
const int BUFFER_SIZE = 10;

// 每个阶段需要传递的缓存
std::queue<cv::Mat> stage_1_frame;
std::queue<bufferItem> stage_2_buffer;
std::queue<cv::Mat> stage_3_frame;

// 每个阶段的互斥锁
std::mutex stage_1_mutex;
std::mutex stage_2_mutex;
std::mutex stage_3_mutex;

// 每个阶段的not_full条件变量
std::condition_variable stage_1_not_full;
std::condition_variable stage_2_not_full;
std::condition_variable stage_3_not_full;

// 每个阶段的not_empty条件变量
std::condition_variable stage_1_not_empty;
std::condition_variable stage_2_not_empty;
std::condition_variable stage_3_not_empty;

class PersonApp
{
public:
    // destructor
    ~PersonApp()
    {
        std::cout << "PersonApp destructor" << std::endl;
    }
    // constructor
    PersonApp(const std::string &engine_file, const std::string &input_video_path, int mode, float dist_threshold, int do_stream, int bitrate)
        : mode{mode}, dist_threshold{dist_threshold}, do_stream{do_stream}, bitrate{bitrate}
    {
        // ========= 1. 创建推理运行时runtime =========
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
        // ======== 2. 反序列化生成engine =========
        // 加载模型文件
        auto plan = load_engine_file(engine_file);
        // 反序列化生成engine
        mEngine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
        // ======== 3. 创建执行上下文context =========
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine_->createExecutionContext());

        // 如果input_video_path是rtsp，则读取rtsp流
        if (input_video_path == "rtsp")
        {
            auto rtsp = "rtsp://192.168.1.241:8556/live1.sdp";
            // auto rtsp = "rtsp://localhost:8554/live1.sdp";
            std::cout << "当前使用的是RTSP流" << std::endl;
            cap = cv::VideoCapture(rtsp, cv::CAP_FFMPEG);
        }
        else
        {
            std::cout << "当前使用的是视频文件" << std::endl;
            cap = cv::VideoCapture(input_video_path);
        }
        // 获取画面尺寸
        frameSize_ = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        // 获取帧率
        video_fps_ = cap.get(cv::CAP_PROP_FPS);
        std::cout << "width: " << frameSize_.width << " height: " << frameSize_.height << " fps: " << video_fps_ << std::endl;
    };

    // read frame
    void readFrame()
    {
        std::cout << "线程1启动" << std::endl;

        // product frame for inference
        cv::Mat frame;
        while (cap.isOpened())
        {
            // step1 start
            auto start_1 = std::chrono::high_resolution_clock::now();
            cap >> frame;
            if (frame.empty())
            {
                std::cout << "文件处理完毕" << std::endl;
                file_processed_done = true;
                break;
            }
            // step1 end
            auto end_1 = std::chrono::high_resolution_clock::now();
            auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count() / 1000.f;
#if PRINT_STEP_TIME
            std::cout << "step1: " << elapsed_1 << "ms"
                      << ", fps: " << 1000.f / elapsed_1 << std::endl;
#endif
            // 互斥锁
            std::unique_lock<std::mutex> lock(stage_1_mutex);
            // 如果缓存满了，就等待
            stage_1_not_full.wait(lock, []
                                  { return stage_1_frame.size() < BUFFER_SIZE; });
            // 增加一个元素
            stage_1_frame.push(frame);
            // 通知下一个线程可以开始了
            stage_1_not_empty.notify_one();
        }
    }
    // inference
    void inference()
    {
        std::cout << "线程2启动" << std::endl;
        // ========== 4. 创建输入输出缓冲区 =========
        samplesCommon::BufferManager buffers(mEngine_);

        cv::Mat frame;

        int img_size = frameSize_.width * frameSize_.height;
        cuda_preprocess_init(img_size); // 申请cuda内存
        while (true)
        {

            // 检查是否退出
            if (file_processed_done && stage_1_frame.empty())
            {
                std::cout << "线程2退出" << std::endl;
                break;
            }
            // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_1_mutex);
                // 如果缓存为空，就等待
                stage_1_not_empty.wait(lock, []
                                       { return !stage_1_frame.empty(); });
                // 取出一个元素
                frame = stage_1_frame.front();
                stage_1_frame.pop();
                // 通知上一个线程可以开始了
                stage_1_not_full.notify_one();
            }

            // step2 start
            auto start_2 = std::chrono::high_resolution_clock::now();

            // 选择预处理方式
            if (mode == 0)
            {
                // 使用CPU做letterbox、归一化、BGR2RGB、NHWC to NCHW
                process_input_cpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
            }
            else if (mode == 1)
            {
                // 使用CPU做letterbox，GPU做归一化、BGR2RGB、NHWC to NCHW
                process_input_cv_affine(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
            }
            else if (mode == 2)
            {
                // 使用cuda预处理所有步骤
                process_input_gpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
            }

            // ========== 5. 执行推理 =========
            context_->executeV2(buffers.getDeviceBindings().data());
            // 拷贝回host
            buffers.copyOutputToHost();

            // 从buffer manager中获取模型输出
            int32_t *num_det = (int32_t *)buffers.getHostBuffer(kOutNumDet); // 检测到的目标个数
            int32_t *cls = (int32_t *)buffers.getHostBuffer(kOutDetCls);     // 检测到的目标类别
            float *conf = (float *)buffers.getHostBuffer(kOutDetScores);     // 检测到的目标置信度
            float *bbox = (float *)buffers.getHostBuffer(kOutDetBBoxes);     // 检测到的目标框
            // 执行nms（非极大值抑制），得到最后的检测框
            std::vector<Detection> bboxs;
            yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);

            bufferItem item;
            // copy frmae to item
            item.frame = frame.clone();
            // item.frame = frame;
            item.bboxs = bboxs;

            // step2 end
            auto end_2 = std::chrono::high_resolution_clock::now();
            elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step2: " << elapsed_2 << "ms"
                      << ", fps: " << 1000.f / elapsed_2 << std::endl;
#endif
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock2(stage_2_mutex);
                // not full
                stage_2_not_full.wait(lock2, []
                                      { return stage_2_buffer.size() < BUFFER_SIZE; });
                //   push
                stage_2_buffer.push(item);
                // not empty
                stage_2_not_empty.notify_one();
            }
        }
    }
    // postprocess
    void postprocess()
    {

        std::cout << "线程3启动" << std::endl;

        Polygon g_ploygon;
        // 检查多边形定点配置文件是否存在
        std::ifstream infile("./config/polygon.txt");
        if (infile)
        {
            std::cout << "检测到多边形顶点配置文件，从文件中恢复多边形定点" << std::endl;
            readPoints("./config/polygon.txt", g_ploygon, frameSize_.width, frameSize_.height);
        }
        else
        {
            std::cout << "未检测到多边形顶点配置文件" << std::endl;
        }
        // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
        // cv::VideoWriter writer("./output/record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), video_fps_, frameSize_);

        bufferItem item;
        cv::Mat frame;
        while (true)
        {
            // 检查是否退出
            if (file_processed_done && stage_2_buffer.empty())
            {
                std::cout << "线程3退出" << std::endl;
                break;
            }

            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_2_mutex);
                // 如果缓存为空，就等待
                stage_2_not_empty.wait(lock, []
                                       { return !stage_2_buffer.empty(); });
                // 取出一个元素
                item = stage_2_buffer.front();
                frame = item.frame.clone();
                stage_2_buffer.pop();
                // 通知上一个线程可以开始了
                stage_2_not_full.notify_one();
            }

            // step3 start
            auto start_3 = std::chrono::high_resolution_clock::now();

            // 记录所有的检测框中心点
            std::vector<Point> all_points;
            // 遍历检测结果
            for (size_t j = 0; j < item.bboxs.size(); j++)
            {

                cv::Rect r = get_rect(frame, item.bboxs[j].bbox);
                // 获取检测框中心点
                Point p_center = {r.x + int(r.width / 2), r.y + int(r.height / 2)};
                // 筛选labelid为0的检测框
                if (item.bboxs[j].class_id == 0)
                {
                    all_points.push_back(p_center);
                }
                // 检测框中心点是否在多边形内，在则画红框，不在则画绿框
                if (isInside(g_ploygon, p_center))
                {
                    cv::rectangle(frame, r, cv::Scalar(0x00, 0x00, 0xFF), 2);
                }
                else
                {
                    cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                }
                // 绘制labelid
                // cv::putText(frame, std::to_string((int)bboxs[j].class_id), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
            }
            // 获取聚集点
            auto gather_points = gather_rule(all_points, dist_threshold);
            for (size_t i = 0; i < gather_points.size(); i++)
            {

                if (gather_points[i].size() < 3)
                    continue;
                for (size_t j = 0; j < gather_points[i].size(); j++)
                {
                    // std::cout << "聚集点：" << gather_points[i][j].x << "," << gather_points[i][j].y << std::endl;
                    // 绘制聚集点
                    blender_overlay(gather_points[i][j].x, gather_points[i][j].y, 80, frame, 0.3, frameSize_.height, frameSize_.width);
                }
            }

            // 绘制多边形
            std::vector<cv::Point> polygon;
            for (size_t i = 0; i < g_ploygon.size(); i++)
            {
                polygon.push_back(cv::Point(g_ploygon[i].x, g_ploygon[i].y));
            }
            cv::polylines(frame, polygon, true, cv::Scalar(0, 0, 255), 2);

            // step3 end
            auto end_3 = std::chrono::high_resolution_clock::now();
            auto elapsed_3 = std::chrono::duration_cast<std::chrono::microseconds>(end_3 - start_3).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step3 time: " << elapsed_3 << "ms"
                      << ", fps: " << 1000.f / elapsed_3 << std::endl;
#endif

            // 绘制时间和帧率
            std::string time_str = "time: " + std::to_string(elapsed_2);
            std::string fps_str = "fps: " + std::to_string(1000.f / elapsed_2);
            cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);

            // cv::imshow("frame", frame);
            // 写入视频文件
            // writer.write(frame);

            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock2(stage_3_mutex);
                // 如果缓存满了，就等待
                stage_3_not_full.wait(lock2, []
                                      { return stage_3_frame.size() < BUFFER_SIZE; });
                // 添加一个元素
                stage_3_frame.push(frame);
                // 通知下一个线程可以开始了
                stage_3_not_empty.notify_one();
            }
        }
    }
    // streamer
    void streamer()
    {
        std::cout << "线程4启动" << std::endl;
        // 实例化推流器
        streamer::Streamer streamer;
        streamer::StreamerConfig streamer_config(frameSize_.width, frameSize_.height,
                                                 frameSize_.width, frameSize_.height,
                                                 video_fps_, bitrate, "main", "rtmp://localhost/live/mystream");
        streamer.init(streamer_config);

        // 记录开始时间
        auto start_all = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        cv::Mat frame;
        while (true)
        {
            // 检查是否退出
            if (file_processed_done && stage_3_frame.empty())
            {
                std::cout << "线程4退出" << std::endl;
                break;
            }
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_3_mutex);
                // 如果缓存为空，就等待
                stage_3_not_empty.wait(lock, []
                                       { return !stage_3_frame.empty(); });
                // 取出一个元素
                frame = stage_3_frame.front();
                stage_3_frame.pop();
                // 通知上一个线程可以开始了
                stage_3_not_full.notify_one();
            }

            // step4 start
            auto start_4 = std::chrono::high_resolution_clock::now();
            // 推流
            streamer.stream_frame(frame.data);
            // step4 end
            auto end_4 = std::chrono::high_resolution_clock::now();
            auto elapsed_4 = std::chrono::duration_cast<std::chrono::microseconds>(end_4 - start_4).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step4 time: " << elapsed_4 << "ms"
                      << ", fps: " << 1000.f / elapsed_4 << std::endl;
#endif

#if PRINT_ALL_TIME
            // 算法2：计算超过 1s 一共处理了多少张图片
            frame_count++;
            // all end
            auto end_all = std::chrono::high_resolution_clock::now();
            auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
            // 每隔1秒打印一次
            if (elapsed_all_2 > 1000)
            {
                std::cout << "method 2 all steps time(ms): " << elapsed_all_2 << ", fps: " << frame_count / (elapsed_all_2 / 1000.0f) << ",frame count: " << frame_count << std::endl;
                frame_count = 0;
                start_all = std::chrono::high_resolution_clock::now();
            }
#endif
        }
    }

private:
    std::string input_video_path = "rtsp"; // 输入源，文件或者rtsp流
    int mode = 0;                          // 预处理方式, 0: cpu, 1: cpu + gpu, 2: gpu
    float dist_threshold;                  // 聚类距离阈值
    int do_stream = 0;                     // 是否推流
    int bitrate = 4000000;                 // 推流码率
    cv::VideoCapture cap;                  // 视频流
    float video_fps_;                      // 视频帧率
    cv::Size frameSize_;                   // 视频帧大小

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_;       // 模型引擎
    std::shared_ptr<nvinfer1::IExecutionContext> context_; // 执行上下文

    bool file_processed_done = false; // 文件处理完成标志
    float elapsed_2 = 0;              // inference time

    // 加载模型文件
    std::vector<unsigned char> load_engine_file(const std::string &file_name)
    {
        std::vector<unsigned char> engine_data;
        std::ifstream engine_file(file_name, std::ios::binary);
        assert(engine_file.is_open() && "Unable to load engine file.");
        engine_file.seekg(0, engine_file.end);
        int length = engine_file.tellg();
        engine_data.resize(length);
        engine_file.seekg(0, engine_file.beg);
        engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
        return engine_data;
    }
    // 从文件中恢复多边形定点
    void readPoints(std::string filename, Polygon &g_ploygon, int width, int height)
    {
        std::ifstream file(filename);
        std::string str;
        while (std::getline(file, str))
        {
            std::stringstream ss(str);
            std::string x, y;
            std::getline(ss, x, ',');
            std::getline(ss, y, ',');

            // recover to original size
            x = std::to_string(std::stof(x) * width);
            y = std::to_string(std::stof(y) * height);

            g_ploygon.push_back({std::stoi(x), std::stoi(y)});
        }
    }
    // 混合图像
    void blender_overlay(int x, int y, int radius, cv::Mat &image, float alpha, int height, int width)
    {
        // initial
        int rect_l = x - radius;
        int rect_t = y - radius;
        int rect_w = radius * 2;
        int rect_h = radius * 2;

        int point_x = radius;
        int point_y = radius;

        // check if out of range
        if (x + radius > width)
        {
            rect_w = radius + (width - x);
        }
        if (y + radius > height)
        {
            rect_h = radius + (height - y);
        }
        if (x - radius < 0)
        {
            rect_l = 0;
            rect_w = radius + x;
            point_x = x;
        }
        if (y - radius < 0)
        {
            rect_t = 0;
            rect_h = radius + y;
            point_y = y;
        }
        // get roi
        cv::Mat roi = image(cv::Rect(rect_l, rect_t, rect_w, rect_h));
        cv::Mat color;
        roi.copyTo(color);
        // draw circle
        cv::circle(color, cv::Point(point_x, point_y), radius, cv::Scalar(255, 0, 255), -1);
        // blend
        cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
    }
};
int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cerr << "用法: " << argv[0] << " <engine_file> <input_path_path> <preprocess_mode> <dist_threshold> <stream> <bitrate> " << std::endl;
        return -1;
    }

    auto engine_file = argv[1];                // 模型文件
    std::string input_video_path = argv[2];    // 输入视频文件
    auto mode = std::stoi(argv[3]);            // 预处理模式
    float dist_threshold = std::stof(argv[4]); // 距离阈值
    auto do_stream = std::stoi(argv[5]);       // 是否推流
    auto bitrate = std::stoi(argv[6]);         // 码率
    // initialize class
    auto app = PersonApp(engine_file, input_video_path, mode, dist_threshold, do_stream, bitrate);

    // thread 1 : read video stream
    std::thread T_readFrame(&PersonApp::readFrame, &app);
    // thread 2: inference
    std::thread T_inference(&PersonApp::inference, &app);
    // thread 3: postprocess
    std::thread T_postprocess(&PersonApp::postprocess, &app);
    // thread 4: streamer
    std::thread T_streamer(&PersonApp::streamer, &app);

    // 等待线程结束
    T_readFrame.join();
    T_inference.join();
    T_postprocess.join();
    T_streamer.join();

    return 0;
}
