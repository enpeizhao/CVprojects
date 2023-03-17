#include "NvInfer.h"
#include "NvOnnxParser.h" // onnxparser头文件
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "cassert"

#include "utils/config.h"
#include "utils/preprocess.h"

// 定义校准数据读取器
// 如果要用entropy的话改为：IInt8EntropyCalibrator2
class CalibrationDataReader : public IInt8MinMaxCalibrator
{
private:
    std::string mDataDir;
    std::string mCacheFileName;
    std::vector<std::string> mFileNames;
    int mBatchSize;
    Dims mInputDims;
    int mInputCount;
    float *mDeviceBatchData{nullptr};
    int mBatchCount;
    int mImgSize;
    int mCurBatch{0};
    std::vector<char> mCalibrationCache;

public:
    // 构造函数，初始化参数
    CalibrationDataReader(const std::string &dataDir, const std::string &list, int batchSize = 1)
        : mDataDir(dataDir), mCacheFileName("weights/calibration.cache"), mBatchSize(batchSize), mImgSize(kInputH * kInputW)
    {
        mInputDims = {1, 3, kInputH, kInputW}; // 设置网络输入尺寸
        mInputCount = mBatchSize * samplesCommon::volume(mInputDims);
        cuda_preprocess_init(mImgSize);                                       // 初始化预处理内存
        cudaMalloc(&mDeviceBatchData, kInputH * kInputW * 3 * sizeof(float)); // 分配预处理内存
        // 加载校准数据集文件列表
        std::ifstream infile(list);
        std::string line;
        while (std::getline(infile, line))
        {
            sample::gLogInfo << line << std::endl;
            mFileNames.push_back(line);
        }
        mBatchCount = mFileNames.size() / mBatchSize;
        std::cout << "CalibrationDataReader: " << mFileNames.size() << " images, " << mBatchCount << " batches." << std::endl;
    }

    int32_t getBatchSize() const noexcept override
    {
        return mBatchSize;
    }
    // 用于提供一批校准数据。在该方法中，需要将当前批次的校准数据读取到内存中，并将其复制到设备内存中。然后，将数据指针传递给 TensorRT 引擎，以供后续的校准计算使用。
    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override
    {
        if (mCurBatch + 1 > mBatchCount)
        {
            return false;
        }
        int offset = kInputW * kInputH * 3 * sizeof(float);
        for (int i = 0; i < mBatchSize; i++)
        {
            int idx = mCurBatch * mBatchSize + i;
            std::string fileName = mDataDir + "/" + mFileNames[idx];
            cv::Mat img = cv::imread(fileName);
            int new_img_size = img.cols * img.rows;
            if (new_img_size > mImgSize)
            {
                mImgSize = new_img_size;
                cuda_preprocess_destroy();      // 释放之前的内存
                cuda_preprocess_init(mImgSize); // 重新分配内存
            }
            // 预处理，并将预处理后的数据复制到设备内存中
            process_input_gpu(img, mDeviceBatchData + i * offset);
        }
        for (int i = 0; i < nbBindings; i++)
        {
            if (!strcmp(names[i], kInputTensorName))
            {
                bindings[i] = mDeviceBatchData + i * offset;
            }
        }

        mCurBatch++;
        return true;
    }
    // 从缓存文件中读取校准缓存，返回一个指向缓存数据的指针，以及缓存数据的大小。如果没有缓存数据，则返回`nullptr`。
    const void *readCalibrationCache(std::size_t &length) noexcept override
    {
        mCalibrationCache.clear();

        std::ifstream input(mCacheFileName, std::ios::binary);
        input >> std::noskipws;

        if (input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();

        return length ? mCalibrationCache.data() : nullptr;
    }
    // 用于将校准缓存写入到缓存文件中。在该方法中，需要将缓存数据指针和缓存数据的大小传递给文件输出流，并将其写入到缓存文件中。
    void writeCalibrationCache(const void *cache, std::size_t length) noexcept override
    {
        std::ofstream output(mCacheFileName, std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
    }
};

// main函数
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "用法: ./build [onnx_file_path] [calib_dir] [calib_list_file]" << std::endl;
        return -1;
    }
    // 命令行获取onnx文件路径、校准数据集路径、校准数据集列表文件
    char *onnx_file_path = argv[1];
    char *calib_dir = argv[2];
    char *calib_list_file = argv[3];

    // =========== 1. 创建builder ===========
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        std::cerr << "Failed to create builder" << std::endl;
        return -1;
    }

    // ========== 2. 创建network：builder--->network ==========
    // 显性batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 调用builder的createNetworkV2方法创建network
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cout << "Failed to create network" << std::endl;
        return -1;
    }
    // 与上节课手动创建网络不同，这次使用onnxparser创建网络

    // 创建onnxparser，用于解析onnx文件
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    // 调用onnxparser的parseFromFile方法解析onnx文件
    auto parsed = parser->parseFromFile(onnx_file_path, static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        std::cout << "Failed to parse onnx file" << std::endl;
        return -1;
    }
    // 配置网络参数
    // 我们需要告诉tensorrt我们最终运行时，输入图像的范围，batch size的范围。这样tensorrt才能对应为我们进行模型构建与优化。
    auto input = network->getInput(0);                                                                             // 获取输入节点
    auto profile = builder->createOptimizationProfile();                                                           // 创建profile，用于设置输入的动态尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 640, 640}); // 设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 640, 640}); // 设置最优尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 640, 640}); // 设置最大尺寸

    // ========== 3. 创建config配置：builder--->config ==========
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "Failed to create config" << std::endl;
        return -1;
    }
    // 使用addOptimizationProfile方法添加profile，用于设置输入的动态尺寸
    config->addOptimizationProfile(profile);

    // 设置精度，不设置是FP32，设置为FP16，设置为INT8需要额外设置calibrator
    if (!builder->platformHasFastInt8())
    {
        sample::gLogInfo << "设备不支持int8." << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else
    {
        // 设置calibrator量化校准器
        auto calibrator = new CalibrationDataReader(calib_dir, calib_list_file);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
    }

    // 设置最大batchsize
    builder->setMaxBatchSize(1);
    // 设置最大工作空间（新版本的TensorRT已经废弃了setWorkspaceSize）
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

    // 创建流，用于设置profile
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return -1;
    }
    config->setProfileStream(*profileStream);

    // ========== 4. 创建engine：builder--->engine(*nework, *config) ==========
    // 使用buildSerializedNetwork方法创建engine，可直接返回序列化的engine（原来的buildEngineWithConfig方法已经废弃，需要先创建engine，再序列化）
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        std::cout << "Failed to create engine" << std::endl;
        return -1;
    }

    // ========== 5. 序列化保存engine ==========
    std::ofstream engine_file("./weights/yolov5.engine", std::ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    // ========== 6. 释放资源 ==========
    // 因为使用了智能指针，所以不需要手动释放资源

    std::cout << "Engine build success!" << std::endl;

    return 0;
}