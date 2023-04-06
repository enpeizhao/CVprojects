/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "yoloPlugins.h"
#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <memory>
#define NANCHORS 3
#define NFEATURES 3

namespace
{
    template <typename T>
    void write(char *&buffer, const T &val)
    {
        *reinterpret_cast<T *>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char *&buffer, T &val)
    {
        val = *reinterpret_cast<const T *>(buffer);
        buffer += sizeof(T);
    }
}

nvinfer1::PluginFieldCollection YoloLayerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> YoloLayerPluginCreator::mPluginAttributes;

YoloLayerPluginCreator::YoloLayerPluginCreator() noexcept
{
    mPluginAttributes.emplace_back(nvinfer1::PluginField("max_stride", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(nvinfer1::PluginField("net_height", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("num_classes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("anchors", nullptr, nvinfer1::PluginFieldType::kFLOAT32, NFEATURES * NANCHORS * 2));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("prenms_score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

nvinfer1::IPluginV2DynamicExt *YoloLayerPluginCreator::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept
{
    const nvinfer1::PluginField *fields = fc->fields;
    int max_stride = 32;
    int num_classes = 80;
    std::vector<float> anchors;
    float score_threshold = 0.0;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char *attrName = fields[i].name;
        if (!strcmp(attrName, "max_stride"))
        {
            assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            max_stride = *(static_cast<const int *>(fields[i].data));
        }
        if (!strcmp(attrName, "num_classes"))
        {
            assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            num_classes = *(static_cast<const int *>(fields[i].data));
        }
        if (!strcmp(attrName, "anchors"))
        {
            assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            const auto anchors_ptr = static_cast<const float *>(fields[i].data);
            anchors.assign(anchors_ptr, anchors_ptr + NFEATURES * NANCHORS * 2);
        }
        if (!strcmp(attrName, "prenms_score_threshold"))
        {
            assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            score_threshold = *(static_cast<const float *>(fields[i].data));
        }
    }
    return new YoloLayer(max_stride, num_classes, anchors, score_threshold);
}

cudaError_t cudaYoloLayer_nc(
    const void *input, void *num_detections, void *detection_boxes, void *detection_scores, void *detection_classes,
    const uint &batchSize, uint64_t &inputSize, uint64_t &outputSize, const float &scoreThreshold, const uint &netWidth,
    const uint &netHeight, const uint &gridSizeX, const uint &gridSizeY, const uint &numOutputClasses, const uint &numBBoxes,
    const float &scaleXY, const void *anchors, cudaStream_t stream);

YoloLayer::YoloLayer(const void *data, size_t length)
{
    const char *d = static_cast<const char *>(data);

    read(d, m_NetWidth);
    read(d, m_NetHeight);
    read(d, m_MaxStride);
    read(d, m_NumClasses);
    read(d, m_ScoreThreshold);
    read(d, m_OutputSize);

    m_Anchors.resize(NFEATURES * NANCHORS * 2);
    for (uint i = 0; i < m_Anchors.size(); i++)
    {
        read(d, m_Anchors[i]);
    }

    for (uint i = 0; i < NFEATURES; i++)
    {
        int height;
        int width;
        read(d, height);
        read(d, width);
        m_FeatureSpatialSize.push_back(nvinfer1::DimsHW(height, width));
    }
};

YoloLayer::YoloLayer(
    const uint &maxStride, const uint &numClasses,
    const std::vector<float> &anchors, const float &scoreThreshold) : m_MaxStride(maxStride),
                                                                      m_NumClasses(numClasses),
                                                                      m_Anchors(anchors),
                                                                      m_ScoreThreshold(scoreThreshold){

                                                                      };

nvinfer1::DimsExprs
YoloLayer::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept
{
    assert(outputIndex < 4);

    nvinfer1::DimsExprs out_dim;
    const nvinfer1::IDimensionExpr *batch_size = inputs[0].d[0];

    const nvinfer1::IDimensionExpr *output_num_boxes = exprBuilder.constant(0);
    // input feature [batch_size, (nc+5) * nanchor, height, width]
    for (int32_t i = 0; i < NFEATURES; i++)
    {
        output_num_boxes = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *output_num_boxes,
                                                 *exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                                                        *inputs[i].d[2], *inputs[i].d[3]));
    }

    output_num_boxes = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                             *output_num_boxes, *exprBuilder.constant(NANCHORS));

    // num_detections [batch_size, 1]
    // detection_boxes [batch_size, numboxes, 4]
    // detection_scores [batch_size, numboxes]
    // detection_classes [batch_size, numboxes]
    if (outputIndex == 0)
    {
        out_dim.nbDims = 2;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = exprBuilder.constant(1);
    }
    else if (outputIndex == 1)
    {
        out_dim.nbDims = 3;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
        out_dim.d[2] = exprBuilder.constant(4);
    }
    else
    {
        out_dim.nbDims = 2;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
    }
    return out_dim;
}

nvinfer1::DataType
YoloLayer::getOutputDataType(int index, const nvinfer1::DataType *inputType, int nbInputs) const noexcept
{
    // num_detection and classes
    if (index == 0 || index == 3)
    {
        return nvinfer1::DataType::kINT32;
    }
    // All others should use the same datatype as the input
    return inputType[0];
}

bool YoloLayer::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept
{
    if (inOut[pos].format != nvinfer1::PluginFormat::kLINEAR)
    {
        return false;
    }

    const int posOut = pos - nbInputs;
    // num_detection and classes
    if (posOut == 0 || posOut == 3)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
    }

    // all other inputs/outputs: fp32 or fp16
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[0].type == inOut[pos].type);
}

void YoloLayer::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept
{
    assert(nbInputs == NFEATURES);
    // input feature [batch_size, (nc+5) * nanchor, height, width]
    m_OutputSize = 0;
    m_FeatureSpatialSize.clear();
    for (int i = 0; i < NFEATURES; i++)
    {
        m_FeatureSpatialSize.push_back(nvinfer1::DimsHW(in[i].desc.dims.d[2], in[i].desc.dims.d[3]));
        m_OutputSize += in[i].desc.dims.d[2] * in[i].desc.dims.d[3] * NANCHORS;
    }
    // Compute the network input by last feature map and max stride
    m_NetHeight = in[NFEATURES - 1].desc.dims.d[2] * m_MaxStride;
    m_NetWidth = in[NFEATURES - 1].desc.dims.d[3] * m_MaxStride;
}

int YoloLayer::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                       void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    const int batchSize = inputDesc[0].dims.d[0];
    void *num_detections = outputs[0];
    void *detection_boxes = outputs[1];
    void *detection_scores = outputs[2];
    void *detection_classes = outputs[3];
    CUDA_CHECK(cudaMemsetAsync((int *)num_detections, 0, sizeof(int) * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync((float *)detection_boxes, 0, sizeof(float) * m_OutputSize * 4 * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync((float *)detection_scores, 0, sizeof(float) * m_OutputSize * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync((int *)detection_classes, 0, sizeof(int) * m_OutputSize * batchSize, stream));

    uint yoloTensorsSize = NFEATURES;
    for (uint i = 0; i < yoloTensorsSize; ++i)
    {
        // TensorInfo& curYoloTensor = m_YoloTensors.at(i);
        const nvinfer1::DimsHW &gridSize = m_FeatureSpatialSize[i];

        uint numBBoxes = NANCHORS;
        float scaleXY = 2.0;
        uint gridSizeX = gridSize.w();
        uint gridSizeY = gridSize.h();
        std::vector<float> anchors(m_Anchors.begin() + i * NANCHORS * 2, m_Anchors.begin() + (i + 1) * NANCHORS * 2);

        void *v_anchors;
        if (anchors.size() > 0)
        {
            float *f_anchors = anchors.data();
            CUDA_CHECK(cudaMalloc(&v_anchors, sizeof(float) * anchors.size()));
            CUDA_CHECK(cudaMemcpyAsync(v_anchors, f_anchors, sizeof(float) * anchors.size(), cudaMemcpyHostToDevice,
                                       stream));
        }

        uint64_t inputSize = gridSizeX * gridSizeY * (numBBoxes * (4 + 1 + m_NumClasses));

        CUDA_CHECK(cudaYoloLayer_nc(
            inputs[i], num_detections, detection_boxes, detection_scores, detection_classes, batchSize,
            inputSize, m_OutputSize, m_ScoreThreshold, m_NetWidth, m_NetHeight, gridSizeX, gridSizeY,
            m_NumClasses, numBBoxes, scaleXY, v_anchors, stream));

        if (anchors.size() > 0)
        {
            CUDA_CHECK(cudaFree(v_anchors));
        }
    }

    return 0;
}

size_t YoloLayer::getSerializationSize() const noexcept
{
    size_t totalSize = 0;

    totalSize += sizeof(m_NetWidth);
    totalSize += sizeof(m_NetHeight);
    totalSize += sizeof(m_MaxStride);
    totalSize += sizeof(m_NumClasses);
    totalSize += sizeof(m_ScoreThreshold);
    totalSize += sizeof(m_OutputSize);

    // anchors
    totalSize += m_Anchors.size() * sizeof(m_Anchors[0]);

    // feature size
    totalSize += m_FeatureSpatialSize.size() * 2 * sizeof(m_FeatureSpatialSize[0].h());

    return totalSize;
}

void YoloLayer::serialize(void *buffer) const noexcept
{
    char *d = static_cast<char *>(buffer);

    write(d, m_NetWidth);
    write(d, m_NetHeight);
    write(d, m_MaxStride);
    write(d, m_NumClasses);
    write(d, m_ScoreThreshold);
    write(d, m_OutputSize);

    // write anchors:
    for (int i = 0; i < m_Anchors.size(); i++)
    {
        write(d, m_Anchors[i]);
    }

    // write feature size:
    uint yoloTensorsSize = m_FeatureSpatialSize.size();
    for (uint i = 0; i < yoloTensorsSize; ++i)
    {
        write(d, m_FeatureSpatialSize[i].h());
        write(d, m_FeatureSpatialSize[i].w());
    }
}

nvinfer1::IPluginV2DynamicExt *YoloLayer::clone() const noexcept
{
    return new YoloLayer(
        m_MaxStride, m_NumClasses, m_Anchors, m_ScoreThreshold);
}
// 注册插件。 在实现了各个类方法后，需要调用宏对plugin进行注册。以方便TensorRT识别并找到对应的Plugin。
REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
