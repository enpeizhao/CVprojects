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

#ifndef __YOLO_PLUGINS__
#define __YOLO_PLUGINS__

#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

#include <vector>

#include "NvInferPlugin.h"

#define CUDA_CHECK(status)                                                                                        \
    {                                                                                                             \
        if (status != 0)                                                                                          \
        {                                                                                                         \
            std::cout << "CUDA failure: " << cudaGetErrorString(status) << " in file " << __FILE__ << " at line " \
                      << __LINE__ << std::endl;                                                                   \
            abort();                                                                                              \
        }                                                                                                         \
    }

// 定义插件版本和插件名
namespace
{
    const char *YOLOLAYER_PLUGIN_VERSION{"1"};
    const char *YOLOLAYER_PLUGIN_NAME{"YoloLayer_TRT"};
} // namespace

// 实现插件类。插件类需要继承IPluginV2DynamicExt类。这份代码中定义的插件类名是YoloLayer，其中实现了IPluginV2DynamicExt中的虚函数和一些成员变量和函数。
class YoloLayer : public nvinfer1::IPluginV2DynamicExt
{
public:
    YoloLayer(const void *data, size_t length);
    YoloLayer(
        const uint &maxStride, const uint &numClasses,
        const std::vector<float> &anchors, const float &scoreThreshold);

    // IPluginV2 methods
    const char *getPluginType() const noexcept override { return YOLOLAYER_PLUGIN_NAME; }
    const char *getPluginVersion() const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }
    int getNbOutputs() const noexcept override { return 4; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override;
    void serialize(void *buffer) const noexcept override;
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char *pluginNamespace) noexcept override
    {
        m_Namespace = pluginNamespace;
    }
    virtual const char *getPluginNamespace() const noexcept override
    {
        return m_Namespace.c_str();
    }

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputType, int nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc *inputs, int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override { return 0; };
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

private:
    std::string m_Namespace{""};
    int m_NetWidth{0};
    int m_NetHeight{0};
    int m_MaxStride{0};
    int m_NumClasses{0};
    std::vector<float> m_Anchors;
    std::vector<nvinfer1::DimsHW> m_FeatureSpatialSize;
    float m_ScoreThreshold{0};
    uint64_t m_OutputSize{0};
};

// 实现插件创建类。插件创建类需要继承IPluginCreator类，并实现其中的虚函数。这份代码中定义的插件创建类名是YoloLayerPluginCreator，其中实现了IPluginCreator中的虚函数和一些成员变量和函数。
class YoloLayerPluginCreator : public nvinfer1::IPluginCreator
{
public:
    YoloLayerPluginCreator() noexcept;

    ~YoloLayerPluginCreator() noexcept {}

    const char *getPluginName() const noexcept override { return YOLOLAYER_PLUGIN_NAME; }

    const char *getPluginVersion() const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }

    nvinfer1::IPluginV2DynamicExt *createPlugin(
        const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

    nvinfer1::IPluginV2DynamicExt *deserializePlugin(
        const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        std::cout << "Deserialize yoloLayer plugin: " << name << std::endl;
        return new YoloLayer(serialData, serialLength);
    }

    void setPluginNamespace(const char *libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char *getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }
    const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override
    {
        return &mFC;
    }

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // __YOLO_PLUGINS__
