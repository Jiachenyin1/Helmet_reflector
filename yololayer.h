//#ifndef YOLOLAYER_H
//#define YOLOLAYER_H
//#include <iostream>
//#include <string.h>
//#include <vector>
//#include "NvInfer.h"
//#include <cublas_v2.h>
//#include <cudnn.h>

//namespace YOLO
//{
//    static constexpr int INPUT_SIZE_W = 416 ;
//    static constexpr int INPUT_SIZE_H = 416 ;
//    static constexpr int NUM_CLASS = 2 ;
//    static constexpr int MAX_OUTPUT_COUNT = 1000 ;
//    static constexpr int CHECK_COUNT = 3 ;
//    static constexpr float IGNORE_THRESH = 0.1f;

//    struct YoloKernel
//    {
//        int width;
//        int height;
//        float anchors[CHECK_COUNT*2];
//    };
//    static  constexpr YoloKernel yolo1={
//        INPUT_SIZE_W / 8,
//        INPUT_SIZE_H / 8,
//        {35, 55,  47,133, 105, 91}
//    };
//    static  constexpr YoloKernel yolo2={
//        INPUT_SIZE_W / 16,
//        INPUT_SIZE_H / 16,
//        {84,209, 174,161, 133,294}
//    };
//    static  constexpr YoloKernel yolo3={
//        INPUT_SIZE_W / 32,
//        INPUT_SIZE_H / 32,
//        {318,200, 225,329, 362,355}
//    };
//    static constexpr int Location = 4 ;
//    struct alignas(float) Detection
//    {
//        float bbox[Location];
//        float det_confidence;
//        float class_id;
//        float class_confidence;
//    };
//}


//namespace nvinfer1
//{
//    class YoloLayerPlugin: public IPluginV2IOExt
//    {
//        public:
//            explicit YoloLayerPlugin();
//            YoloLayerPlugin(const void* data, size_t length);

//            ~YoloLayerPlugin();

//            int getNbOutputs() const override
//            {
//                return 1;
//            }

//            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

//            int initialize() override;

//            virtual void terminate() override {};

//            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

//            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

//            virtual size_t getSerializationSize() const override;

//            virtual void serialize(void* buffer) const override;

//            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
//                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
//            }

//            const char* getPluginType() const override;

//            const char* getPluginVersion() const override;

//            void destroy() override;

//            IPluginV2IOExt* clone() const override;

//            void setPluginNamespace(const char* pluginNamespace) override;

//            const char* getPluginNamespace() const override;

//            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

//            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

//            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

//            void attachToContext(
//                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

//            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

//            void detachFromContext() override;

//            int input_size_;
//        private:
//            void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
//            int thread_count_ = 256;
//            const char* mPluginNamespace;
//            std::vector<YOLO::YoloKernel> mYoloKernel;
//            int mClassCount;
//            int mKernelCount;
//    };

//    class YoloLayerPluginCreator : public IPluginCreator
//    {
//        public:
//            YoloLayerPluginCreator();

//            ~YoloLayerPluginCreator() override = default;

//            const char* getPluginName() const override;

//            const char* getPluginVersion() const override;

//            const PluginFieldCollection* getFieldNames() override;

//            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

//            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

//            void setPluginNamespace(const char* libNamespace) override
//            {
//                mNamespace = libNamespace;
//            }

//            const char* getPluginNamespace() const override
//            {
//                return mNamespace.c_str();
//            }

//        private:
//            std::string mNamespace;
//            static PluginFieldCollection mFC;
//    };
//};
//#endif
#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>
#include <cudnn.h>

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 4;
    static constexpr int INPUT_H = 608;
    static constexpr int INPUT_W = 608;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 8,
        INPUT_H / 8,
        { 8, 13,  17,28, 27, 52}
    };
    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {45,76, 57,119, 136,108}
    };
    static constexpr YoloKernel yolo3 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {84,177, 141,226, 280,225}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}


namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginV2IOExt
    {
        public:
            explicit YoloLayerPlugin();
            YoloLayerPlugin(const void* data, size_t length);

            ~YoloLayerPlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            int initialize() override;

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            virtual size_t getSerializationSize() const override;

            virtual void serialize(void* buffer) const override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override;

            const char* getPluginVersion() const override;

            void destroy() override;

            IPluginV2IOExt* clone() const override;

            void setPluginNamespace(const char* pluginNamespace) override;

            const char* getPluginNamespace() const override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

            void detachFromContext() override;

        private:
            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
            int mClassCount;
            int mKernelCount;
            std::vector<Yolo::YoloKernel> mYoloKernel;
            int mThreadCount = 256;
            const char* mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };



};

#endif
