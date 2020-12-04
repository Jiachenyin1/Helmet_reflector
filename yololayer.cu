//#include "yololayer.h".h"
//#include <cmath>
//#include <stdio.h>
//#include <cassert>
//#include <iostream>
//#ifndef CUDA_CHECK

//#define CUDA_CHECK(callstr)                                                                    \
//    {                                                                                          \
//        cudaError_t error_code = callstr;                                                      \
//        if (error_code != cudaSuccess) {                                                       \
//            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
//            assert(0);                                                                         \
//        }                                                                                      \
//    }

//#endif

//template <typename T>
//void write(char* &buffer , const T& val)
//{
//    *reinterpret_cast<T*>(buffer) = val;
//    buffer += sizeof(T);
//}
//template <typename T>
//void read(const char* &buffer , T& val)
//{
//    val = *reinterpret_cast<const T*>(buffer);
//    buffer += sizeof(T);
//}

//using namespace YOLO;
//namespace nvinfer1
//{
//    YoloLayerPlugin::YoloLayerPlugin()
//    {
//        mClassCount = YOLO::NUM_CLASS;
//        mYoloKernel.clear();

//        mYoloKernel.push_back(yolo1);
//        mYoloKernel.push_back(yolo2);
//        mYoloKernel.push_back(yolo3);
//        mKernelCount = mYoloKernel.size();
//    }

//    YoloLayerPlugin::~YoloLayerPlugin(){}
//    YoloLayerPlugin::YoloLayerPlugin(const void* seriaData , size_t seriaLen)
//    {
//        const char* d = reinterpret_cast<const char* >(seriaData) , *a = d;
//        read(d , mClassCount);
//        read(d , thread_count_);
//        read(d , mKernelCount);
//        mYoloKernel.resize(mKernelCount);
//        auto kernel_size = mKernelCount * sizeof(mYoloKernel);
//        memcpy( mYoloKernel.data(),d , kernel_size);
//        d += kernel_size;

//        assert(d == a + seriaLen);
//    }

//    void YoloLayerPlugin::serialize(void *buffer) const
//    {
//        char* d = static_cast<char* >(buffer) , *a = d;
//        write(d , mClassCount);
//        write(d , thread_count_);
//        write(d , mKernelCount);
//        auto kernel_size = mKernelCount * sizeof(mYoloKernel);
//        memcpy(d , mYoloKernel.data() , kernel_size);
//        d += kernel_size;

//        assert(d == a + getSerializationSize());
//    }

//    size_t YoloLayerPlugin::getSerializationSize() const
//    {
//        return sizeof(mClassCount) + sizeof(thread_count_)+ sizeof(mKernelCount)+ sizeof(YOLO::YoloKernel) * mYoloKernel.size();
//    }

//    int YoloLayerPlugin::initialize()
//    {
//        return 0;
//    }

//    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
//    {
//        int totalsize = MAX_OUTPUT_COUNT * sizeof(Detection) / sizeof(float);
//        return Dims3(totalsize+1 ,1 , 1);
//    }

//    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespce)
//    {
//        mPluginNamespace = pluginNamespce;
//    }

//    const char* YoloLayerPlugin::getPluginNamespace() const
//    {
//        return mPluginNamespace;
//    }

//    DataType YoloLayerPlugin::getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
//    {
//        return DataType::kFLOAT;
//    }

//    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const
//    {
//        return false;
//    }

//    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
//    {
//        return false ;
//    }

//    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput)
//    {

//    }
//    //一个都不能少。。。
//    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
//    {
//    }
//    void YoloLayerPlugin::detachFromContext() {}

//    const char* YoloLayerPlugin::getPluginType() const
//    {
//        return "YoloLayer_TRT";
//    }

//    const char* YoloLayerPlugin::getPluginVersion() const
//    {
//        return "1";
//    }


//    void YoloLayerPlugin::destroy()
//    {
//        delete this ;
//    }

//    IPluginV2IOExt* YoloLayerPlugin::clone() const
//    {
//        YoloLayerPlugin* p = new YoloLayerPlugin();
//        p->setPluginNamespace(mPluginNamespace);
//        return p;
//    }
//    __device__ float logist(float data){return (1./(1 + exp(-data)));}
//    //反编码 anchor
//    __global__ void Cal_Detection(const float *input, float *output, int num_elem ,
//                                 int yolowidth , int yoloheight , const float anchor[CHECK_COUNT*2] ,  int classes , int outputElem ) {
//        //即tid=blockIdx.x（当前块的ID）*blockDim.x（当前块里面的线程数量）+threadIdx.x（当前线程在块中的ID）。
//        int idx = threadIdx.x + blockDim.x * blockIdx.x;
//        if (idx >= num_elem) return;
//        int total_grid = yolowidth * yoloheight ;
//        int Anchorlen_info = 5+ classes ;
//        const float* curInput = input ;

//        for(int k = 0 ; k< 3 ; k++)
//        {
//            int class_ind = 0 ;
//            float max_class_confidence = 0.;
//            for(int i = 5 ; i < Anchorlen_info ; i++)
//            {
//                float p = logist(curInput[idx+k*Anchorlen_info*total_grid+i*total_grid]);
//                if(p>max_class_confidence)
//                {
//                    class_ind = i - 5;
//                    max_class_confidence = p;
//                }
//            }
//            float bbox_confidence = logist(curInput[idx + k* Anchorlen_info*total_grid + 4*total_grid]);
//            if(bbox_confidence<IGNORE_THRESH || max_class_confidence< IGNORE_THRESH) continue;

//            float *res_count = output;
//            int count = (int)atomicAdd(res_count, 1);
//            if (count >= MAX_OUTPUT_COUNT) return;
//            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
//            Detection* det =  (Detection*)(data);

//            int row = idx / yolowidth;
//            int col = idx % yoloheight;

//            //Location
//            det->bbox[0] = (col + logist(curInput[idx + k * Anchorlen_info * total_grid + 0 * total_grid])) * INPUT_SIZE_W / yolowidth;
//            det->bbox[1] = (row + logist(curInput[idx + k * Anchorlen_info * total_grid + 1 * total_grid])) * INPUT_SIZE_W / yoloheight;
//            det->bbox[2] = exp(curInput[idx + k * Anchorlen_info * total_grid + 2 * total_grid]) * anchor[2*k];
//            det->bbox[3] = exp(curInput[idx + k * Anchorlen_info * total_grid + 3 * total_grid]) * anchor[2*k + 1];
//            det->det_confidence = bbox_confidence;
//            det->class_id = class_ind;
//            det->class_confidence = max_class_confidence;

//        }
//    }

//    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
//        void* devAnchor;
//        size_t AnchorLen = sizeof(float) * CHECK_COUNT * 2;
//        CUDA_CHECK(cudaMalloc(&devAnchor , AnchorLen));

//        int outputElem = 1 + MAX_OUTPUT_COUNT * sizeof(Detection) / sizeof(float);
//        for(size_t i = 0 ; i < batchSize ; i++)
//        {
//            CUDA_CHECK(cudaMemset(output + i*outputElem , 0 , sizeof(float) ));
//        }
//        int num_thread ;
//        for(size_t i = 0 ; i < mYoloKernel.size() ; i++)
//        {
//            const auto &yolo = mYoloKernel[i];
//            num_thread = yolo.height * yolo.height * batchSize;
//            if(num_thread < thread_count_)
//                thread_count_ = num_thread;
//            CUDA_CHECK(cudaMemcpy(devAnchor , yolo.anchors , AnchorLen , cudaMemcpyHostToDevice));
//            //为了计算得到下一个最小能满足要求的整数结果 ， N需要加上 block_size(thread-1），再除以 thread ， 基本上属于向上取证的操作
//            int grid_size = (num_thread + thread_count_ - 1) / thread_count_;
//            Cal_Detection<<<grid_size, thread_count_>>>(inputs[i], output, num_thread , yolo.width , yolo.height , (float *)devAnchor ,mClassCount , outputElem);
//        }

//    }
//    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
//    {
//        //assert(batchSize == 1);
//        //GPU
//        //CUDA_CHECK(cudaStreamSynchronize(stream));
//        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
//        return 0;
//    }

//    PluginFieldCollection YoloLayerPluginCreator::mFC{};
//    YoloLayerPluginCreator::YoloLayerPluginCreator()
//    {
//        mFC.nbFields = 0;
//        mFC.fields = nullptr;
//    }

//    const char* YoloLayerPluginCreator::getPluginName() const
//    {
//        return "YoloLayer_TRT";
//    }
//    const char* YoloLayerPluginCreator::getPluginVersion() const
//    {
//        return "1";
//    }
//    const PluginFieldCollection* YoloLayerPluginCreator::getFieldNames()
//    {
//        return &mFC;
//    }

//    IPluginV2IOExt* YoloLayerPluginCreator::createPlugin(const char* name , const PluginFieldCollection* fc)
//    {
//        YoloLayerPlugin* obj  =  new YoloLayerPlugin();
//        obj->setPluginNamespace(name);
//        return obj;
//    }
//    IPluginV2IOExt* YoloLayerPluginCreator::deserializePlugin(const char* name ,const void* seriaData , size_t seriaLen)
//    {
//        YoloLayerPlugin* obj  =  new YoloLayerPlugin(seriaData , seriaLen);
//        obj->setPluginNamespace(mNamespace.c_str());
//        return obj;
//    }
//}
#include "yololayer.h"

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin()
    {
        mClassCount = CLASS_NUM;
        mYoloKernel.clear();
        mYoloKernel.push_back(yolo1);
        mYoloKernel.push_back(yolo2);
        mYoloKernel.push_back(yolo3);

        mKernelCount = mYoloKernel.size();
    }

    YoloLayerPlugin::~YoloLayerPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(mYoloKernel.data(),d,kernelSize);
        d += kernelSize;

        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(),kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(Yolo::YoloKernel) * mYoloKernel.size();
    }

    int YoloLayerPlugin::initialize()
    {
        return 0;
    }

    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() {}

    const char* YoloLayerPlugin::getPluginType() const
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const
    {
        return "1";
    }

    void YoloLayerPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPlugin::clone() const
    {
        YoloLayerPlugin *p = new YoloLayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1./(1. + exp(-data)); };

    __global__ void CalDetection(const float *input, float *output,int noElements,
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i)
            {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob)
                {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

            float *res_count = output + bnIdx*outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;
            //sizeof(float)
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location 归一化
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
            det->bbox[2] = exp(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = exp(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

        int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CUDA_CHECK(cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
            CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, yolo.width, yolo.height, (float *)devAnchor, mClassCount ,outputElem);
        }

        CUDA_CHECK(cudaFree(devAnchor));
    }


    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);

        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const
    {
            return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
