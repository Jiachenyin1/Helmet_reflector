#include <iostream>
#include <fstream>
#include "logging.h"
#include "mish.h"
#include "yololayer.h"
#include "utils.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

#define DEVICE 0
//#define ENGINE
#define BATCHSIZE 1


int main(int argc , char** argv)
{
    cudaSetDevice(DEVICE);
    char* trtModelStream{nullptr};
    size_t size;
#ifdef ENGINE
    IHostMemory* modelStream{nullptr};
    APItoModel(BATCHSIZE , &modelStream);
    assert(modelStream != nullptr);
    std::ofstream p("yolov4.engine" , std::ios::binary);
    if(!p)
    {
        std::cerr << "can't open plan file " << std::endl;
    }
    p.write(reinterpret_cast<char* >(modelStream->data()) , modelStream->size());
    modelStream->destroy();
    return 0;   
#endif
#ifndef ENGINE
    std::ifstream file("yolov4.engine",std::ios::binary);
    if(file.good())
    {
        file.seekg(0,file.end);
        size= file.tellg();
        file.seekg(0 , file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream , size);
        file.close();
    }
    IRuntime* runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream , size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    doInference(context , BATCHSIZE);

#endif
//    return 0;
}
