#ifndef _UTILS_H
#define _UTULS_H
#include <fstream>
#include <iostream>
#include <sstream>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yololayer.h"
#include "mish.h"
#include <map>
#include <string.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <chrono>

static Logger logger ;
using namespace nvinfer1;
REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

const char* INPUT_NAME_BLOB = "data";
const char* OUTPUT_NAME_BLOB = "prob";
const int DETECTION_SIZE =(sizeof(Yolo::Detection) / sizeof(float)) ;
const float NMS_THRESH = 0.1;
static constexpr int OUTPUT_SIZE = (sizeof(Yolo::Detection) / sizeof(float)) * Yolo::MAX_OUTPUT_BBOX_COUNT + 1;
std::string path = "yolov4.wts";
std::string video = "VID_20190919_155623.mp4";
//#define FP_16;
#define BBOX_CONF_THRESH 0.4
#define CHECK(status) \
    do\
{\
    auto ret = (status);\
    if (ret != 0)\
{\
    std::cerr << "Cuda failure: " << ret << std::endl;\
    abort();\
    }\
    } while (0)
std::map<std::string , nvinfer1::Weights> load_weights(const std::string path)
{
   std::cout << "load weights ing" <<std::endl;
    std::map<std::string , nvinfer1::Weights> weights;

   std::ifstream input(path);
   assert(input.is_open());

   int32_t count ;
   input >> count ;
   assert(count > 0);
   while(count--)
   {
       nvinfer1::Weights weight{DataType::kFLOAT , nullptr , 0};
       int32_t size ;
       std::string name ;
       input >> name >> std::dec >> size ;

       weight.type = DataType::kFLOAT ;
       uint32_t* value = reinterpret_cast<uint32_t* >(malloc(sizeof(value) * size));
       for(size_t i = 0 ; i< size ; i++)
       {
           input >>std::hex>> value[i];
       }
       weight.values = value;
       weight.count = size ;
       weights[name] = weight;
   }
   return weights;
}
IScaleLayer* addBN2dLayer(INetworkDefinition* network , std::map<std::string , nvinfer1::Weights>& weights , ITensor& input , std::string lname , float eps)
{

    float* gamma = (float*)weights[lname + ".weight"].values;
    float* beta = (float*)weights[lname + ".bias"].values;
    float* mean = (float*)weights[lname + ".running_mean"].values;
    float* var = (float*)weights[lname + ".running_var"].values;
    int len = weights[lname + ".running_var"].count;

    float* scale_value = reinterpret_cast<float* >(malloc(sizeof(float) * len)) ;
    for(size_t i = 0 ; i < len ; i++)
    {
        scale_value[i] = gamma[i] / sqrt(var[i] + eps) ;
    }
    nvinfer1::Weights scale{DataType::kFLOAT , scale_value , len} ;

    float* shift_value = reinterpret_cast<float* >(malloc(sizeof(float) * len)) ;
    for(size_t i = 0 ; i < len ; i++)
    {
        shift_value[i] = beta[i] - (gamma[i] * mean[i])/ sqrt(var[i] + eps) ;
    }
    nvinfer1::Weights shift{DataType::kFLOAT , shift_value , len} ;

    float* power_value = reinterpret_cast<float* >(malloc(sizeof(float) * len)) ;
    for(size_t i = 0 ; i < len ; i++)
    {
        power_value[i] = 1.0 ;
    }
    nvinfer1::Weights power{DataType::kFLOAT , power_value , len} ;
    IScaleLayer* scale_layer = network->addScale(input ,ScaleMode::kCHANNEL , shift , scale , power ) ;
    assert(scale_layer);
    return scale_layer;
}

ILayer* convBnLeaky(INetworkDefinition* network , std::map<std::string , nvinfer1::Weights> weights , ITensor& input , int out_channel , int ksize , int stride , int padding , int linx)
{
    std::cout << "enter conv : " << linx << std::endl;
    nvinfer1::Weights bia_weight{DataType::kFLOAT , nullptr , 0} ;

    IConvolutionLayer* conv = network->addConvolutionNd(input , out_channel , DimsHW{ksize , ksize} , weights["module_list." + std::to_string(linx) + ".Conv2d.weight"] , bia_weight);
    assert(conv);
    conv->setStrideNd(DimsHW{stride , stride});
    conv->setPaddingNd(DimsHW{padding , padding});

    IScaleLayer* scale = addBN2dLayer(network , weights , *conv->getOutput(0) , "module_list." + std::to_string(linx) + ".BatchNorm2d" , 1e-4);

    //leaky_reli Plugin 层的插入
    auto leaky = network->addActivation(*scale->getOutput(0) ,ActivationType::kLEAKY_RELU);
    leaky->setAlpha(0.1);
    return leaky ;
}

ILayer* convBnMish(INetworkDefinition* network , std::map<std::string , nvinfer1::Weights> weights , ITensor& input , int out_channel , int ksize , int stride , int padding , int linx)
{
    std::cout << "enter conv : " << linx << std::endl;
    nvinfer1::Weights bia_weight{DataType::kFLOAT , nullptr , 0} ;

    IConvolutionLayer* conv = network->addConvolutionNd(input , out_channel , DimsHW{ksize , ksize} , weights["module_list." + std::to_string(linx) + ".Conv2d.weight"] , bia_weight);
    assert(conv);
    conv->setStrideNd(DimsHW{stride , stride});
    conv->setPaddingNd(DimsHW{padding , padding});

    IScaleLayer* scale = addBN2dLayer(network , weights , *conv->getOutput(0) , "module_list." + std::to_string(linx) + ".BatchNorm2d" , 1e-4);

    //mish Plugin 层的插入
    auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT" , "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2* pluginObj = creator->createPlugin(("mish" + std::to_string(linx)).c_str() , pluginData);
    ITensor* inputTensor[] = {scale->getOutput(0)};
    auto mish = network->addPluginV2(&inputTensor[0] , 1 , *pluginObj);
    return mish ;
}

ICudaEngine* createEngine(size_t batch_size , IBuilder* builder , IBuilderConfig* config , DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_NAME_BLOB , dt , Dims3{3 , Yolo::INPUT_H , Yolo::INPUT_W }) ;
    assert(data);
    //load weights
    std::map<std::string , nvinfer1::Weights> weights = load_weights(path);
    nvinfer1::Weights empty_weights{DataType::kFLOAT , nullptr , 0};

    //define network yolov4 using API
    auto l0 = convBnMish(network , weights , *data , 32 , 3 , 1 , 1 , 0) ;
    auto l1 = convBnMish(network , weights , *l0->getOutput(0) , 64 , 3 , 2 , 1 , 1);
    //res1
    auto l2 = convBnMish(network , weights , *l1->getOutput(0) , 64 , 1 , 1 , 0 , 2);
    auto l3 = l1 ;
    auto l4 = convBnMish(network , weights , *l3->getOutput(0) , 64  , 1 , 1 , 0 , 4);
    auto l5 = convBnMish(network , weights , *l4->getOutput(0) , 32  , 1 , 1 , 0 , 5);
    auto l6 = convBnMish(network , weights , *l5->getOutput(0) , 64  , 3 , 1 , 1 , 6);
    auto ew7 = network->addElementWise(*l6->getOutput(0) , *l4->getOutput(0) , ElementWiseOperation::kSUM);
    auto l8 = convBnMish(network, weights , *ew7->getOutput(0) , 64  , 1 , 1 , 0 , 8);
    ITensor* inputTensor9[] = {l8->getOutput(0) , l2->getOutput(0)};
    auto cat9 = network->addConcatenation(inputTensor9 , 2);
    auto l10 = convBnMish(network , weights , *cat9->getOutput(0) , 64  , 1 , 1 , 0 , 10);
    //res2
    auto l11 = convBnMish(network , weights , *l10->getOutput(0) , 128  , 3 , 2 , 1 , 11);
    auto l12 = convBnMish(network , weights , *l11->getOutput(0) , 64  , 1 , 1 , 0 , 12);
    auto l13 = l11 ;
    auto l14 = convBnMish(network , weights , *l13->getOutput(0) , 64  , 1 , 1 , 0 , 14);
    auto l15 = convBnMish(network , weights , *l14->getOutput(0) , 64  , 1 , 1  , 0 , 15);
    auto l16 = convBnMish(network , weights , *l15->getOutput(0) , 64  , 3 , 1 , 1 , 16);
    auto ew17 = network->addElementWise(*l16->getOutput(0) , *l14->getOutput(0) , ElementWiseOperation::kSUM);
    auto l18 = convBnMish(network, weights , *ew17->getOutput(0) , 64  , 1 , 1 , 0 , 18);
    auto l19 = convBnMish(network, weights , *l18->getOutput(0) , 64  , 3 , 1 , 1 , 19);
    auto ew20 = network->addElementWise(*l19->getOutput(0) , *ew17->getOutput(0) , ElementWiseOperation::kSUM);
    auto l21 = convBnMish(network, weights , *ew20->getOutput(0) , 64  , 1 , 1 , 0 , 21);
    ITensor* inputTensor22[] = {l21->getOutput(0) , l12->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensor22 , 2);
    auto l23 = convBnMish(network , weights , *cat22->getOutput(0) , 128  , 1 , 1 , 0 , 23);
    //res8
    auto l24 = convBnMish(network , weights , *l23->getOutput(0) , 256  , 3 , 2 , 1 , 24);
    auto l25 = convBnMish(network , weights , *l24->getOutput(0) , 128  , 1 , 1 , 0 , 25);
    auto l26 = l24 ;
    auto l27 = convBnMish(network , weights , *l26->getOutput(0) , 128  , 1 , 1 , 0 , 27);
    auto l28 = convBnMish(network , weights , *l27->getOutput(0) , 128  , 1 , 1  , 0 , 28);
    auto l29 = convBnMish(network , weights , *l28->getOutput(0) , 128  , 3 , 1 , 1 , 29);
    auto ew30 = network->addElementWise(*l29->getOutput(0) , *l27->getOutput(0) , ElementWiseOperation::kSUM);
    auto l31 = convBnMish(network , weights , *ew30->getOutput(0) , 128  , 1 , 1  , 0 , 31);
    auto l32 = convBnMish(network , weights , *l31->getOutput(0) , 128  , 3 , 1 , 1 , 32);
    auto ew33 = network->addElementWise(*l32->getOutput(0) , *ew30->getOutput(0) , ElementWiseOperation::kSUM);
    auto l34 = convBnMish(network , weights , *ew33->getOutput(0) , 128  , 1 , 1  , 0 , 34);
    auto l35 = convBnMish(network , weights , *l34->getOutput(0) , 128  , 3 , 1 , 1 , 35);
    auto ew36 = network->addElementWise(*l35->getOutput(0) , *ew33->getOutput(0) , ElementWiseOperation::kSUM);
    auto l37 = convBnMish(network , weights , *ew36->getOutput(0) , 128  , 1 , 1  , 0 , 37);
    auto l38 = convBnMish(network , weights , *l37->getOutput(0) , 128  , 3 , 1 , 1 , 38);
    auto ew39 = network->addElementWise(*l38->getOutput(0) , *ew36->getOutput(0) , ElementWiseOperation::kSUM);
    auto l40 = convBnMish(network , weights , *ew39->getOutput(0) , 128  , 1 , 1  , 0 , 40);
    auto l41 = convBnMish(network , weights , *l40->getOutput(0) , 128  , 3 , 1 , 1 , 41);
    auto ew42 = network->addElementWise(*l41->getOutput(0) , *ew39->getOutput(0) , ElementWiseOperation::kSUM);
    auto l43 = convBnMish(network , weights , *ew42->getOutput(0) , 128  , 1 , 1  , 0 , 43);
    auto l44 = convBnMish(network , weights , *l43->getOutput(0) , 128  , 3 , 1 , 1 , 44);
    auto ew45 = network->addElementWise(*l44->getOutput(0) , *ew42->getOutput(0) , ElementWiseOperation::kSUM);
    auto l46 = convBnMish(network , weights , *ew45->getOutput(0) , 128  , 1 , 1  , 0 , 46);
    auto l47 = convBnMish(network , weights , *l46->getOutput(0) , 128  , 3 , 1 , 1 , 47);
    auto ew48 = network->addElementWise(*l47->getOutput(0) , *ew45->getOutput(0) , ElementWiseOperation::kSUM);
    auto l49 = convBnMish(network , weights , *ew48->getOutput(0) , 128  , 1 , 1  , 0 , 49);
    auto l50 = convBnMish(network , weights , *l49->getOutput(0) , 128  , 3 , 1 , 1 , 50);
    auto ew51 = network->addElementWise(*l50->getOutput(0) , *ew48->getOutput(0) , ElementWiseOperation::kSUM);
    auto l52 = convBnMish(network, weights , *ew51->getOutput(0) , 128  , 1 , 1 , 0 , 52);
    ITensor* inputTensor53[] = {l52->getOutput(0) , l25->getOutput(0)};

    auto cat53 = network->addConcatenation(inputTensor53 , 2);
    auto l54 = convBnMish(network , weights , *cat53->getOutput(0) , 256  , 1 , 1 , 0 , 54);
    //res8
    auto l55 = convBnMish(network , weights , *l54->getOutput(0) , 512  , 3 , 2 , 1 , 55);
    auto l56 = convBnMish(network , weights , *l55->getOutput(0) , 256  , 1 , 1 , 0 , 56);
    auto l57 = l55 ;
    auto l58 = convBnMish(network , weights , *l57->getOutput(0) , 256  , 1 , 1 , 0 , 58);
    auto l59 = convBnMish(network , weights , *l58->getOutput(0) , 256  , 1 , 1  , 0 , 59);
    auto l60 = convBnMish(network , weights , *l59->getOutput(0) , 256  , 3 , 1 , 1 , 60);
    auto ew61 = network->addElementWise(*l60->getOutput(0) , *l58->getOutput(0) , ElementWiseOperation::kSUM);
    auto l62 = convBnMish(network , weights , *ew61->getOutput(0) , 256  , 1 , 1  , 0 , 62);
    auto l63 = convBnMish(network , weights , *l62->getOutput(0) , 256  , 3 , 1 , 1 , 63);
    auto ew64 = network->addElementWise(*l63->getOutput(0) , *ew61->getOutput(0) , ElementWiseOperation::kSUM);
    auto l65 = convBnMish(network , weights , *ew64->getOutput(0) , 256  , 1 , 1  , 0 , 65);
    auto l66 = convBnMish(network , weights , *l65->getOutput(0) , 256  , 3 , 1 , 1 , 66);
    auto ew67 = network->addElementWise(*l66->getOutput(0) , *ew64->getOutput(0) , ElementWiseOperation::kSUM);
    auto l68 = convBnMish(network , weights , *ew67->getOutput(0) , 256  , 1 , 1  , 0 , 68);
    auto l69 = convBnMish(network , weights , *l68->getOutput(0) , 256  , 3 , 1 , 1 , 69);
    auto ew70 = network->addElementWise(*l69->getOutput(0) , *ew67->getOutput(0) , ElementWiseOperation::kSUM);
    auto l71 = convBnMish(network , weights , *ew70->getOutput(0) , 256  , 1 , 1  , 0 , 71);
    auto l72 = convBnMish(network , weights , *l71->getOutput(0) , 256  , 3 , 1 , 1 , 72);
    auto ew73 = network->addElementWise(*l72->getOutput(0) , *ew70->getOutput(0) , ElementWiseOperation::kSUM);
    auto l74 = convBnMish(network , weights , *ew73->getOutput(0) , 256  , 1 , 1  , 0 , 74);
    auto l75 = convBnMish(network , weights , *l74->getOutput(0) , 256  , 3 , 1 , 1 , 75);
    auto ew76 = network->addElementWise(*l75->getOutput(0) , *ew73->getOutput(0) , ElementWiseOperation::kSUM);
    auto l77 = convBnMish(network , weights , *ew76->getOutput(0) , 256  , 1 , 1  , 0 , 77);
    auto l78 = convBnMish(network , weights , *l77->getOutput(0) , 256  , 3 , 1 , 1 , 78);
    auto ew79 = network->addElementWise(*l78->getOutput(0) , *ew76->getOutput(0) , ElementWiseOperation::kSUM);
    auto l80 = convBnMish(network , weights , *ew79->getOutput(0) , 256  , 1 , 1  , 0 , 80);
    auto l81 = convBnMish(network , weights , *l80->getOutput(0) , 256  , 3 , 1 , 1 , 81);
    auto ew82 = network->addElementWise(*l81->getOutput(0) , *ew79->getOutput(0) , ElementWiseOperation::kSUM);
    auto l83 = convBnMish(network, weights , *ew82->getOutput(0) , 256  , 1 , 1 , 0 , 83);
    ITensor* inputTensor84[] = {l83->getOutput(0) , l56->getOutput(0)};
    auto cat84 = network->addConcatenation(inputTensor84 , 2);
    auto l85 = convBnMish(network , weights , *cat84->getOutput(0) , 512  , 1 , 1 , 0 , 85);
    //res4
    auto l86 = convBnMish(network , weights , *l85->getOutput(0) , 1024  , 3 , 2 , 1 , 86);
    auto l87 = convBnMish(network , weights , *l86->getOutput(0) , 512  , 1 , 1  , 0 , 87);
    auto l88 = l86 ;
    auto l89 = convBnMish(network , weights , *l88->getOutput(0) , 512  , 1 , 1  , 0 , 89);
    auto l90 = convBnMish(network , weights , *l89->getOutput(0) , 512  , 1 , 1  , 0 , 90);
    auto l91 = convBnMish(network , weights , *l90->getOutput(0) , 512  , 3 , 1 , 1 , 91);
    auto ew92 = network->addElementWise(*l91->getOutput(0) , *l89->getOutput(0) , ElementWiseOperation::kSUM);
    auto l93 = convBnMish(network , weights , *ew92->getOutput(0) , 512  , 1 , 1  , 0 , 93);
    auto l94 = convBnMish(network , weights , *l93->getOutput(0) , 512  , 3 , 1 , 1 , 94);
    auto ew95 = network->addElementWise(*l94->getOutput(0) , *ew92->getOutput(0) , ElementWiseOperation::kSUM);
    auto l96 = convBnMish(network , weights , *ew95->getOutput(0) , 512  , 1 , 1  , 0 , 96);
    auto l97 = convBnMish(network , weights , *l96->getOutput(0) , 512  , 3 , 1 , 1 , 97);
    auto ew98 = network->addElementWise(*l97->getOutput(0) , *ew95->getOutput(0) , ElementWiseOperation::kSUM);
    auto l99 = convBnMish(network , weights , *ew98->getOutput(0) , 512  , 1 , 1  , 0 , 99);
    auto l100 = convBnMish(network , weights , *l99->getOutput(0) , 512  , 3 , 1 , 1 , 100);
    auto ew101 = network->addElementWise(*l100->getOutput(0) , *ew98->getOutput(0) , ElementWiseOperation::kSUM);
    auto l102 = convBnMish(network, weights , *ew101->getOutput(0) , 512  , 1 , 1 , 0 , 102);
    ITensor* inputTensor103[] = {l102->getOutput(0) , l87->getOutput(0)};
    auto cat103 = network->addConcatenation(inputTensor103 , 2);
    auto l104= convBnMish(network , weights , *cat103->getOutput(0) , 1024  , 1 , 1 , 0 , 104);
    //enter DBL模块
    auto l105= convBnLeaky(network , weights , *l104->getOutput(0) , 512  , 1 , 1 , 0 , 105);
    auto l106= convBnLeaky(network , weights , *l105->getOutput(0) , 1024  , 3 , 1 , 1 , 106);
    auto l107= convBnLeaky(network , weights , *l106->getOutput(0) , 512  , 1 , 1 , 0 , 107);
    //SPP 模块
    auto p108 = network->addPoolingNd(*l107->getOutput(0) , PoolingType::kMAX , DimsHW{5,5});
    p108->setPaddingNd(DimsHW{2 , 2});
    p108->setStrideNd(DimsHW{1 , 1});
    auto l109 = l107 ;
    auto p110 = network->addPoolingNd(*l109->getOutput(0) , PoolingType::kMAX , DimsHW{9,9});
    p110->setPaddingNd(DimsHW{4 , 4});
    p110->setStrideNd(DimsHW{1 , 1});
    auto l111 = l107 ;
    auto p112 = network->addPoolingNd(*l111->getOutput(0) , PoolingType::kMAX , DimsHW{13,13});
    p112->setPaddingNd(DimsHW{6 , 6});
    p112->setStrideNd(DimsHW{1 , 1});
    ITensor* inputTensor113[] = {p112->getOutput(0) ,p110->getOutput(0),p108->getOutput(0),l107->getOutput(0)};
    auto cat113 = network->addConcatenation(inputTensor113 , 4);
    //enter DBL模块
    auto l114= convBnLeaky(network , weights , *cat113->getOutput(0) , 512  , 1 , 1 , 0 , 114);
    auto l115= convBnLeaky(network , weights , *l114->getOutput(0) , 1024  , 3 , 1 , 1 , 115);
    auto l116= convBnLeaky(network , weights , *l115->getOutput(0) , 512  , 1 , 1 , 0 , 116);
    auto l117= convBnLeaky(network , weights , *l116->getOutput(0) , 256  , 1 , 1 , 0 , 117);
    //upsample_1 模块
    float* deval = reinterpret_cast<float* >(malloc(sizeof(float) * 256 * 2 * 2));
    for(size_t i = 0; i< (256*2*2);i++)
    {
        deval[i] = 1.0;
    }
    nvinfer1::Weights DeconvWeight118{DataType::kFLOAT , deval , 256*2*2};
    IDeconvolutionLayer* l118 = network->addDeconvolutionNd(*l117->getOutput(0) , 256 , DimsHW{2,2} ,DeconvWeight118 ,empty_weights);
    assert(l118);
    l118->setStrideNd(DimsHW{2 ,2});
    l118->setNbGroups(256);
    auto l119 = l85;
    auto l120 = convBnLeaky(network , weights , *l119->getOutput(0) , 256  , 1 , 1 , 0 , 120);
    ITensor* inputTensor121[] = {l120->getOutput(0) , l118->getOutput(0)};
    auto cat121 = network->addConcatenation(inputTensor121 , 2);
    //enter DBL模块
    auto l122= convBnLeaky(network , weights , *cat121->getOutput(0) , 256  , 1 , 1 , 0 , 122);
    auto l123= convBnLeaky(network , weights , *l122->getOutput(0) , 512  , 3 , 1 , 1 , 123);
    auto l124= convBnLeaky(network , weights , *l123->getOutput(0) , 256  , 1 , 1 , 0 , 124);
    auto l125= convBnLeaky(network , weights , *l124->getOutput(0) , 512  , 3 , 1 , 1 , 125);
    auto l126= convBnLeaky(network , weights , *l125->getOutput(0) , 256  , 1 , 1 , 0 , 126);
    auto l127= convBnLeaky(network , weights , *l126->getOutput(0) , 128  , 1 , 1 , 0 , 127);
    //upsample_2 模块
    nvinfer1::Weights DeconvWeight128{DataType::kFLOAT , deval , 128*2*2};
    IDeconvolutionLayer* l128 = network->addDeconvolutionNd(*l127->getOutput(0) , 128 , DimsHW{2,2} ,DeconvWeight128 ,empty_weights);
    assert(l128);
    l128->setStrideNd(DimsHW{2 ,2});
    l128->setNbGroups(128);
    auto l129 = l54;
    auto l130 = convBnLeaky(network , weights , *l129->getOutput(0) , 128  , 1 , 1 , 0 , 130);
    ITensor* inputTensor131[] = {l130->getOutput(0) , l128->getOutput(0)};
    auto cat131 = network->addConcatenation(inputTensor131 , 2);
    //enter DBL模块 head1
    auto l132= convBnLeaky(network , weights , *cat131->getOutput(0) , 128  , 1 , 1 , 0 , 132);
    auto l133= convBnLeaky(network , weights , *l132->getOutput(0) , 256  , 3 , 1 , 1 , 133);
    auto l134= convBnLeaky(network , weights , *l133->getOutput(0) , 128  , 1 , 1 , 0 , 134);
    auto l135= convBnLeaky(network , weights , *l134->getOutput(0) , 256  , 3 , 1 , 1 , 135);
    auto l136= convBnLeaky(network , weights , *l135->getOutput(0) , 128  , 1 , 1 , 0 , 136);
    auto l137= convBnLeaky(network , weights , *l136->getOutput(0) , 256  , 3 , 1 , 1 , 137);

    IConvolutionLayer* l138 = network->addConvolutionNd(*l137->getOutput(0) , 3*(Yolo::CLASS_NUM + 5) , DimsHW{1 , 1} , weights["module_list.138.Conv2d.weight"] ,  weights["module_list.138.Conv2d.bias"]);
    assert(l138);
    //head2
    auto l140 = l136;
    auto l141= convBnLeaky(network , weights , *l140->getOutput(0) , 256  , 3 , 2 , 1 , 141);
    ITensor* inputTensor142[] = {l141->getOutput(0) , l126->getOutput(0)};
    auto cat142 = network->addConcatenation(inputTensor142  , 2) ;
    auto l143= convBnLeaky(network , weights , *cat142->getOutput(0) , 256  , 1 , 1 , 0 , 143);
    auto l144= convBnLeaky(network , weights , *l143->getOutput(0) , 512  , 3 , 1 , 1 , 144);
    auto l145= convBnLeaky(network , weights , *l144->getOutput(0) , 256  , 1 , 1 , 0 , 145);
    auto l146= convBnLeaky(network , weights , *l145->getOutput(0) , 512  , 3 , 1 , 1 , 146);
    auto l147= convBnLeaky(network , weights , *l146->getOutput(0) , 256  , 1 , 1 , 0 , 147);
    auto l148= convBnLeaky(network , weights , *l147->getOutput(0) , 512  , 3 , 1 , 1 , 148);

    IConvolutionLayer* l149 = network->addConvolutionNd(*l148->getOutput(0) , 3*(Yolo::CLASS_NUM + 5) , DimsHW{1 , 1} , weights["module_list.149.Conv2d.weight"] ,  weights["module_list.149.Conv2d.bias"]);
    assert(l149);
    //head3
    auto l151 = l147;

    auto l152= convBnLeaky(network , weights , *l151->getOutput(0) , 512  , 3 , 2 , 1 , 152);
    ITensor* inputTensor153[] = {l152->getOutput(0) , l116->getOutput(0)};

    auto cat153 = network->addConcatenation(inputTensor153  , 2) ;
    auto l154= convBnLeaky(network , weights , *cat153->getOutput(0) , 512  , 1 , 1 , 0 , 154);
    auto l155= convBnLeaky(network , weights , *l154->getOutput(0) , 1024  , 3 , 1 , 1 , 155);
    auto l156= convBnLeaky(network , weights , *l155->getOutput(0) , 512  , 1 , 1 , 0 , 156);
    auto l157= convBnLeaky(network , weights , *l156->getOutput(0) , 1024  , 3 , 1 , 1 , 157);
    auto l158= convBnLeaky(network , weights , *l157->getOutput(0) , 512  , 1 , 1 , 0 , 158);
    auto l159= convBnLeaky(network , weights , *l158->getOutput(0) , 1024  , 3 , 1 , 1 , 159);

    IConvolutionLayer* l160 = network->addConvolutionNd(*l159->getOutput(0) , 3*(Yolo::CLASS_NUM + 5) , DimsHW{1 , 1} , weights["module_list.160.Conv2d.weight"] , weights["module_list.160.Conv2d.bias"]);
    assert(160);
//    auto x1 = l138->getOutput(0)->getDimensions() ;
//    auto x2 = l149->getOutput(0)->getDimensions() ;
//    auto x3 = l160->getOutput(0)->getDimensions() ;

    //add yololayer plugin
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT" , "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2* pluginObj = creator->createPlugin("yoloLayer" , pluginData);
    ITensor* getOutTensor[] = {l138->getOutput(0) , l149->getOutput(0) ,l160->getOutput(0) };
    auto yolo = network->addPluginV2(getOutTensor , 3 , *pluginObj);
    yolo->getOutput(0)->setName(OUTPUT_NAME_BLOB);
    network->markOutput(*yolo->getOutput(0));
    //std::cout << "tensor shape is " << l160->getOutput(0)->getDimensions()<<std::endl;

    builder->setMaxBatchSize(batch_size);
    config->setMaxWorkspaceSize(16*(1<<20));
#ifdef FP_16
    config->setFlag(BuilderFlag::kINT8);
#endif
    ICudaEngine* engine =builder->buildEngineWithConfig(*network , *config) ;

    network->destroy();

    for(auto& men : weights )
    {
        free((void*)(men.second.values));
    }

    return engine;
}

void APItoModel(size_t batchsize , IHostMemory** modelStream)
{
    IBuilder* builder = createInferBuilder(logger.getTRTLogger());

    IBuilderConfig* config = builder->createBuilderConfig();


    ICudaEngine* engine =createEngine(batchsize , builder , config , DataType::kFLOAT);

    assert(engine!=nullptr);

    *modelStream = engine->serialize();
    //destory everything down
    engine->destroy();
    builder->destroy();
}

cv::Mat prepare_img(cv::Mat img)
{
    int w, h, x, y;
    float r_w = Yolo::INPUT_W / (img.cols*1.0);
    float r_h = Yolo::INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = Yolo::INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (Yolo::INPUT_H - h) / 2;
    } else {
        w = r_h* img.cols;
        h = Yolo::INPUT_H;
        x = (Yolo::INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(Yolo::INPUT_W, Yolo::INPUT_H, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}
cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (Yolo::INPUT_W - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Yolo::Detection& a, Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}
void nms(std::vector<Yolo::Detection>& res , float* output , float nms_thresh = NMS_THRESH)
{
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + DETECTION_SIZE * i + 4] <= BBOX_CONF_THRESH) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++)
    {
//        std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;

        std::sort(dets.begin() , dets.end() , cmp);
        for(size_t m = 0 ; m < dets.size() ; ++m)
        {
            auto & item = dets[m];
            res.push_back(item);
            for(size_t n = m+1 ; n < dets.size() ;++n)
            {
                if(iou(item.bbox , dets[n].bbox) > nms_thresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

void show_result(std::vector<Yolo::Detection> res , cv::Mat img)
{
    std::map<int, char*> mycls ;
    mycls.insert(std::pair<int , char*>(0,"hat"));
    mycls.insert(std::pair<int , char*>(1,"person"));
    mycls.insert(std::pair<int , char*>(2,"safety"));
    mycls.insert(std::pair<int , char*>(3,"unsafety"));

    std::map<int, cv::Scalar> mycorcle ;
    mycorcle.insert(std::pair<int ,cv::Scalar>(0,cv::Scalar(0,255 ,0)));
    mycorcle.insert(std::pair<int ,cv::Scalar>(1,cv::Scalar(0,0,255)));
    mycorcle.insert(std::pair<int ,cv::Scalar>(2,cv::Scalar(255,255 ,0)));
    mycorcle.insert(std::pair<int ,cv::Scalar>(3,cv::Scalar(0,255,255)));

    int label_scale = Yolo::INPUT_H * 0.00001;
    int base_line;
    cv::Mat dst;
    img.copyTo(dst);
    //std::cout << res.size() << std::endl;
    for (size_t j = 0; j < res.size(); j++)
    {
        cv::Rect r = get_rect(img, res[j].bbox);
        //TO YJC 2020.5.30
        cv::rectangle(img, r, mycorcle[(int)res[j].class_id], 1.0);
        cv::Point p0[4];
        p0[0] = cv::Point(r.x , r.y);
        p0[1] = cv::Point(r.x + r.width , r.y);
        p0[2] = cv::Point(r.x + r.width, r.y + r.height);
        p0[3] = cv::Point(r.x , r.y + r.height);
        //这里p0 就是一维数组的，p[] 编程数组{{}};
        const cv::Point *p[] = {p0};
        int num[] = {4};
//        cv::fillPoly(img, p, num, 1, mycorcle[(int)res[j].class_id], 8);
        cv::addWeighted(dst,0.7,img,0.3,0,dst);
        char showText[128] = {0};
        char* cls = mycls[(int)res[j].class_id] ;
        //                sprintf(showText , "%s : %.2f" , cls, res[j].class_confidence);
        sprintf(showText , "%s" , cls);
        auto size = cv::getTextSize(showText,cv::FONT_HERSHEY_COMPLEX_SMALL,label_scale,1,&base_line);
        cv::putText(dst, showText, cv::Point(r.x, r.y - size.height), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0xFF, 0xFF, 0xFF), 1.0);
    }
    cv::namedWindow("result" , cv::WINDOW_GUI_EXPANDED);
    cv::imshow("result", dst);
    cv::waitKey(1);
}

void doInference(IExecutionContext* context , int batchsize)
{

    float data[batchsize * 3 * Yolo::INPUT_W * Yolo::INPUT_H];
    float prob[batchsize * OUTPUT_SIZE];

    cv::Mat img ;
    cv::VideoCapture cap(video);
    while (cap.read(img))
    {
        cv::Mat pre_img = prepare_img(img);
        for(size_t i = 0 ; i<Yolo::INPUT_W * Yolo::INPUT_H ; i++)
        {
            data[i] = pre_img.at<cv::Vec3b>(i)[2]/ 255.;
            data[i + Yolo::INPUT_W * Yolo::INPUT_H] = pre_img.at<cv::Vec3b>(i)[1]/ 255.;
            data[i + Yolo::INPUT_W * Yolo::INPUT_H *2] = pre_img.at<cv::Vec3b>(i)[0]/ 255.;
        }
        //run inference
        auto start = std::chrono::system_clock::now();
        const ICudaEngine& engine = context->getEngine();
        assert(engine.getNbBindings() == 2);
        void* buffer[2];

        const int input_index = engine.getBindingIndex(INPUT_NAME_BLOB);
        const int output_index = engine.getBindingIndex(OUTPUT_NAME_BLOB);

        CHECK(cudaMalloc(&buffer[input_index] , sizeof(float) * batchsize * 3 * Yolo::INPUT_W * Yolo::INPUT_H));
        CHECK(cudaMalloc(&buffer[output_index] , batchsize *OUTPUT_SIZE * sizeof(float)));

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        CHECK(cudaMemcpyAsync(buffer[input_index], data, batchsize * 3 * Yolo::INPUT_W  * Yolo::INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueue(batchsize, buffer, stream, nullptr);
        CHECK(cudaMemcpyAsync(prob, buffer[output_index], batchsize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffer[input_index]));
        CHECK(cudaFree(buffer[output_index]));
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<Yolo::Detection> res;
        nms(res , prob);
        show_result(res,img);
    }
}
#endif
