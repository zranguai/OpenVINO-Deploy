#pragma once
#ifndef CPPDLL_EXPORTS
#define CPPDLL_API __declspec(dllexport)
#else
#define CPPDLL_API __declspec(dllimport)
#endif
#include "TensorRT_Yolo.h"
#include "TensorRT_Classify.h"
#include "TensorRT_BuildModel.h"
/******************************************
const char* model_path: 模型文件路径
const char* engine_Path: engine文件路径
*******************************************/
extern "C" CPPDLL_API int Build_Model(const char* onnx_Path, const char* engine_Path);



/******************************************
int Input_H_size: 网络模型输入高大小
int Input_W_size: 网络模型输入宽大小
int Input_W_size: 网络模型输入batch大小
float confidence_threshold: 置信度
float nms_threshold: IOU阈值
*******************************************/
extern "C" CPPDLL_API TRT_Classify_Imp * Create_Classify_Imp(int Input_H_size, int Input_W_size, int Batch_Size);
extern "C" CPPDLL_API TRT_YOLO_Imp * Create_YOLO_Imp(int Input_H_size, int Input_W_size, int Batch_Size, float confidence_threshold, float nms_threshold);

/******************************************
const char* model_path: 模型文件路径
const char* classes_path: 缺陷种类文件路径
const char* type：模型类型
*******************************************/
extern "C" CPPDLL_API int Initial_Model_YOLO(const char* model_path, const char* classes_path, TRT_YOLO_Imp * TRT_YOLO_ptr, const char* type);
extern "C" CPPDLL_API int Initial_Model_Classify(const char* model_path, const char* classes_path, TRT_Classify_Imp * TRT_Classify_ptr, const char* type);
/******************************************
InputImageParm input_image_parm: 输入结构体数据
const char*& output: 输出结果数据
*******************************************/
extern "C" CPPDLL_API int DL_Detect_YOLO(InputImageParm input_image_parm, TRT_YOLO_Imp * TRT_YOLO_ptr, const char*& output);
extern "C" CPPDLL_API int DL_Detect_Classify(InputImageParm input_image_parm, TRT_Classify_Imp * TRT_Classify_ptr, const char*& output);

//释放内存
extern "C" CPPDLL_API int Free_Memory_YOLO(TRT_YOLO_Imp * TRT_YOLO_ptr);
extern "C" CPPDLL_API int Free_Memory_Classify(TRT_Classify_Imp * TRT_Classify_ptr);
