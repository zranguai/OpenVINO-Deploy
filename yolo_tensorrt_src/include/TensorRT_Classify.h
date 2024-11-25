#pragma once
#ifndef __TENSORRT_CLASSIFY_H__
#define __TENSORRT_CLASSIFY_H__
#include "TensorRT_BaseStruct.h"
#include "TensorRT_Log.h"
#include "TensorRT_ClassifyModel.hpp"
#include "TensorRT_Json.h"
#include <algorithm>
class TRT_Classify_Imp {

public:
	TRT_Classify_Imp(int input_height = 256, int input_width = 256, int input_batch = 4);
	~TRT_Classify_Imp();
public:
	int Get_BatchSize();
	std::vector<std::string> Get_ClassName(std::string class_name_path);
	int Init_Classify_Model(const char* model_path, const char* classes_path, const char* type);
	std::string Run_Classify_Model(InputImageParm input_image_parm);
	void Free_Classify_Memoy();

private:
	//检测方法
	std::vector<std::vector<DLResultData>> ClassifyDet(std::vector<sImage> input_mats);

private:
	//子函数
	std::string GetNowTime();
	std::vector<unsigned char> Load_File(const std::string& file);
	std::vector<sImage> Get_Mats(InputImageParm input_parm);
	int Run_PreProcessingImage(InputImageParm input_image_parm, std::vector<sImage>& input_mats);
	std::vector<std::vector<DLResultData>> Infer_Predict(std::vector<sImage>input_imgs);
	int Get_TRT_Classify_Result(std::vector<std::vector<DLResultData>> dl_data, int& result);
	int VectorArrayCopy(std::vector<std::vector<DLResultData>> vSource, std::vector<std::vector<DLResultData>>& vDst);//检测结果合并
private:
	std::shared_ptr<Classify::Infer>_fast_classify_models = nullptr;
	int _input_height; //输入模型尺寸-高
	int _input_width; //输入模型尺寸-宽
	int _input_batch; //输入模型batch 推荐16,导出时，目前设置最大为8，后面修改48左右
	std::vector<std::string> _class_names;

	std::vector<float> _pyrdown_flag; // 从外部获取参数
	int _pyrdown_scale = 0;    // 内部计算获取
	std::vector<std::vector<DLResultData>>_slice_result;
	Logger& myLogTrt = Logger::getIntance();
};

#endif
