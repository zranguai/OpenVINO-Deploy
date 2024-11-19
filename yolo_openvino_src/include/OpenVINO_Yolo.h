#pragma once

#ifndef __OPENVINO_YOLO_H__
#define __OPENVINO_YOLO_H__
#include "OpenVINO_BaseStruct.h"
#include "OpenVINO_Log.h"
#include "OpenVINO_YoloModel.h"
#include "OpenVINO_Json.h"
#include "Calibrator.h"
#include <algorithm>
class OV_YOLO_Imp {

public:
	OV_YOLO_Imp(int input_height = 640, int input_width = 640, int input_batch = 4, float confidence_threshold = 0.5, float nms_threshold = 0.6);
	~OV_YOLO_Imp();
public:
	int Get_BatchSize();
	std::vector<std::string> Get_ClassName(std::string class_name_path);
	int Init_YOLO_Model(const char* model_path, const char* classes_path, const char* type);
	std::string Run_YOLO_Model(InputImageParm input_image_parm);
	void Free_YOLO_Memoy();

private:
	//检测方法
	std::vector<std::vector<DLResultData>> YoloDet(std::vector<sImage> input_mats, bool bSliceFlag = false);

private:
	//子函数
	std::string GetNowTime();

	std::vector<sImage> Get_Mats(InputImageParm input_parm, const bool IsCalibrator);
	int Run_PreProcessingImage(InputImageParm input_image_parm, const bool IsCalibrator, std::vector<sImage>& input_mats);
	std::vector<std::vector<DLResultData>> Infer_Predict(std::vector<sImage>input_imgs);
	int Get_TRT_Det_Result(std::vector<std::vector<DLResultData>> dl_data, int& result);
	std::vector<DLResultData> Merge_Det_Result(std::vector<std::vector<DLResultData>>slice_result, SliceMergeData slice_mats);
	int VectorArrayCopy(std::vector<std::vector<DLResultData>> vSource, std::vector<std::vector<DLResultData>>& vDst);//检测结果合并
private:
	std::shared_ptr<yolo::Infer>_fast_yolo_models = nullptr;
	int _input_height; //输入模型尺寸-高
	int _input_width; //输入模型尺寸-宽
	int _input_batch; //输入模型batch 推荐16,导出时，目前设置最大为8，后面修改48左右
	float _confidence_threshold;
	float _nms_threshold;
	std::vector<std::string> _class_names;

	std::vector<float> _pyrdown_flag; // 从外部获取参数
	int _pyrdown_scale = 0;    // 内部计算获取
	std::vector<std::vector<DLResultData>>_slice_result;
	Logger& myLogTrt = Logger::getIntance();
};
#endif
