#include "OpenVINO_Detection.h"

//TRT_YOLO_Imp* TRT_YOLOV8Seg_ptr = new TRT_YOLO_Imp(640, 640, 4, 0.5, 0.6);
Logger& myDetectionLog = Logger::getIntance();

OV_Classify_Imp* Create_Classify_Imp(int Input_H_size, int Input_W_size, int Batch_Size)
{
	OV_Classify_Imp* OV_Classify_ptr = new OV_Classify_Imp(Input_H_size, Input_W_size, Batch_Size);
	return OV_Classify_ptr;
}
OV_YOLO_Imp* Create_YOLO_Imp(int Input_H_size, int Input_W_size, int Batch_Size, float confidence_threshold, float nms_threshold)
{
	OV_YOLO_Imp* OV_YOLO_ptr = new OV_YOLO_Imp(Input_H_size, Input_W_size, Batch_Size, confidence_threshold, nms_threshold);
	return OV_YOLO_ptr;
}


/// <summary>
/// 初始化模型文件
/// </summary>
/// <returns></returns>
int Initial_Model_Classify(const char* model_path, const char* classes_path, OV_Classify_Imp* TRT_Classify_ptr, const char* type)
{
	myDetectionLog.info("...start init model...");
	int ret_init = TRT_Classify_ptr->Init_Classify_Model(model_path, classes_path, type);
	int batch_size = TRT_Classify_ptr->Get_BatchSize();
	std::cout << batch_size << std::endl;
	std::string msg = "";
	if (ret_init == 0) {
		msg = "...model init success!!!...";
	}
	else {
		msg = "...model init error!!!...";
	}
	std::cout << msg << std::endl;
	myDetectionLog.info(msg);
	return ret_init;
}
int Initial_Model_YOLO(const char* model_path, const char* classes_path, OV_YOLO_Imp* TRT_YOLO_ptr, const char* type)
{
	myDetectionLog.info("...start init model...");
	int ret_init = TRT_YOLO_ptr->Init_YOLO_Model(model_path, classes_path, type);
	int batch_size = TRT_YOLO_ptr->Get_BatchSize();
	std::cout << batch_size << std::endl;
	std::string msg = "";
	if (ret_init == 0) {
		msg = "...model init success!!!...";
	}
	else {
		msg = "...model init error!!!...";
	}
	std::cout << msg << std::endl;
	myDetectionLog.info(msg);
	return ret_init;
}


/// <summary>
/// 检测接口
/// </summary>
/// <param name="input_image_parm">结构体指针</param>
/// <param name="wafeid">wafeid</param>
/// <param name="output">output</param>
/// <returns>0</returns>
int DL_Detect_YOLO(InputImageParm input_image_parm, OV_YOLO_Imp* TRT_YOLO_ptr, const char*& output)
{
	auto start = std::chrono::system_clock::now();
	myDetectionLog.info("...DL_Detect_YOLO start...");
	std::string result = TRT_YOLO_ptr->Run_YOLO_Model(input_image_parm);
	myDetectionLog.info(result);
	if (result == "") {
		myDetectionLog.info("...DL_Detect_YOLO detect error!!!...");
		return -1;
	}
	else
	{
		myDetectionLog.info("...DL_Detect_YOLO detect error!!!...");
	}

	char out[1024 * 512];
	int i;
	for (i = 0; i < result.length(); i++)
		out[i] = result[i];
	out[i] = '\0'; //注意，一定要手动加上 '\0
	output = out;



	auto end = std::chrono::system_clock::now();
	std::cout << "cost_time =" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
	std::chrono::milliseconds milliseconds_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	myDetectionLog.info("DL_Detect_YOLO detect times：" + std::to_string(milliseconds_time.count()) + " ms");
	return 0;
}
int DL_Detect_Classify(InputImageParm input_image_parm, OV_Classify_Imp* TRT_Classify_ptr, const char*& output)
{
	auto start = std::chrono::system_clock::now();
	myDetectionLog.info("...DL_Detect_Classify start detect...");
	std::string result = TRT_Classify_ptr->Run_Classify_Model(input_image_parm);
	myDetectionLog.info(result);
	if (result == "") {
		myDetectionLog.info("...DL_Detect_Classify detect error...");
		return -1;
	}
	else
	{
		myDetectionLog.info("...DL_Detect_Classify detect success...");
	}

	char out[1024 * 512];
	int i;
	for (i = 0; i < result.length(); i++)
		out[i] = result[i];
	out[i] = '\0'; //注意，一定要手动加上 '\0
	output = out;



	auto end = std::chrono::system_clock::now();
	std::cout << "cost_time =" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
	std::chrono::milliseconds milliseconds_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	myDetectionLog.info("DL_Detect_Classify times：" + std::to_string(milliseconds_time.count()) + " ms");
	return 0;
}
/// <summary>
/// 释放内存接口
/// </summary>
/// <returns>返回值(0:OK; 其他:NG)</returns>
/// 
int Free_Memory_YOLO(OV_YOLO_Imp* OV_YOLO_ptr)
{
	OV_YOLO_ptr->Free_YOLO_Memoy();
	delete OV_YOLO_ptr;
	OV_YOLO_ptr = NULL;

	return 0;
}
int Free_Memory_Classify(OV_Classify_Imp* OV_Classify_ptr)
{

	OV_Classify_ptr->Free_Classify_Memoy();
	delete OV_Classify_ptr;
	OV_Classify_ptr = NULL;

	return 0;
}

