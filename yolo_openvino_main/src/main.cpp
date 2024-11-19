//测试
#include "OpenVINO_Detection.h"

#define EXPORT_ENGINE false

int main()
{
	//模型路径、类别路径
	const char* onnxPath = "D:\\DL-Deploy\\OpenVINO_Code\\OpenVINO_Deploy\\resources\\models\\yolov8n-seg.onnx";  //yolov5s-det.onnx  MobileNetV2.onnx  yolov8n-seg.onnx
	const char* classesPath = "D:\\DL-Deploy\\OpenVINO_Code\\OpenVINO_Deploy\\resources\\models\\yolov8n-seg.name";  //yolov5s-det.name  MobileNetV2.name  yolov8n-seg.name
	int Width_Size = 640;  // 640 256
	int Height_Size = 640;  // 640 256
	int batch_size = 4;
	//OV_Classify_Imp* OV_MobileNetV2_Imp = Create_Classify_Imp(Width_Size, Height_Size, batch_size);
	OV_YOLO_Imp* OV_MobileNetV2_Imp = Create_YOLO_Imp(Width_Size, Height_Size, batch_size, 0.5, 0.5);
	//2、模型初始化;
	//int model_init = Initial_Model_Classify(onnxPath, classesPath, OV_MobileNetV2_Imp, "MobileNetV2");
	int model_init = Initial_Model_YOLO(onnxPath, classesPath, OV_MobileNetV2_Imp, "YOLOV8Seg");  // YOLOV5Det  YOLOV8Det  YOLOV8Seg

	//3、获取图片数据（自己模拟时需要写，实际中从软件获取）
	std::vector<cv::String> filenames;
	cv::String folder = "D:\\DL-Deploy\\OpenVINO_Code\\OpenVINO_Deploy\\resources\\datas\\V8";  // V8 V5 flowers
	cv::glob(folder, filenames);

	int srcImage_Width = 2856;  //2856 500 4508
	int srcImage_Height = 3792;  //3792 375  4960
	uchar** temp_data_ptr = new uchar * [srcImage_Width * srcImage_Height * 3];
	char** temp_camera_name_ptr = new char* [100];
	char** temp_camera_status_ptr = new char* [100];
	std::vector<std::string> vCameraNameList;

	for (int i = 0; i < batch_size; i++)
	{
		vCameraNameList.push_back("Camera_" + std::to_string(i + 1));
	}



	InputImageParm input_parm;
	std::vector<int>widths;
	std::vector<int>heights;
	std::vector<int>channels;
	for (size_t i = 0; i < batch_size; i++) {
		temp_data_ptr[i] = new uchar[srcImage_Width * srcImage_Height * 3];
	}


	for (size_t i = 0; i < batch_size; i++) {
		temp_camera_name_ptr[i] = new char[100];
	}
	std::vector<std::string> cstr_list;
	for (size_t i = 0; i < batch_size; i++) {
		temp_camera_status_ptr[i] = new char[100];
		std::string index_status = std::to_string(int(1));

		cstr_list.push_back(index_status);
	}

	input_parm._img_data = temp_data_ptr;
	input_parm._camera_name = temp_camera_name_ptr;
	input_parm._camera_status = temp_camera_status_ptr;
	std::vector<int>size_arry;

	size_arry.push_back(0);
	for (int i = 0; i < batch_size; i++)
	{
		std::cout << filenames[i] << std::endl;

		cv::Mat img = cv::imread(filenames[i]);
		heights.push_back(img.rows);
		widths.push_back(img.cols);
		channels.push_back(img.channels());
		size_arry.push_back(img.rows * img.cols * img.channels());

		std::strcpy(input_parm._camera_name[i], vCameraNameList[i].c_str());
		std::memcpy(input_parm._img_data[i], img.data, img.rows * img.cols * img.channels());
		std::strcpy(input_parm._camera_status[i], cstr_list[i].c_str());

	}
	input_parm._num_image = heights.size();
	input_parm._img_height = heights.data();
	input_parm._img_width = widths.data();
	input_parm._num_channel = channels.data();

	// 4. 开始推理
	int i = 0;
	const char* output = nullptr;
	while (i < 1)
	{
		char* seg_result = nullptr;
		auto start3 = std::chrono::system_clock::now();

		//int ret_dete = DL_Detect_Classify(input_parm, OV_MobileNetV2_Imp, output);
		int ret_dete = DL_Detect_YOLO(input_parm, OV_MobileNetV2_Imp, output);
		std::string re = output;
		std::cout << re << std::endl;

		auto end3 = std::chrono::system_clock::now();
		std::cout << "cost_time1=" << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() << std::endl;

		i++;
	}

	for (size_t i = 0; i < batch_size; i++) {
		temp_data_ptr[i] = nullptr;
	}


	for (size_t i = 0; i < batch_size; i++) {
		temp_camera_name_ptr[i] = nullptr;
	}
	for (size_t i = 0; i < batch_size; i++) {
		temp_camera_status_ptr[i] = nullptr;
	}

	//释放内存
	//int ret_free = Free_Memory_Classify(OV_MobileNetV2_Imp);
	int ret_free = Free_Memory_YOLO(OV_MobileNetV2_Imp);

	return 0;
}
