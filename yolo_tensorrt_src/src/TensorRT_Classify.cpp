#include "TensorRT_Classify.h"
#define Debug_Save_Image true
#define DEBUG_Message true

std::string TRT_Classify_Imp::GetNowTime() {
	auto now = std::chrono::system_clock::now();

	// 转换为时间戳（毫秒）
	auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

	// 转换为本地时间
	std::time_t time = std::chrono::system_clock::to_time_t(now);
	std::tm local_time = *std::localtime(&time);

	// 格式化为字符串
	std::ostringstream oss;
	oss << std::put_time(&local_time, "%Y_%m_%d_%H_%M_%S") << "." << std::setfill('0') << std::setw(3) << timestamp % 1000;

	std::string current_time = oss.str();
	return current_time;
}

static Classify::Image cvimg(const cv::Mat& image) {
	return Classify::Image(image.data, image.cols, image.rows);
}



TRT_Classify_Imp::TRT_Classify_Imp(int input_height, int input_width, int input_batch) {
	this->_input_height = input_height; //输入模型尺寸-高
	this->_input_width = input_width; //输入模型尺寸-宽
	this->_input_batch = input_batch; //输入模型batch 推荐16,导出时，目前设置最大为8，后面修改48左右

}

TRT_Classify_Imp::~TRT_Classify_Imp() {
	//delete _dl_input_instancev8_parm;
	//_dl_input_instancev8_parm = nullptr;
}

int TRT_Classify_Imp::Get_BatchSize()
{
	return this->_input_batch;
}

std::vector<std::string> TRT_Classify_Imp::Get_ClassName(std::string class_name_path)
{
	std::vector<std::string> class_names; class_names.clear();
	try
	{
		std::ifstream inFile;
		inFile.open(class_name_path);
		if (!inFile) {
			std::string msg = "GetClassName: 打开缺陷文件失败_" + class_name_path;
			std::cout << msg << std::endl;
			myLogTrt.error(msg);
			return class_names;
		}
		std::string single_name;
		while (inFile >> single_name) {
			class_names.push_back(single_name);
		}

		inFile.close();
	}
	catch (const std::exception& e)
	{
		std::string msg = "GetClassName 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
	}
	return class_names;
}

int TRT_Classify_Imp::Init_Classify_Model(const char* model_path, const char* classes_path, const char* type)
{

	try
	{
		std::vector<uchar>data_model;
		std::vector<uchar>data = Load_File(model_path);
		if (data.size() <= 0) {
			std::string msg = "Init_Classify_Model: engine文件加载为空，请检查模型文件是否存在或是否正常导出!";
			myLogTrt.error(msg);
			return -1;
		}

		

		std::string strtype = type;
		std::string msg = "Init_Classify_Model中，你传入的模型类型是:" + strtype + "!";
		std::cout << msg << std::endl;
		myLogTrt.info(msg);
		_class_names = Get_ClassName(classes_path);
		if (_class_names.size() <= 0) {
			myLogTrt.error("Init_Classify_Model: 获取缺陷列表为空，请检查缺陷文件是否存在或是否正常配置!");
			return -1;
		}

		_fast_classify_models = Classify::load_from_memory(data, _class_names.size());
		if (_fast_classify_models == nullptr) {
			std::string msg = "Init_Classify_Model: 初始化失败，模型为空!";
			std::cout << msg << std::endl;
			myLogTrt.error(msg);
			return -1;
		}
		
		return 0;
	}
	catch (const std::exception& e)
	{
		std::string msg = "Init_Classify_Model 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return -1;
	}

}

std::string TRT_Classify_Imp::Run_Classify_Model(InputImageParm input_image_parm)
{
	try
	{
		if (_fast_classify_models == nullptr) {
			std::string msg = "Run_Classify_Model: 模型为空!";
			std::cout << msg << std::endl;
			myLogTrt.error(msg);
			return "";
		}
		//1.预处理图片数据
		std::vector<sImage> input_mats;
		int ret_pre = Run_PreProcessingImage(input_image_parm, input_mats);
		//2.执行检测
		std::vector < std::vector < DLResultData >> vDetData = ClassifyDet(input_mats);
		//3.汇总结果
		//std::vector<std::vector<DLResultData>> vAllResultData;
		//VectorArrayCopy(vDetData, vAllResultData);

		std::string result_json;
		RAPIDJSON rapid_json;

		result_json = rapid_json.Write_Json(vDetData, input_image_parm);

		return result_json;
	}
	catch (const std::exception& e)
	{
		std::string msg = "Run_Classify_Model 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return "";
	}
}

void TRT_Classify_Imp::Free_Classify_Memoy() {

	if (_fast_classify_models != nullptr) {
		_fast_classify_models.reset();
	}
}


std::vector<std::vector<DLResultData>> TRT_Classify_Imp::ClassifyDet(std::vector<sImage> input_mats)
{
	std::vector<std::vector<DLResultData>> vResultData; vResultData.clear();
	try
	{
		int pyrdown_flag = 0;//_pyrdown_flag[0];
		// 此处可多线程优化
		for (int p = 0; p < pyrdown_flag; p++) {
#pragma omp parallel for
			for (int i = 0; i < input_mats.size(); i++) {
				cv::pyrDown(input_mats[i]._image, input_mats[i]._image);
			}
		}
		std::vector<std::vector<DLResultData>> merge_result = Infer_Predict(input_mats);
		VectorArrayCopy(merge_result, vResultData);
		return vResultData;
	}
	catch (const std::exception& e)
	{
		std::string msg = "ClassifyDet 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return vResultData;//空列表
	}
}

std::vector<unsigned char> TRT_Classify_Imp::Load_File(const std::string& file)
{
	std::vector<uint8_t> data; data.clear();
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open()) {
		std::string msg = "Load_File: 模型文件打开失败，请检查!";
		myLogTrt.error(msg);
		return data;
	}
	in.seekg(0, std::ios::end);
	size_t length = in.tellg();
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();

	return data;
}

std::vector<sImage> TRT_Classify_Imp::Get_Mats(InputImageParm input_parm)
{
	std::vector<sImage>input_imgs; input_imgs.clear();
	try
	{
		int num_image = input_parm._num_image;
		std::vector<int>size_arry;
		std::vector<int>totals;
		size_arry.push_back(0);
		for (int n = 0; n < num_image; n++)
		{
			sImage temp;
			int img_height = input_parm._img_height[n]; //height
			int img_width = input_parm._img_width[n]; //width
			int img_channel = input_parm._num_channel[n]; //channel
			std::cout << "image info--" << "ImageIndex: " << n << " Height: " << img_height << " Width: " << img_width << " Channel: " << img_channel << std::endl;
			size_arry.push_back(img_height * img_width * img_channel);
			int acc = accumulate(size_arry.begin(), size_arry.end() - 1, 0);
			totals.push_back(acc);
			uchar* data_i = input_parm._img_data[n];
			temp._camera_name = input_parm._camera_name[n];
			char* status = input_parm._camera_status[n];
			if (strcmp(status, "1") == 0)
			{
				temp._image_status = true;
			}
			else
			{
				temp._image_status = false;
			}
			if (img_channel == 1)
			{
				temp._image = cv::Mat(cv::Size(img_width, img_height), CV_8UC1, data_i);
				input_imgs.push_back(temp);
			}
			else {
				cv::Mat _temp;
				temp._image = cv::Mat(cv::Size(img_width, img_height), CV_8UC3, data_i);
				input_imgs.push_back(temp);
				//cv::imwrite("./1.jpg", img);
			}

		}
	}
	catch (const std::exception& e)
	{
		std::string msg = "Get_Mats 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
	}

	return input_imgs;
}

int TRT_Classify_Imp::Run_PreProcessingImage(InputImageParm input_image_parm, std::vector<sImage>& input_mats)
{
	input_mats.clear();
	try
	{
		input_mats = Get_Mats(input_image_parm);
		if (input_mats.size() <= 0) {
			std::string msg = "Run_PreProcessingImage: 获取输入图片为空，请检查输入图片或解析是否正确!";
			std::cout << msg << std::endl;
			myLogTrt.error(msg);
			return -1;
		}

	}
	catch (const std::exception& e)
	{
		std::string msg = "Run_PreProcessingImage 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return -1;
	}
	return 0;
}

std::vector<std::vector<DLResultData>> TRT_Classify_Imp::Infer_Predict(std::vector<sImage> input_imgs)
{
	//推理接口
	std::vector<std::vector<DLResultData>> merge_output;
	std::vector <std::vector<cv::Point2f>> maskcountours_list;
	std::map<std::string, std::vector <std::vector<cv::Point2f>>> maskcountours_map;
	int fcount = 0;
	int bcount = 0;
	for (int f = 0; f < (int)input_imgs.size(); f++) {  //循环读取图像
		fcount++;
		if (fcount < _input_batch && f + 1 != (int)input_imgs.size()) continue;

		std::vector<cv::Mat> batch_mats;
		std::vector<std::string> batch_cameraName;
		std::vector<bool> batch_status;
		bcount = bcount * _input_batch;
		for (int b = 0; b < fcount; b++) {
			cv::Mat img = input_imgs[f - fcount + 1 + b]._image;
			batch_cameraName.push_back(input_imgs[f - fcount + 1 + b]._camera_name);

			if (img.empty()) continue; // 判断图像是否为空
			if (img.channels() == 1)
			{
				cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
			}
			batch_mats.push_back(img);
			batch_status.push_back(input_imgs[f - fcount + 1 + b]._image_status);
		}

		std::vector<Classify::Image> classifyimages(batch_mats.size());
		std::transform(batch_mats.begin(), batch_mats.end(), classifyimages.begin(), cvimg);

		auto batched_result = _fast_classify_models->forwards(classifyimages);

		for (int ib = 0; ib < (int)batched_result.size(); ++ib) {

			std::vector<DLResultData> image_output;
			auto& objs = batched_result[ib];
			auto& image = batch_mats[ib];

			
			if ((!objs.empty()) && batch_status[ib] == true) {//有缺陷数据时
				for (auto& obj : objs) {
					DLResultData single_output;
					//std::cout << obj.class_label << std::endl;
					std::string name = _class_names[obj.class_label];
					single_output._camera_name = batch_cameraName[ib];
					single_output._defect_name = name;
					single_output._angel = 0;
					single_output._x = 0;
					single_output._y = 0;
					single_output._w = 1;
					single_output._h = 1;
					single_output._index_label = obj.class_label;
					single_output._score = obj.confidence;

					/*if (DEBUG_Det) {
						single_output._area = single_output._w * single_output._h;
						std::vector<cv::Point>contour;
						contour.push_back(cv::Point(obj.left, obj.top));
						contour.push_back(cv::Point(obj.right, obj.top));
						contour.push_back(cv::Point(obj.right, obj.bottom));
						contour.push_back(cv::Point(obj.left, obj.bottom));
						single_output._contours = contour;
					}*/

					if (DEBUG_Message) {
						uint8_t b, g, r;
						std::tie(b, g, r) = Classify::random_color(obj.class_label);
						

						auto caption = cv::format("%s %.2f", name, obj.confidence);
						int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
						
						cv::putText(image, caption, cv::Point(30, 30), 0, 1, cv::Scalar::all(0), 2, 16);

					}
					

					image_output.push_back(single_output);
				}
				if (Debug_Save_Image)
				{
					std::string strTime = GetNowTime();
					std::string save_path = "./DeepLearning/Save_Result\\" + strTime + "_s.jpg";
					cv::imwrite(save_path, image);
				}

			}
			else {//OK时，也要保存基础信息(如相机Name)
				DLResultData single_output;
				single_output._camera_name = batch_cameraName[ib];
				image_output.push_back(single_output);
			}
			merge_output.push_back(image_output);

		}
		fcount = 0;
		bcount++;

	}
	return merge_output;
}

int TRT_Classify_Imp::Get_TRT_Classify_Result(std::vector<std::vector<DLResultData>> dl_data, int& result)
{
	//功能：单相机接口结果解析，多个异常位置时，按优先级最高的报错
	//-1:检测异常;  0:OK;  1:堵片;  2:碎片;
	try
	{
		//获取缺陷标签列表
		std::vector<int> vDefectName; vDefectName.clear();
		for (size_t i = 0; i < dl_data.size(); i++) {
			for (size_t j = 0; j < dl_data[i].size(); j++) {
				DLResultData tempData = dl_data[i][j];
				vDefectName.push_back(tempData._index_label);
			}
		}

		if (vDefectName.size() > 0) {//获取下标小(缺陷等级高)
			std::sort(vDefectName.begin(), vDefectName.end());
			result = vDefectName[0] + 1;//标签默认下标从0开始，这里传出0是OK，所以+1从1开始
		}
		return 0;
	}
	catch (const std::exception& e)
	{
		std::string msg = "Get_TRT_Classify_Result 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return -1;
	}

}


int TRT_Classify_Imp::VectorArrayCopy(std::vector<std::vector<DLResultData>> vSource, std::vector<std::vector<DLResultData>>& vDst)
{

	for (const auto& sData : vSource) {
		std::vector<DLResultData> tempData;
		for (const auto& data : sData) {
			tempData.push_back(data);
		}
		vDst.push_back(tempData);
	}
	return 0;
}




