#include "OpenVINO_Yolo.h"
#define DEBUG_Rectangle true
#define DEBUG_Seg true
#define Debug_Save_Image true

std::string OV_YOLO_Imp::GetNowTime() {
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

static yolo::Image cvimg(const cv::Mat& image) {
	return yolo::Image(image.data, image.cols, image.rows);
}

bool CompareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	double area1 = cv::contourArea(contour1);
	double area2 = cv::contourArea(contour2);
	return (area1 > area2);
}

OV_YOLO_Imp::OV_YOLO_Imp(int input_height, int input_width, int input_batch, float confidence_threshold, float nms_threshold) {
	this->_input_height = input_height; //输入模型尺寸-高
	this->_input_width = input_width; //输入模型尺寸-宽
	this->_input_batch = input_batch; //输入模型batch 推荐16,导出时，目前设置最大为8，后面修改48左右
	this->_confidence_threshold = confidence_threshold;
	this->_nms_threshold = nms_threshold;
}

OV_YOLO_Imp::~OV_YOLO_Imp() {
	//delete _dl_input_instancev8_parm;
	//_dl_input_instancev8_parm = nullptr;
}

int OV_YOLO_Imp::Get_BatchSize()
{
	return this->_input_batch;
}

std::vector<std::string> OV_YOLO_Imp::Get_ClassName(std::string class_name_path)
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

int OV_YOLO_Imp::Init_YOLO_Model(const char* model_path, const char* classes_path, const char* type)
{

	try
	{
		yolo::Type model_type;

		std::string strtype = type;
		const char* str1 = "YOLOV5Det";
		const char* str2 = "YOLOV8Det";
		const char* str3 = "YOLOV8Seg";
		if (strcmp(str1, type) == 0) {
			model_type = yolo::Type::V5Det;
			std::string msg = "Init_YOLO_Model, The model type you passed in is:" + strtype + "!";
			std::cout << msg << std::endl;
			myLogTrt.info(msg);
		}
		else if (strcmp(str2, type) == 0) {
			model_type = yolo::Type::V8Det;
			std::string msg = "Init_YOLO_Model, The model type you passed in is:" + strtype + "!";
			std::cout << msg << std::endl;
			myLogTrt.info(msg);
		}
		else if (strcmp(str3, type) == 0) {
			model_type = yolo::Type::V8Seg;
			std::string msg = "Init_YOLO_Model, The model type you passed in is:" + strtype + "!";
			std::cout << msg << std::endl;
			myLogTrt.info(msg);
		}
		else {

			std::string msg = "Init_YOLO_Model, The model type you passed in is:" + strtype + "!" + "you should input YOLOV5Det，YOLOV8Det，YOLOV8Seg！";
			std::cout << msg << std::endl;
			myLogTrt.error(msg);
			return -1;
		}

		_fast_yolo_models = yolo::load_from_memory(model_path, model_type, _input_batch, _input_width, _input_height, _confidence_threshold, _nms_threshold);
		if (_fast_yolo_models == nullptr) {
			std::string msg = "Init_YOLO_Model: init error，model is none!";
			std::cout << msg << std::endl;
			myLogTrt.error(msg);
			return -1;
		}
		_class_names = Get_ClassName(classes_path);
		if (_class_names.size() <= 0) {
			myLogTrt.error("Init_YOLO_Model: The defect list is empty. Please check whether the defect file exists or is configured normally!");
			return -1;
		}
		return 0;
	}
	catch (const std::exception& e)
	{
		std::string msg = "Init_Det_Model Caught exception: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return -1;
	}

}

std::string OV_YOLO_Imp::Run_YOLO_Model(InputImageParm input_image_parm)
{
	try
	{
		if (_fast_yolo_models == nullptr) {
			std::string msg = "Run_YOLO_Model: Model is empty!";
			std::cout << msg << std::endl;
			myLogTrt.error(msg);
			return "";
		}
		//1.预处理图片数据
		std::vector<sImage> input_mats;
		const bool IsCalibrator = true;  // 开启畸变矫正
		int ret_pre = Run_PreProcessingImage(input_image_parm, IsCalibrator, input_mats);
		bool bSliceFlag = false;  // 开启切图检测
		//2.执行检测
		std::vector < std::vector < DLResultData >> vDetData = YoloDet(input_mats, bSliceFlag);
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
		std::string msg = "Run_YOLO_Model Caught exception: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return "";
	}
}

void OV_YOLO_Imp::Free_YOLO_Memoy() {

	if (_fast_yolo_models != nullptr) {
		_fast_yolo_models.reset();
	}
}

// 切图算法
SliceMergeData Slice_Mat(sImage input_simage, const int col_numbers, const int row_numbers, const int w_overlap, const int h_overlap)
{
	try
	{
		SliceMergeData Slice_data;
		//if (int(col_numbers * row_numbers) % 2 != 0)
		//{
		//	cout << "请输入2的倍数!" << endl;
		//	exit(0);
		//}
		cv::Mat frame = input_simage._image;

		int w = frame.cols;
		int h = frame.rows;
		int sub_width = w / col_numbers;
		int sub_height = h / row_numbers;

		sImage simage;
		vector<sImage> simage_batch;

		if (col_numbers != 1 || row_numbers != 1)
		{
			for (int yy = 0; yy < row_numbers; yy++)
			{
				for (int xx = 0; xx < col_numbers; xx++)
				{
					cv::Mat src;
					if (xx != 0 && yy != 0)
					{
						frame({ sub_width * xx - w_overlap, sub_height * yy - h_overlap, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					}
					else if (xx == 0 && yy != 0)
					{
						frame({ 0, sub_height * yy - h_overlap, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					}
					else if (xx != 0 && yy == 0)
					{
						frame({ sub_width * xx - w_overlap, 0, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					}
					else if (xx == 0 && yy == 0)
					{
						frame({ 0, 0, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					}
					simage._image = src;
					simage._camera_name = input_simage._camera_name;
					simage._image_status = input_simage._image_status;
					simage_batch.push_back(simage);
				}
			}
			//// 原图也加进去
			//simage._image = frame;
			//simage._camera_name = input_simage._camera_name;
			//simage._image_status = input_simage._image_status;
		}

		Slice_data._origin_mat = frame;
		Slice_data._slice_mats = simage_batch;
		Slice_data._slice_n_col = col_numbers;
		Slice_data._slice_m_row = row_numbers;
		Slice_data._slice_width = sub_width;
		Slice_data._slice_height = sub_height;
		Slice_data._overlap_w = w_overlap;
		Slice_data._overlap_h = h_overlap;
		return Slice_data;
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << std::endl;
	}
}


std::vector<std::vector<DLResultData>> OV_YOLO_Imp::YoloDet(std::vector<sImage> input_mats, bool bSliceFlag)
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

		std::vector<int> slice_judge;
		slice_judge.clear();
		if (bSliceFlag) {//是否需要开启切块,屏蔽
			for (int i = 0; i < input_mats.size(); i++)
			{
				std::vector<cv::Mat> input_slice;
				int p_width = input_mats[i]._image.cols;
				int p_height = input_mats[i]._image.rows;
				int slice_col = (float(p_width) / float(_input_width)) > 1 ? int(round(p_width / _input_width)) : 1;
				int slice_row = (float(p_height) / float(_input_height)) > 1 ? int(round(p_height / _input_height)) : 1;
				if (slice_row > 1 || slice_col > 1) {
					slice_judge.push_back(1);
				}
			}
		}


		if (slice_judge.size() != 0)
		{
			for (int i = 0; i < input_mats.size(); i++) {
				std::vector<sImage> input_slice;
				int p_width = input_mats[i]._image.cols;
				int p_height = input_mats[i]._image.rows;
				int slice_col = (float(p_width) / float(_input_width)) > 1 ? int(round(p_width / _input_width)) : 1;
				int slice_row = (float(p_height) / float(_input_height)) > 1 ? int(round(p_height / _input_height)) : 1;
				int overlap_h = int(_input_height * 0.05);
				int overlap_w = int(_input_width * 0.05);
				SliceMergeData slice_merge_data;
				if (slice_row > 1 || slice_col > 1) {
#if 0
					//这一行启用需调用Slice_Mat函数，需要调用图像预处理库
					slice_col = 3;
					slice_row = 3;
					overlap_h = 100;
					overlap_w = 100;
					slice_merge_data = Slice_Mat(input_mats[i], slice_row, slice_col, overlap_h, overlap_w);
#endif
					input_slice = slice_merge_data._slice_mats;

					// // 针对元素大图单独处理
					if (slice_row >= slice_col) {
						//_pyrdown_scale = std::ceil(slice_row / 2);
						_pyrdown_scale = std::ceil(slice_row / 2) - 1;
					}
					else {
						//_pyrdown_scale = std::ceil(slice_col / 2);
						_pyrdown_scale = std::ceil(slice_col / 2) - 1;
					}
					for (int d_s = 0; d_s < _pyrdown_scale; d_s++) {
						cv::pyrDown(input_mats[i]._image, input_mats[i]._image);
					}
					input_slice.push_back(input_mats[i]); // 最后添加未切块的原图推理
				}
				else {
					// 小于等于网络输入尺寸的图
					input_slice.push_back(input_mats[i]);
				}

				std::vector<std::vector<DLResultData>> slice_result = Infer_Predict(input_slice);
				// 结果合并
				if (slice_result.size() == 1) {
					// 无切块结果
					VectorArrayCopy(slice_result, vResultData);
				}
				else { // 切块结果
					std::vector<DLResultData> merge_result = Merge_Det_Result(slice_result, slice_merge_data);

					vResultData.push_back(merge_result);//添加合并后的结果到列表中
				}
			}
		}
		else
		{
			std::vector<std::vector<DLResultData>> merge_result = Infer_Predict(input_mats);
			VectorArrayCopy(merge_result, vResultData);
		}
		return vResultData;
	}
	catch (const std::exception& e)
	{
		std::string msg = "YoloDet 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return vResultData;//空列表
	}
}


std::vector<sImage> OV_YOLO_Imp::Get_Mats(InputImageParm input_parm, const bool IsCalibrator)
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
			//uchar* data_cali = input_parm._img_data[n];

			/* 新增畸变校正 */
			const char* fileNameChar = "CalibrationParamsR.yaml";

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
				if (IsCalibrator)
				{
					Calibrate(fileNameChar, data_i, img_height, img_width, data_i);
					temp._image = cv::Mat(cv::Size(img_width, img_height), CV_8UC1, data_i);
				}
				else
				{
					temp._image = cv::Mat(cv::Size(img_width, img_height), CV_8UC1, data_i);
				}
				input_imgs.push_back(temp);
			}
			else 
			{
				if (IsCalibrator)
				{
					Calibrate(fileNameChar, data_i, img_height, img_width, data_i);
					temp._image = cv::Mat(cv::Size(img_width, img_height), CV_8UC3, data_i);
				}
				else
				{
					cv::Mat _temp;
					temp._image = cv::Mat(cv::Size(img_width, img_height), CV_8UC3, data_i);
					//cv::imwrite("./1.jpg", img);
				}
				input_imgs.push_back(temp);
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

int OV_YOLO_Imp::Run_PreProcessingImage(InputImageParm input_image_parm, const bool IsCalibrator, std::vector<sImage>& input_mats)
{
	input_mats.clear();
	try
	{
		input_mats = Get_Mats(input_image_parm, IsCalibrator);
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

std::vector<std::vector<DLResultData>> OV_YOLO_Imp::Infer_Predict(std::vector<sImage> input_imgs)
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

		std::vector<yolo::Image> yoloimages(batch_mats.size());
		std::transform(batch_mats.begin(), batch_mats.end(), yoloimages.begin(), cvimg);

		auto batched_result = _fast_yolo_models->forwards(yoloimages);

		for (int ib = 0; ib < (int)batched_result.size(); ++ib) {

			std::vector<DLResultData> image_output;
			auto& objs = batched_result[ib];
			auto& image = batch_mats[ib];

			float ratio_h = float(image.rows) / float(_input_height);
			float ratio_w = float(image.cols) / float(_input_width);
			float ratio = 0;
			if (ratio_h >= ratio_w) {
				ratio = ratio_h;
			}
			else {
				ratio = ratio_w;
			}
			if ((!objs.empty()) && batch_status[ib] == true) {//有缺陷数据时
				for (auto& obj : objs) {
					DLResultData single_output;
					//std::cout << obj.class_label << std::endl;
					std::string name = _class_names[obj.class_label];
					single_output._camera_name = batch_cameraName[ib];
					single_output._defect_name = name;
					single_output._angel = 0;
					single_output._x = obj.left;
					single_output._y = obj.top;
					single_output._w = obj.right - obj.left;
					single_output._h = obj.bottom - obj.top;
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

					if (DEBUG_Rectangle) {
						uint8_t b, g, r;
						std::tie(b, g, r) = yolo::random_color(obj.class_label);
						cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
							cv::Scalar(b, g, r), 5);

						auto caption = cv::format("%s %.2f", name, obj.confidence);
						int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
						cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
							cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
						cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

					}
					if (obj.seg) {
						cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
						cv::Mat binary_mask;
						cv::threshold(mask, binary_mask, 127, 255, cv::THRESH_BINARY);

						std::vector<std::vector<cv::Point>> contours;
						cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
						std::sort(contours.begin(), contours.end(), CompareContourAreas);
						if (contours.size() != 0) {
							for (int c = 0; c < contours.size(); c++) {

								for (int n = 0; n < contours[c].size(); n++) {
									int xx = std::round(contours[c][n].x) + int(obj.left);
									int yy = std::round(contours[c][n].y) + int(obj.top);
									contours[c][n] = cv::Point(xx, yy);
								}
							}
							////暂时屏蔽轮廓赋值(软件不需要)
							//single_output._contours = contours[0];
							single_output._area = cv::contourArea(contours[0]);

							if (DEBUG_Seg) {
								cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 0), 1);
							}
						}
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

int OV_YOLO_Imp::Get_TRT_Det_Result(std::vector<std::vector<DLResultData>> dl_data, int& result)
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
		std::string msg = "Get_TRT_Det_Result 捕获到异常: ";
		std::string strExcep = e.what();
		msg = msg + strExcep;
		std::cout << msg << std::endl;
		myLogTrt.error(msg);
		return -1;
	}

}

/*保留接口未使用，未测试*/
std::vector<DLResultData> OV_YOLO_Imp::Merge_Det_Result(std::vector<std::vector<DLResultData>>slice_result, SliceMergeData slice_mats) {

	std::vector<DLResultData> merge_reuslts;

	for (int i = 0; i < slice_result.size() - 1; i++) // 减1 是去掉最后的原图结果
	{
		for (int j = 0; j < slice_result[i].size(); j++)
		{
			if (slice_result[i][j]._score == -1) {//过滤掉块OK的结果
				continue;
			}
			DLResultData single_det;
			single_det = slice_result[i][j];
			single_det._x = slice_result[i][j]._x + slice_mats._slice_points[i].x;
			single_det._y = slice_result[i][j]._y + slice_mats._slice_points[i].y;
			std::vector<cv::Point> pyrup_contours;
			for (int c = 0; c < slice_result[i][j]._contours.size(); c++)
			{
				pyrup_contours.push_back(slice_result[i][j]._contours[c] + cv::Point(slice_mats._slice_points[i].x, slice_mats._slice_points[i].y));
			}
			single_det._contours = pyrup_contours;

			merge_reuslts.push_back(single_det);
		}
	}

	std::vector<DLResultData> origin_mat_result = slice_result.back();  // 最后一张图单独处理
	for (int i = 0; i < origin_mat_result.size(); i++)
	{
		if (origin_mat_result[i]._score == -1) {//过滤掉块OK的结果
			continue;
		}
		cv::Mat origin_mat = slice_mats._origin_mat;
		int height = origin_mat.rows;
		int width = origin_mat.cols;

		DLResultData single_det;
		single_det = origin_mat_result[i];
		single_det._h = origin_mat_result[i]._h * (pow(2, _pyrdown_scale));
		single_det._w = origin_mat_result[i]._w * (pow(2, _pyrdown_scale));
		/*if ((float(single_det._h) / height) > _set_min_scale && (float(single_det._w) / width) > _set_min_scale) {*/
		single_det._x = origin_mat_result[i]._x * (pow(2, _pyrdown_scale));
		single_det._y = origin_mat_result[i]._y * (pow(2, _pyrdown_scale));
		single_det._area = origin_mat_result[i]._area * (pow(2, _pyrdown_scale) * (pow(2, _pyrdown_scale)));//面积平方
		std::vector<cv::Point>pyrup_contours;
		for (int c = 0; c < origin_mat_result[i]._contours.size(); c++)
		{
			pyrup_contours.push_back(origin_mat_result[i]._contours[c] * pow(2, _pyrdown_scale));
		}

		single_det._contours = pyrup_contours;
		merge_reuslts.push_back(single_det);
		//}

	}

	//新增nms 算法
   //vector<DetResultData>nms_result = NMS(merge_det_result);
   //return nms_result;

	return merge_reuslts;
}

int OV_YOLO_Imp::VectorArrayCopy(std::vector<std::vector<DLResultData>> vSource, std::vector<std::vector<DLResultData>>& vDst)
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
