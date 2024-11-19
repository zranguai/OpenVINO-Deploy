#include "OpenVINO_Json.h"
/*
// 对应的场景：
(1)单张图输入，不切块，多检测结果输出
(2)单张图输入，切块，内部拼接，多检测结果输出
*/
std::string RAPIDJSON::Write_Json(std::vector<std::vector<DLResultData>> dl_result_data, InputImageParm input_image_parm) {
	rapidjson::StringBuffer buffer;

	//Writer<StringBuffer> writer(buffer);  // 非格式化输出
	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);  // 格式化输出
	writer.StartObject(); // 最外层
	writer.Key("ResultSet");
	writer.StartArray();// list

	for (int num_img = 0; num_img < dl_result_data.size(); num_img++)
	{
		writer.StartObject();
		writer.Key("ImageInfo");
		writer.StartObject();

		////修改前
		//writer.Key("ImageIndex");
		//writer.Int(num_img);
		//修改后
		writer.Key("CameraName");//相机Name名
		writer.String(dl_result_data[num_img][0]._camera_name.c_str());

		writer.Key("Channel");
		writer.Int(input_image_parm._num_channel[num_img]);
		writer.Key("Width");
		writer.Int(input_image_parm._img_width[num_img]);
		writer.Key("Height");
		writer.Int(input_image_parm._img_height[num_img]);
		writer.EndObject();

		writer.Key("DefectList");
		writer.StartArray();
		for (int i = 0; i < dl_result_data[num_img].size() && dl_result_data[num_img][0]._score != -1; i++)  // 每个图片缺陷数
		{
			writer.StartObject();

			// DefectInfo  信息开始
			writer.Key("DefectInfo");
			writer.StartObject();

			writer.Key("DefectX");
			writer.Int(dl_result_data[num_img][i]._x);
			writer.Key("DefectY");
			writer.Int(dl_result_data[num_img][i]._y);
			writer.Key("DefectWidth");
			writer.Int(dl_result_data[num_img][i]._w);
			writer.Key("DefectHeight");
			writer.Int(dl_result_data[num_img][i]._h);
			writer.Key("DefectAngle");
			writer.Double(dl_result_data[num_img][i]._angel);

			writer.Key("DefectName");
			writer.String(dl_result_data[num_img][i]._defect_name.c_str());

			writer.Key("DefectArea");
			writer.Double(dl_result_data[num_img][i]._area);
			writer.Key("DefectScore");
			writer.Double(dl_result_data[num_img][i]._score);
			writer.Key("DefectIndexLabel");
			writer.Int(dl_result_data[num_img][i]._index_label);
			////暂时用不到轮廓点，屏蔽
			//writer.StartArray();
			//writer.Key("DefectContour");
			//for (int c = 0; c < dl_result_data[num_img][i]._contours.size(); c++)
			//{
			//	writer.StartObject();
			//	writer.Key("X");
			//	writer.Int(dl_result_data[num_img][i]._contours[c].x);
			//	writer.Key("Y");
			//	writer.Int(dl_result_data[num_img][i]._contours[c].y);
			//	writer.EndObject();
			//}
			//writer.EndArray();
			writer.EndObject();
			writer.EndObject();

		}

		writer.EndArray();
		writer.EndObject();
	}
	writer.EndArray();
	writer.EndObject();
	std::string result = buffer.GetString();//string(buffer.GetString());
	return result;
}

std::string RAPIDJSON::Write_Json(std::vector<DLResultData> dl_result_data, InputImageParm input_image_parm) {
	rapidjson::StringBuffer buffer;

	//Writer<StringBuffer> writer(buffer);  // 非格式化输出
	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);  // 格式化输出
	writer.StartObject(); // 最外层
	writer.Key("ResultSet");
	writer.StartArray();// list

	for (int num_img = 0; num_img < 1; num_img++)
	{
		writer.StartObject();
		writer.Key("ImageInfo");
		writer.StartObject();

		////修改前
		//writer.Key("Imageindex");
		//writer.Int(-1);
		//修改后
		writer.Key("CameraName");//相机Name名
		writer.String(dl_result_data[num_img]._camera_name.c_str());

		writer.Key("Channel");
		writer.Int(input_image_parm._num_channel[num_img]);
		writer.Key("Stride");
		writer.Int(input_image_parm._num_channel[num_img] * input_image_parm._img_width[num_img]);
		writer.Key("Width");
		writer.Int(input_image_parm._img_width[num_img]);
		writer.Key("Height");
		writer.Int(input_image_parm._img_height[num_img]);
		writer.EndObject();

		writer.Key("DefectList");
		writer.StartArray();
		for (int i = 0; i < dl_result_data.size(); i++)  // 每个图片缺陷数
		{
			writer.StartObject();

			// DefectInfo  信息开始
			writer.Key("DefectInfo");
			writer.StartObject();

			writer.Key("DefectX");
			writer.Int(dl_result_data[i]._x);
			writer.Key("DefectY");
			writer.Int(dl_result_data[i]._y);
			writer.Key("DefectWidth");
			writer.Int(dl_result_data[i]._w);
			writer.Key("DefectHeight");
			writer.Int(dl_result_data[i]._h);
			writer.Key("DefectAngle");
			writer.Double(dl_result_data[i]._angel);

			writer.Key("DefectName");
			writer.String(dl_result_data[i]._defect_name.c_str());

			writer.Key("DefectArea");
			writer.Double(dl_result_data[i]._area);
			writer.Key("DefectScore");
			writer.Double(dl_result_data[i]._score);
			writer.Key("DefectIndexLabel");
			writer.Int(dl_result_data[i]._index_label);

			/*writer.StartArray();
			writer.Key("DefectContour");
			for (int c = 0; c < dl_result_data[i]._contours.size(); c++)
			{
				writer.StartObject();
				writer.Key("X");
				writer.Int(dl_result_data[i]._contours[c].x);
				writer.Key("Y");
				writer.Int(dl_result_data[i]._contours[c].y);
				writer.EndObject();
			}
			writer.EndArray();*/
			writer.EndObject();
			writer.EndObject();

		}

		writer.EndArray();
		writer.EndObject();
	}
	writer.EndArray();
	writer.EndObject();


	std::string result = buffer.GetString();

	return result;
}