#pragma once
#ifndef __OPENVINO_JSON_H__
#define __OPENVINO_JSON_H__

#include <string>
#include <iostream>
#include <vector>
#include <rapidjson/document.h>         //https://github.com/Tencent/rapidjson
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/stringbuffer.h>
#include "OpenVINO_BaseStruct.h"
class  RAPIDJSON {
public:
	std::string  Write_Json(std::vector<std::vector<DLResultData>>dl_result_data, InputImageParm input_image_parm);
	std::string  Write_Json(std::vector<DLResultData>dl_result_data, InputImageParm input_image_parm);
};
#endif