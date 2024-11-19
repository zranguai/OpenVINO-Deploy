#pragma once

#ifndef __OPENVINO_BASESTRUCT_H__
#define __OPENVINO_BASESTRUCT_H__
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <ctime>
#include <assert.h>
#pragma pack(4)

struct InputImageParm {
	int _num_image = 1;
	char** _camera_name = nullptr; //相机Name
	char** _camera_status = nullptr; //相机启用状态
	int* _img_height = nullptr;
	int* _img_width = nullptr;
	int* _num_channel = nullptr;
	uchar** _img_data = nullptr;
};

struct sImage {
	std::string _camera_name;
	cv::Mat _image;
	bool _image_status;
};


struct DLResultData
{
	std::string _camera_name; //相机Name
	int _x = -1;
	int _y = -1;
	int _w = -1;
	int _h = -1;
	float _score = -1;
	int _area = -1;
	int _index_label = -1;
	std::string _defect_name;
	float _angel = 0;
	std::vector<cv::Point> _contours;

};

struct SliceMergeData 
{
	// 输入的原图
	cv::Mat _origin_mat;  

	// 分组之后的Mat数组
	std::vector<sImage>_slice_mats;  

	std::vector<cv::Point>_slice_points; 

	// 分组的行与列
	int _slice_m_row;  
	int _slice_n_col;  

	// 没有做overlap时的各分图的宽和高
	int _slice_width;
	int _slice_height;

	// 分图时宽和高的overlap
	int _overlap_w;  
	int _overlap_h;  
};

#endif