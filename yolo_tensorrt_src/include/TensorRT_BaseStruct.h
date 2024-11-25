#pragma once

#ifndef __TENSORRT_BASESTRUCT_H__
#define __TENSORRT_BASESTRUCT_H__
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

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
	std::string _camera_name;//添加相机Name
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

struct SliceMergeData {
	cv::Mat _origin_mat;
	std::vector<sImage>_slice_mats;
	std::vector<cv::Point>_slice_points;
	int _slice_m_row;
	int _slice_n_col;
	int _overlap_w;
	int _overlap_h;
};

#endif