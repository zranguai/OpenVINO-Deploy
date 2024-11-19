#include"calibrator.h"


void Calibrate(const char* fileNameChar, uchar* input_image, int rows, int cols, uchar* output_image)
{
	try
	{
		/* 读取yaml文件来获取标定参数（获取cameraMatrix, distCoeffs参数） */
		cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
		cv::Mat distCoeffs = cv::Mat(1, 5, CV_64F, cv::Scalar(0));

		/* 标定的参数文件 */
		std::string yaml_path = camera_config + fileNameChar;
		cv::FileStorage ff(yaml_path, cv::FileStorage::READ);
		if (!ff.isOpened())
		{
			std::cout << "标定配置文件读取失败" << std::endl;
		}
		ff["camera_matrix"] >> cameraMatrix;
		ff["camera_distCoeffs"] >> distCoeffs;
		ff.release();

		/* 通过读取的标定的参数对输入图像进行矫正 */
		cv::Size image_size(cols, rows);

		/* 定义畸变校正的输入参数，映射矩阵，旋转矩阵 */
		cv::Mat mapx = cv::Mat(image_size, CV_32FC1);
		cv::Mat mapy = cv::Mat(image_size, CV_32FC1);
		cv::Mat R = cv::Mat::eye(3, 3, CV_32F);

		/* 计算映射矩阵 */
		cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
		cv::Mat Input_Image = cv::Mat(image_size.height, image_size.width, CV_8UC3, input_image);
		/* 这里需要动态分配一下内存给矫正后的图像矩阵，因为如果是局部变量的话，在退出函数时会自动销毁变量，返回的指针是野指针 */
		//cv::Mat* Output_Image = new cv::Mat(image_size.height, image_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat Output_Image;
		if (Input_Image.channels() == 3)
		{
			Output_Image = cv::Mat(image_size.height, image_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
		}
		else
		{
			Output_Image = cv::Mat(image_size.height, image_size.width, CV_8UC1, cv::Scalar(0));
		}

		cv::remap(Input_Image, Output_Image, mapx, mapy, cv::INTER_LINEAR);

		/* 动态分配内存法，还需要将output_image改为引用的方式传入，uchar*& output_image */
		//output_image = Output_Image->data;
		std::memcpy(output_image, Output_Image.data, Output_Image.channels() * Output_Image.rows * Output_Image.cols * sizeof(uint8_t));

	}
	catch (const std::exception& ex)
	{
		std::cout << "畸变校正出错:" << ex.what() << std::endl;
		std::cerr << ex.what() << std::endl;
	}

}