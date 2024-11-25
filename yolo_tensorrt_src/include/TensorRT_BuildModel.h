#pragma once
#ifndef __TENSORRT_BUILD_MODEL__
#define __TENSORRT_BUILD_MODEL__
#include <NvInfer.h>
#include <string>
#include <iostream>
#include <NvOnnxParser.h>
#include <assert.h>
#include <fstream>
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

class TrtLogger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) noexcept override
	{
		// suppress info-level messages
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
};
int OnnxToEngine(const char* onnxPath, const char* enginePath, bool isHalf, int batch_size);
#endif
