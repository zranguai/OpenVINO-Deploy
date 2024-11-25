#include "TensorRT_BuildModel.h"
TrtLogger trt_logger;


bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}


int OnnxToEngine(const char* onnxPath, const char* enginePath, bool isHalf, int batch_size)
{
	// Create builder(创建生成器并配置显式批次)
	auto builder = nvinfer1::createInferBuilder(trt_logger);
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	// Create model to populate the network(创建网络并解析 ONNX 模型)
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	// Parse ONNX file
	auto parser = nvonnxparser::createParser(*network, trt_logger);

	bool parser_status = parser->parseFromFile(onnxPath, 3);

	// Get the name of network input(获取网络输入尺寸)
	nvinfer1::Dims dim = network->getInput(0)->getDimensions();
	//nvinfer1::Dims dim1 = network->getInput(1)->getDimensions();

	if (dim.d[0] == -1 /*|| dim1.d[0] == -1*/)  // -1 means it is a dynamic model(配置动态批次配置文件)
	{
		const char* name = network->getInput(0)->getName();
		nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
		profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(batch_size, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(batch_size + 2, dim.d[1], dim.d[2], dim.d[3]));

		/*const char* name2 = network->getInput(1)->getName();
		profile->setDimensions(name2, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name2, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(4, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name2, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(8, dim.d[1], dim.d[2], dim.d[3]));*/

		config->addOptimizationProfile(profile);
	}


	// Build engine(配置工作区和精度)
	config->setMaxWorkspaceSize(1 << 30);
	if (isHalf)//一般为半精度，int8需量化
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16); // 半精度
	}
	else
	{
		config->setFlag(nvinfer1::BuilderFlag::kINT8); // 8位整型
	}

	// 构建和序列化引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	// Serialize the model to engine file
	nvinfer1::IHostMemory* modelStream{ nullptr };
	assert(engine != nullptr);
	modelStream = engine->serialize();

	// 将序列化引擎保存到文件
	std::ofstream p(enginePath, std::ios::binary);
	if (!p) {
		std::cerr << "could not open output file to save model" << std::endl;
		return -1;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	std::cout << "generate file success!" << std::endl;

	// Release resources
	modelStream->destroy();
	network->destroy();
	engine->destroy();
	builder->destroy();
	config->destroy();
	return 0;
}