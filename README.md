# yolo模型的TensorRT部署

## 文件目录结构
```text
├─.vscode
├─build
│  ├─*
├─resources
│  ├─datas
│  │  └─V8(image data)
│  └─models(*.onnx,*.engine,*.name)
├─yolo_tensorrt_main
│  └─src
└─yolo_tensorrt_src
    ├─cmake
    ├─cuda_lib(cuda的头文件和源文件)
    │  ├─include
    │  └─src
    ├─include
    └─src
```
+ resources里面的datas,models自行准备
+ cmake生成的在build目录下(可以根据不同平台[windows,linux自行编译等])

## 操作指南
+ 依赖库: rapidjson, spdlog, opencv, cuda[121, 118], tensorrt[8.6.1.6]
### step1: vcpkg安装rapidjson,spdlog库
+ https://learn.microsoft.com/zh-cn/vcpkg/get_started/get-started?pivots=shell-powershell

按照上述指令按照好vcpkg后，安装rapidjson,spdlog小型库等

1. vcpkg new --application：生成vcpkg.json等文件
2. vcpkg add port rapidjson: 添加库名字到vcpkg.json里面
3. CMakePresets.json里面配置
4. CMakeUserPresets.json配置vcpkg路径等

### 自行下载OpenCV, CUDA, TensorRT库等
1. 下载opencv库，cuda, tensorrt库
2. 填好相应的路径到CMakeLists.txt里面
3. 编译文件到build目录中
4. 到vs2019等编译器中生成，运行调试代码等

#### QA:
1. vs2019编译生成时提示缺少nvinfer.lib库等？
```
将C:\tools\TensorRT-8.6.1.6\tensorRT\lib的路径拷贝到
链接器>常规>附件库目录中
```