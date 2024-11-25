# yolo模型的openvino部署

## 文件目录结构
```text
D:.
├─.vscode
├─build
│  ├─*
├─resources
│  ├─datas
│  │  ├─CameraCalibrator(相机标定配置yaml)
│  │  └─V8(模型配置数据)
│  └─models(*.onnx, *.name)
├─yolo_openvino_main
│  └─src
└─yolo_openvino_src
    ├─include
    └─src
```
+ resources里面的datas,models自行准备
+ cmake生成的在build目录下

## 操作指南
+ 依赖库: rapidjson,spdlog,opencv[版本号:3.4.16],openvino[版本号:2023.3.0]
### step1: vcpkg安装rapidjson,spdlog库
+ https://learn.microsoft.com/zh-cn/vcpkg/get_started/get-started?pivots=shell-powershell

+ 操作指令
```
按照上述指令按照好vcpkg后，安装rapidjson,spdlog小型库等

1.vcpkg new --application：生成vcpkg.json等文件
2.vcpkg add port rapidjson: 添加库名字到vcpkg.json里面
3.CMakePresets.json里面配置
4.CMakeUserPresets.json配置vcpkg路径等
```

### step2: 自行下载opencv,openvino库等
1. 下载opencv库，openvino库

2. 将opencv, openvino库的bin目录添加到环境变量path里面并开机重启等

3. 使用cmake-gui工具进行编译，选择opencv,openvino这些库的OpenVINOConfig.cmake所在目录

4. 到vs2019里面生成，运行代码等。