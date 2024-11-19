# yolo模型的openvino部署

## 文件目录结构
```text
D:.
├─.vscode
├─build
│  ├─.cmake
│  │  └─api
│  │      └─v1
│  │          ├─query
│  │          │  └─client-vscode   
│  │          └─reply
│  ├─CMakeFiles
│  │  ├─3.29.3
│  │  │  ├─CompilerIdCXX
│  │  │  │  ├─Debug
│  │  │  │  │  └─CompilerIdCXX.tlog
│  │  │  │  └─tmp
│  │  │  └─x64
│  │  │      └─Debug
│  │  │          └─VCTargetsPath.tlog
│  │  ├─5462702e9e8a012eabebdb7d2744f090
│  │  └─pkgRedirects
│  ├─Debug
│  │  ├─bin
│  │  └─lib
│  ├─vcpkg_installed
│  │  ├─vcpkg
│  │  │  ├─info
│  │  │  └─updates
│  │  └─x64-windows
│  │      ├─bin
│  │      ├─debug
│  │      │  ├─bin
│  │      │  └─lib
│  │      │      └─pkgconfig
│  │      ├─include
│  │      │  ├─fmt
│  │      │  ├─rapidjson
│  │      │  │  ├─error
│  │      │  │  ├─internal
│  │      │  │  └─msinttypes
│  │      │  └─spdlog
│  │      │      ├─cfg
│  │      │      ├─details
│  │      │      ├─fmt
│  │      │      └─sinks
│  │      ├─lib
│  │      │  └─pkgconfig
│  │      └─share
│  │          ├─fmt
│  │          ├─rapidjson
│  │          ├─spdlog
│  │          ├─vcpkg-cmake
│  │          └─vcpkg-cmake-config
│  ├─x64
│  │  └─Debug
│  │      ├─ALL_BUILD
│  │      │  └─ALL_BUILD.tlog
│  │      └─ZERO_CHECK
│  │          └─ZERO_CHECK.tlog
│  ├─yolo_openvino_main
│  │  ├─CMakeFiles
│  │  ├─DeepLearning
│  │  │  └─Log
│  │  └─yolo_openvino_main.dir
│  │      └─Debug
│  │          └─yolo_ope.9F82707F.tlog
│  └─yolo_openvino_src
│      ├─CMakeFiles
│      └─yolo_openvino_src.dir
│          └─Debug
│              └─yolo_ope.81463A0C.tlog
├─resources
│  ├─datas
│  │  ├─CameraCalibrator
│  │  └─V8
│  └─models
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