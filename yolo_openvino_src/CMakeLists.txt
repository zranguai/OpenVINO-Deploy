cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(yolo_openvino_src CXX)

# 在项目中禁用对齐检查, 如果项目中使用了pack对齐那么可以禁用该错误检查
add_definitions(-DWINDOWS_IGNORE_PACKING_MISMATCH)

file(
  GLOB_RECURSE SHARED_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SHARED_SOURCE_FILES})

# dependencies
# find_package(fmt 10.0.0 CONFIG REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt-header-only)

# rapidjson
find_package(RapidJSON CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC rapidjson)
if(rapidjson_FOUND)
  message(STATUS "rapidjson found")
endif()

# spdlog
find_package(spdlog CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog)
if(spdlog_FOUND)
  message(STATUS "spdlog found")
endif()

# OpenCV
# link: https://github.com/opencv/opencv
# 指定cmake model搜索路径
set(CMAKE_PREFIX_PATH "C:/tools/opencv-3.4.16/opencv/build")
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
if(OpenCV_FOUND)
  message(STATUS "OpenCV found")
endif()

# OpenVINO
# link: https://storage.openvinotoolkit.org
set(CMAKE_PREFIX_PATH "C:/tools/w_openvino_toolkit_windows_2023.3.0.13775.ceeafaf64f3_x86_64/runtime/cmake")
find_package(OpenVINO CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC openvino::runtime)
if(OpenVINO_FOUND)
  message(STATUS "OpenVINO found")
endif()

# header
target_include_directories(${PROJECT_NAME}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE CPPDLL_EXPORTS)
endif()                           