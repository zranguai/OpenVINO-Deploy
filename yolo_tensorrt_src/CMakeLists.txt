cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(yolo_tensorrt_src LANGUAGES CXX)

# 在项目中禁用对齐检查, 如果项目中使用了pack对齐那么可以禁用该错误检查
add_definitions(-DWINDOWS_IGNORE_PACKING_MISMATCH)

file(
  GLOB_RECURSE SHARED_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SHARED_SOURCE_FILES})

# dependencies

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Function)
# CUDA
find_package(CUDAToolkit REQUIRED)
print_var(CUDA_LIBRARIES)
print_var(CUDA_INCLUDE_DIRS)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
print_var(CUDA_LIB_DIR)


# OpenCV
# 指定cmake model搜索路径
set(CMAKE_PREFIX_PATH "C:/tools/opencv-3.4.16/opencv/build")
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
if(OpenCV_FOUND)
  message(STATUS "OpenCV found")
endif()


# TensorRT
find_package(TensorRT REQUIRED)
print_var(TensorRT_LIBRARIES)
print_var(TensorRT_INCLUDE_DIRS)
print_var(TensorRT_LIB_DIR)
if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUS "Build with -DTRT_10")
    add_definitions(-DTRT_10)
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PUBLIC ${TensorRT_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${TensorRT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC "C:\\tools\\TensorRT-8.6.1.6\\tensorRT\\lib\\nvinfer.lib")

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# if(WIN32)
#   target_compile_definitions(${PROJECT_NAME} PRIVATE CPPDLL_EXPORTS)
# endif()

add_subdirectory(cuda_lib)
target_link_libraries(${PROJECT_NAME} PUBLIC cuda_lib)
