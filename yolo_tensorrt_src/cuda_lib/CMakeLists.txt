project(cuda_lib CUDA)

file(
  GLOB_RECURSE CUDALIB_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS src/*.c*)

add_library(${PROJECT_NAME} STATIC ${CUDALIB_SOURCE_FILES})

# CUDA Toolkit
target_include_directories(${PROJECT_NAME} SYSTEM
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${PROJECT_NAME}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TensorRT
set(TENSORRT_PATH "C:/tools/TensorRT-8.6.1.6/tensorRT")
set(TENSORRT_LIB_DIR "C:/tools/TensorRT-8.6.1.6/tensorRT/lib")
target_include_directories(${PROJECT_NAME} PUBLIC ${TENSORRT_PATH}/include)
link_directories(${TENSORRT_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC nvinfer nvinfer_plugin nvonnxparser)

# Add CUDA compiler options
target_compile_options(${PROJECT_NAME} PRIVATE -lineinfo -use_fast_math
                                               -Xptxas=-v)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                 CUDA_RESOLVE_DEVICE_SYMBOLS ON)