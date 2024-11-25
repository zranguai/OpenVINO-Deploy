cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(yolo_tensorrt_main CXX)

file(
  GLOB_RECURSE MAIN_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS src/*.c*)

add_executable(${PROJECT_NAME} ${MAIN_SOURCE_FILES})

  # link libraries
target_link_libraries(${PROJECT_NAME} PUBLIC yolo_tensorrt_src)