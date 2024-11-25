#[=======================================================================[
FindTensorRT
------------

Input Variables (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^
The following variables are optionally searched for defaults

``TENSORRT_ROOT_PATH``
  Base directory where TensorRT is found.
  If this variable is not provided, default paths are searched.


Result Variables
^^^^^^^^^^^^^^^^
The script defines the following variables:

``TENSORRT_VERSION``
``TENSORRT_ROOT_DIR`` (optional)
``TENSORRT_INCLUDE_DIR``
``TENSORRT_LIBRARY_DIR``

``TENSORRT_nvinfer_LIBRARY``
``TENSORRT_nvinfer_static_LIBRARY``
``TENSORRT_nvinfer_plugin_LIBRARY``
``TENSORRT_nvinfer_plugin_static_LIBRARY``
``TENSORRT_nvonnxparser_LIBRARY``
``TENSORRT_nvonnxparser_static_LIBRARY``
``TENSORRT_nvparsers_LIBRARY``
``TENSORRT_nvparsers_static_LIBRARY``
``TENSORRT_nvcaffe_parser_LIBRARY``
``TENSORRT_nvinfer_dispatch_LIBRARY``
``TENSORRT_nvinfer_dispatch_static_LIBRARY``
``TENSORRT_nvinfer_lean_LIBRARY``
``TENSORRT_nvinfer_lean_static_LIBRARY``

``TENSORRT_VERSION_MAJOR`` INTERNAL
``TENSORRT_VERSION_MINOR`` INTERNAL
``TENSORRT_VERSION_PATCH`` INTERNAL

#]=======================================================================]
unset(TENSORRT_ROOT_DIR CACHE)
unset(TENSORRT_VERSION_MAJOR CACHE)
unset(TENSORRT_VERSION_MINOR CACHE)
unset(TENSORRT_VERSION_PATCH CACHE)
unset(TENSORRT_VERSION CACHE)
unset(TENSORRT_INCLUDE_DIR CACHE)
unset(TENSORRT_LIBRARY_DIR CACHE)

# tensorrt root directory
set(path_hint)
if (DEFINED TENSORRT_ROOT_PATH)
    list(APPEND path_hint ${TENSORRT_ROOT_PATH})
else()
    lisT(APPEND path_hint $ENV{LD_LIBRARY_PATH})
endif()
set(TENSORRT_ROOT_DIR ${TENSORRT_ROOT_PATH} CACHE PATH "Folder containing NVIDIA TensorRT")

# find tensorrt include directory
find_path(TENSORRT_INCLUDE_DIR
    NvInfer.h
    PATHS ${path_hint}
    PATH_SUFFIXES include
    DOC "Folder containing NVIDIA TensorRT header files"
)

# extract version from the include
if (EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" trt_version_content)

    string(REGEX MATCH "define NV_TENSORRT_MAJOR ([0-9]+)" _ "${trt_version_content}")
    set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "TensorRT Version Major")
    string(REGEX MATCH "define NV_TENSORRT_MINOR ([0-9]+)" _ "${trt_version_content}")
    set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1} CACHE INTERNAL "TensorRT Version Minor")
    string(REGEX MATCH "define NV_TENSORRT_PATCH ([0-9]+)" _ "${trt_version_content}")
    set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1} CACHE INTERNAL "TensorRT Version Patch")

    set(TENSORRT_VERSION
        "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}"
        CACHE STRING "TensorRT Library Version"
    )
endif()

# find tensorrt library directory
if (TENSORRT_ROOT_DIR)
    set(TENSORRT_LIBRARY_DIR "${TENSORRT_ROOT_PATH}/lib" CACHE PATH "Path to the TensorRT library files (e.g., libnvinfer1.so)")
else()
    find_library(_nvinfer_library
        nvinfer
        PATHS ${path_hint}
        NO_CACHE
    )
    get_filename_component(TENSORRT_LIBRARY_DIR ${_nvinfer_library} DIRECTORY)
    set(TENSORRT_LIBRARY_DIR ${TENSORRT_LIBRARY_DIR} CACHE PATH "Path to the TensorRT library files (e.g., libnvinfer1.so)")
    unset(_nvinfer_library CACHE)
endif()

macro(tensorrt_find_library _var _names _doc)
    find_library(${_var}
        NAMES ${_names}
        PATHS ${TENSORRT_LIBRARY_DIR}
        DOC ${_doc}
        NO_DEFAULT_PATH
    )
endmacro()

macro(FIND_TENSORRT_LIBS _name)
    tensorrt_find_library(TENSORRT_${_name}_LIBRARY ${_name} "\"${_name}\" library")
    mark_as_advanced(TENSORT_${_name}_LIBRARY)
endmacro()

# find tensorrt libraries
set(tensorrt_libs_list
        nvinfer
        nvinfer_static
        nvinfer_plugin
        nvinfer_plugin_static
        nvonnxparser
        nvonnxparser_static
        nvparsers
        nvparsers_static
        nvcaffe_parser
)
if (TENSORRT_VERSION VERSION_GREATER_EQUAL 8.6.0)
    list(APPEND tensorrt_libs_list
        nvinfer_dispatch
        nvinfer_dispatch_static
        nvinfer_lean
        nvinfer_lean_static
    )
endif()

foreach(lib ${tensorrt_libs_list})
    unset(TENSORRT_${lib}_LIBRARY CACHE)
    FIND_TENSORRT_LIBS(${lib})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TENSORRT
    FOUND_VAR TENSORRT_FOUND
    REQUIRED_VARS
        TENSORRT_INCLUDE_DIR
        TENSORRT_LIBRARY_DIR
    VERSION_VAR
        TENSORRT_VERSION
)

mark_as_advanced(
    TENSORRT_ROOT_DIR
    TENSORRT_INCLUDE_DIR
    TENSORRT_LIBRARY_DIR
    TENSORRT_VERSION
)