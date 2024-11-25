#pragma once

#ifndef __TENSORRT_CLASSIFYMODEL_H__
#define __TENSORRT_CLASSIFYMODEL_H__
#include "TensorRT_Infer.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>


namespace Classify {


    struct ConfInfo {
        float confidence;
        int class_label;


        ConfInfo() = default;
        ConfInfo(float confidence, int class_label)
            : confidence(confidence),
            class_label(class_label) {}
    };

    struct Image {
        const void* bgrptr = nullptr;
        int width = 0, height = 0;

        Image() = default;
        Image(const void* bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
    };

    typedef std::vector<ConfInfo> ConfInfoArray;

  
    class Infer {
    public:
        virtual ConfInfoArray forward(const Image& image, void* stream = nullptr) = 0;
        virtual std::vector<ConfInfoArray> forwards(const std::vector<Image>& images,
            void* stream = nullptr) = 0;
    };

    std::shared_ptr<Infer> load_from_memory(std::vector<uint8_t>& model_data,int class_num);

    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
};
#endif
