#pragma once

#ifndef __OPENVINO_CLASSIFYMODEL_H__
#define __OPENVINO_CLASSIFYMODEL_H__
#include <openvino/openvino.hpp>
#include "OpenVINO_BaseStruct.h"
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

    class Infer
    {
    public:
        std::vector<ConfInfoArray> forwards(const std::vector<Image>& images);
        bool load_from_memory(const std::string& model_file, int batch_size, int width_size, int height_size, int class_num);
    private:
        ov::Shape tensor_shape_;
        ov::InferRequest infer_request_object_;
        size_t input_height_;
        size_t input_width_;
        size_t input_batch_;
        int class_num_;
        

    };



    std::shared_ptr<Infer> load_from_memory(const std::string& model_file, int batch_size, int width_size, int height_size, int class_num);


    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);


};
#endif
