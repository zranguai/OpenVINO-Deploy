#pragma once

#ifndef __OPENVINO_YOLOMODEL_H__
#define __OPENVINO_YOLOMODEL_H__
#include <openvino/openvino.hpp>
#include "OpenVINO_BaseStruct.h"
namespace yolo {

    enum class Type : int {
        V5Det = 0,
        V8Det = 1,
        V8Seg = 2  // yolov8 instance segmentation
    };

    struct InstanceSegmentMap {
        int width = 0, height = 0;      // width % 8 == 0
        unsigned char* data = nullptr;  // is width * height memory

        InstanceSegmentMap(int width, int height);
        virtual ~InstanceSegmentMap();
    };

    struct Box {
        float left, top, right, bottom, confidence;
        int class_label;
        std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label) {}
    };

    struct Image {
        const void* bgrptr = nullptr;
        int width = 0, height = 0;

        Image() = default;
        Image(const void* bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
    };

    typedef std::vector<Box> BoxArray;

    class Infer
    {
    public:
        std::vector<BoxArray> forwards(const std::vector<Image>& images);
        bool load_from_memory(const std::string& model_file, Type type, int batch_size, int width_size, int height_size, float confidence_threshold, float nms_threshold);
    private:
        ov::Shape tensor_shape_;
        ov::InferRequest infer_request_object_;
        float confidence_threshold_;
        float nms_threshold_;
        size_t input_height_;
        size_t input_width_;
        size_t input_batch_;
        int class_num_;
        Type type_;
    };

    cv::Mat PreprocessImg(cv::Mat& img, int input_h, int input_w, std::vector<int>& padsize);

    std::shared_ptr<Infer> load_from_memory(const std::string& model_file, Type type, int batch_size, int width_size, int height_size,float confidence_threshold, float nms_threshold);
    


    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

    
};
#endif
