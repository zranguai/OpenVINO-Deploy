#include "OpenVINO_ClassifyModel.h"

namespace Classify {

    bool Infer::load_from_memory(const std::string& model_file, int batch_size, int width_size, int height_size, int class_num)
    {
        try
        {
            ov::Core core;
            input_batch_ = batch_size;
            input_width_ = width_size;
            input_height_ = height_size;
            class_num_ = class_num;

            // 模型加载
            std::shared_ptr<ov::Model> model_object = core.read_model(model_file);
            // 更改模型形状
            const ov::Layout model_layout{ "NCHW" };
            tensor_shape_ = model_object->input().get_shape();
            tensor_shape_[ov::layout::batch_idx(model_layout)] = batch_size;
            tensor_shape_[ov::layout::channels_idx(model_layout)] = 3;
            tensor_shape_[ov::layout::height_idx(model_layout)] = height_size;
            tensor_shape_[ov::layout::width_idx(model_layout)] = width_size;
            model_object->reshape({ {model_object->input().get_any_name(), tensor_shape_} });
            // 模型加载并编译
            ov::CompiledModel compiled_model_object = core.compile_model(model_object, "CPU");
            // 创建用于推断已编译模型的推理请求对象  创建的请求分配了输入和输出张量
            infer_request_object_ = compiled_model_object.create_infer_request();
            return true;
        }
        catch (const std::exception& e)
        {
            std::string msg = "load_from_memory 捕获到异常: ";
            std::string strExcep = e.what();
            msg = msg + strExcep;
            std::cout << msg << std::endl;
            return false;
        }
    }

    std::vector<ConfInfoArray> Infer::forwards(const std::vector<Image>& images)
    {
        std::vector<ConfInfoArray> arrout(images.size());
        try
        {
            auto input_tensor = infer_request_object_.get_input_tensor(0);
            auto output_tensor = infer_request_object_.get_output_tensor(0);
            input_tensor.set_shape({ input_batch_, 3, input_width_, input_height_ }); // 指定shape的大小
            float* input_data_host = input_tensor.data<float>();  // 获取输入的地址，并传递给指针input_data_host
            std::vector<cv::Mat> blob_image_list;
            for (int i = 0; i < images.size(); i++)
            {
                Image srcImage = images[i];
                cv::Mat srcMat(srcImage.height, srcImage.width, CV_8UC3, (uchar*)srcImage.bgrptr);
                cv::Mat dstMat;
                cv::resize(srcMat, dstMat, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);
                cv::Mat flt_image;
                dstMat.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
                blob_image_list.push_back(flt_image);
            }

            for (size_t b = 0; b < input_batch_; b++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t h = 0; h < input_height_; h++) {
                        for (size_t w = 0; w < input_width_; w++) {
                            input_data_host[b * 3 * input_width_ * input_height_ + c * input_width_ * input_height_ + h * input_width_ + w] = blob_image_list[b].at<cv::Vec<float, 3>>(h, w)[c];
                        }
                    }
                }
            }
            infer_request_object_.infer();  // 模型推理
            float* prob_array = output_tensor.data<float>();
            for (int j = 0; j < input_batch_; j++)
            {
                float sum_prob = 0;
                float* prob = prob_array + j * class_num_;

                int predict_label = std::max_element(prob, prob + class_num_) - prob;  // 确定预测类别的下标

                for (int i = 0; i < class_num_; i++) {

                    sum_prob += exp(prob[i]);
                }

                float confidence = exp(prob[predict_label]) / sum_prob;    // 获得预测值的置信度
                ConfInfoArray& output = arrout[j];
                ConfInfo Conf(confidence, predict_label);
                output.emplace_back(Conf);
                prob = NULL;
                delete prob;
            }
            return arrout;
        }

        catch (const std::exception& e)
        {
            std::string msg = "forwards 捕获到异常: ";
            std::string strExcep = e.what();
            msg = msg + strExcep;
            std::cout << msg << std::endl;
            return arrout;
        }
    }

    Infer* loadraw_from_memory(const std::string& model_file, int batch_size, int width_size, int height_size, int class_num) {
        Infer* impl = new Infer();
        if (!impl->load_from_memory(model_file, batch_size, width_size, height_size, class_num)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load_from_memory(const std::string& model_file, int batch_size, int width_size, int height_size, int class_num)
    {
        return std::shared_ptr<Infer>(
            (Infer*)loadraw_from_memory(model_file, batch_size, width_size, height_size, class_num));
    }

    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
        const int h_i = static_cast<int>(h * 6);
        const float f = h * 6 - h_i;
        const float p = v * (1 - s);
        const float q = v * (1 - f * s);
        const float t = v * (1 - (1 - f) * s);
        float r, g, b;
        switch (h_i) {
        case 0:
            r = v, g = t, b = p;
            break;
        case 1:
            r = q, g = v, b = p;
            break;
        case 2:
            r = p, g = v, b = t;
            break;
        case 3:
            r = p, g = q, b = v;
            break;
        case 4:
            r = t, g = p, b = v;
            break;
        case 5:
            r = v, g = p, b = q;
            break;
        default:
            r = 1, g = 1, b = 1;
            break;
        }
        return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
            static_cast<uint8_t>(r * 255));
    }

    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
        float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
        float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
        return hsv2bgr(h_plane, s_plane, 1);
    }
}

