#include "OpenVINO_YoloModel.h"
namespace yolo {


    //模型初始化
    bool Infer::load_from_memory(const std::string& model_file, Type type, int batch_size, int width_size, int height_size, float confidence_threshold, float nms_threshold)
    {
        try
        {

            ov::Core core;
            input_batch_ = batch_size;
            input_width_ = width_size;
            input_height_ = height_size;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_ = nms_threshold;
            type_ = type;

            // 模型加载
            std::shared_ptr<ov::Model> model_object = core.read_model(model_file);
            // 更改模型形状
            const ov::Layout model_layout{ "NCHW" };
            tensor_shape_ = model_object->input().get_shape();
            tensor_shape_[ov::layout::batch_idx(model_layout)] = batch_size;
            tensor_shape_[ov::layout::channels_idx(model_layout)] = 3;
            tensor_shape_[ov::layout::height_idx(model_layout)] = height_size;
            tensor_shape_[ov::layout::width_idx(model_layout)] = width_size;
            model_object->reshape({ {model_object->input().get_any_name(),tensor_shape_} });
            // 模型加载并编译
            ov::CompiledModel compiled_mode_object = core.compile_model(model_object, "CPU");
            // 创建用于推断已编译模型的推理请求对象，创建的请求分配了输入和输出张量
            infer_request_object_ = compiled_mode_object.create_infer_request();
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

    InstanceSegmentMap::InstanceSegmentMap(int width, int height) {
        this->width = width;
        this->height = height;
        data = new unsigned char[width * height]();
    }

    InstanceSegmentMap::~InstanceSegmentMap() {
        if (this->data) {
            delete[] data;
            this->data = nullptr;
        }
        this->width = 0;
        this->height = 0;
    }


    cv::Mat PreprocessImg(cv::Mat& img, int input_h, int input_w, std::vector<int>& padsize) {
        int w, h, x, y;
        float r_w = input_w / (img.cols * 1.0);
        float r_h = input_h / (img.rows * 1.0);
        if (r_h > r_w) {//宽大于高
            w = input_w;
            h = r_w * img.rows;
            x = 0;
            y = (input_h - h) / 2;
        }
        else {
            w = r_h * img.cols;
            h = input_h;
            x = (input_w - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        padsize.push_back(h);
        padsize.push_back(w);
        padsize.push_back(y);
        padsize.push_back(x);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
        return out;
    }



    //模型推理加后处理 batch推理， yolov5，yolov8 yolov8seg
    std::vector<BoxArray> Infer::forwards(const std::vector<Image>& images)
    {
        std::vector<BoxArray> arrout(images.size());
        try
        {
            auto input_tensor = infer_request_object_.get_input_tensor(0);
            input_tensor.set_shape({ input_batch_, 3, input_width_, input_height_ }); // 指定shape的大小
            float* input_data_host = input_tensor.data<float>();  // 获取输入的地址，并传递给指针input_data_host
            std::vector<std::vector<int>> pad_sizes;
            std::vector<cv::Mat> src_mats;
            bool has_segment_ = false;
            has_segment_ = type_ == Type::V8Seg;
            int index = 0;
            for (int i = 0; i < images.size(); i++)
            {
                Image srcImage = images[i];
                cv::Mat srcMat(srcImage.height, srcImage.width, CV_8UC3, (uchar*)srcImage.bgrptr);
                
                src_mats.push_back(srcMat.clone());
                cv::Mat pr_img;
                std::vector<int> padsize;
                pr_img = PreprocessImg(srcMat, input_height_, input_width_, padsize);       // Resize
                int image_area = pr_img.cols * pr_img.rows;
                unsigned char* pimage = pr_img.data;
                float* tensor_num = input_data_host + image_area * 3 * i;
                float* phost_b = tensor_num + image_area * 0;   // input_data_host和phost_*进行地址关联
                float* phost_g = tensor_num + image_area * 1;
                float* phost_r = tensor_num + image_area * 2;
                for (int j = 0; j < image_area; ++j, pimage += 3) {
                    // 注意这里的顺序rgb调换了
                    *phost_r++ = (pimage[0] / 255.0f);  // 将图片中的像素点进行减去均值除方差，并赋值给input
                    *phost_g++ = (pimage[1] / 255.0f);
                    *phost_b++ = (pimage[2] / 255.0f);
                }
                pad_sizes.push_back(padsize);

            }

            infer_request_object_.infer();  // 模型推理

            int output_num;
            int output_pro;
            int Seg_channel;
            int Seg_height;
            int Seg_width;
            ov::Tensor output_det; 
            ov::Tensor output_seg;
            float* Det_ptr = NULL;
            float* Seg_ptr = NULL;

            output_det = infer_request_object_.get_output_tensor(0);
            output_num = output_det.get_shape()[1];
            output_pro = output_det.get_shape()[2];
            Det_ptr = output_det.data<float>();

            if (type_ == Type::V5Det)
            {
                class_num_ = output_pro - 5;
            }
            else if(type_ == Type::V8Det || type_ == Type::V8Seg)
            {
                class_num_ = output_pro - 4;
                if (has_segment_)
                {
                    output_seg = infer_request_object_.get_output_tensor(1);;
                    Seg_ptr = output_seg.data<float>();
                    Seg_channel = output_seg.get_shape()[1];
                    Seg_height = output_seg.get_shape()[2];
                    Seg_width = output_seg.get_shape()[3];
                    class_num_ = output_pro - 4 - Seg_channel;
                }
            }

            for (int i_batch = 0; i_batch < input_batch_; i_batch++)
            {
                int newh = pad_sizes[i_batch][0], neww = pad_sizes[i_batch][1],
                    padh = pad_sizes[i_batch][2], padw = pad_sizes[i_batch][3];
                float ratio_h = (float)src_mats[i_batch].rows / newh;
                float ratio_w = (float)src_mats[i_batch].cols / neww;
                
                std::vector<int> classIds;//结果id数组
                std::vector<float> confidences;//结果每个id对应置信度数组
                std::vector<cv::Rect> boxes;//每个id矩形框
                std::vector<std::vector<float>> picked_proposals;  //存储output0[:,:, 5 + _className.size():net_width]用以后续计算mask

                if (type_ == Type::V5Det)
                {
                    int net_width = output_pro;
                    float* ptr_data = Det_ptr + i_batch * output_num * output_pro;
                    std::cout << i_batch << std::endl;
                    for (int j = 0; j < output_num; ++j) {
                        float box_score = ptr_data[4]; ;//获取每一行的box框中含有某个物体的概率
                        if (box_score >= confidence_threshold_) {
                            cv::Mat scoresv5(1, class_num_, CV_32FC1, ptr_data + 5);
                            cv::Point classIdPoint;
                            double max_class_socre;
                            minMaxLoc(scoresv5, 0, &max_class_socre, 0, &classIdPoint);
                            max_class_socre = (float)max_class_socre;
                            if (max_class_socre >= confidence_threshold_) {
                                float x = (ptr_data[0] - padw) * ratio_w;  //x
                                float y = (ptr_data[1] - padh) * ratio_h;  //y
                                float w = ptr_data[2] * ratio_w;  //w
                                float h = ptr_data[3] * ratio_h;  //h
                                int left = MAX((x - 0.5 * w), 0);
                                int top = MAX((y - 0.5 * h), 0);
                                classIds.push_back(classIdPoint.x);
                                confidences.push_back(max_class_socre * box_score);
                                boxes.push_back(cv::Rect(left, top, int(w), int(h)));
                            }
                        }
                        ptr_data += net_width;//下一行
                    }
                    ptr_data = NULL;
                    delete ptr_data;
                    std::vector<int> nms_result;
                    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, nms_result);
                    BoxArray& output = arrout[i_batch];
                    output.reserve(nms_result.size());
                    cv::Mat src_origin = src_mats[i_batch];
                    cv::Rect holeImgRect(0, 0, src_origin.cols, src_origin.rows);
                    for (int i = 0; i < nms_result.size(); ++i) {
                        int idx = nms_result[i];

                        cv::Rect rect = boxes[idx] & holeImgRect;;
                        Box result_object_box(rect.tl().x, rect.tl().y, rect.br().x, rect.br().y, confidences[idx], classIds[idx]);
                        output.emplace_back(result_object_box);
                    }
                }
                else if (type_ == Type::V8Det || type_ == Type::V8Seg)
                {
                    int net_width = output_pro;
                    float* ptr_data = Det_ptr + i_batch * output_num * output_pro;
                    for (size_t i = 0; i < output_num; ++i) {

                        cv::Mat scoresv8(1, class_num_, CV_32FC1, ptr_data + 4);
                        cv::Point classIdPoint;
                        double max_class_socre;
                        minMaxLoc(scoresv8, 0, &max_class_socre, 0, &classIdPoint);
                        max_class_socre = (float)max_class_socre;
                        //std::cout << max_class_socre << std::endl;
                        if (max_class_socre >= confidence_threshold_) {
                            if (has_segment_)
                            {
                                std::vector<float> temp_proto(ptr_data + 4 + class_num_, ptr_data + net_width);
                                picked_proposals.push_back(temp_proto);
                            }
                            float x = (ptr_data[0] - padw) * ratio_w;  //x
                            float y = (ptr_data[1] - padh) * ratio_h;  //y
                            float w = ptr_data[2] * ratio_w;  //w
                            float h = ptr_data[3] * ratio_h;  //h
                            int left = MAX((x - 0.5 * w), 0);
                            int top = MAX((y - 0.5 * h), 0);
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back(max_class_socre);
                            boxes.push_back(cv::Rect(left, top, int(w), int(h)));
                        }
                        ptr_data += net_width;//下一行
                    }
                    ptr_data = NULL;
                    delete ptr_data;
                    std::vector<int> nms_result;
                    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, nms_result);
                    BoxArray& output = arrout[i_batch];
                    output.reserve(nms_result.size());
                    cv::Mat src_origin = src_mats[i_batch];
                    cv::Rect holeImgRect(0, 0, src_origin.cols, src_origin.rows);
                    float* seg_ptr_data;
                    cv::Mat mask_protos;
                    if (has_segment_)
                    {
                        seg_ptr_data = Seg_ptr + i_batch * Seg_channel * Seg_height * Seg_width;
                        mask_protos = cv::Mat(Seg_channel, Seg_height * Seg_width, CV_32F, seg_ptr_data);

                    }
                    for (int i = 0; i < nms_result.size(); ++i) {
                        int idx = nms_result[i];

                        cv::Rect rect = boxes[idx] & holeImgRect;;
                        Box result_object_box(rect.tl().x, rect.tl().y, rect.br().x, rect.br().y, confidences[idx], classIds[idx]);
                        if (has_segment_)
                        {
                            cv::Mat maskProposals = cv::Mat(picked_proposals[idx]).t();
                            cv::Mat protos = mask_protos.reshape(0, { Seg_channel,Seg_width * Seg_height });
                            if (maskProposals.cols == 0 || maskProposals.rows == 0) {

                                continue;
                            }
                            cv::Mat matmulRes = (maskProposals * protos).t();
                            cv::Mat masks = matmulRes.reshape(1, { Seg_width , Seg_height });

                            cv::Mat dest;
                            cv::exp(-masks, dest);
                            dest = 1.0f / (1.0f + dest);

                            cv::Rect roi(int((float)padw / input_width_ * Seg_width), int((float)padh / input_height_ * Seg_height), int(Seg_width - padw / 2), int(Seg_height - padh / 2));
                            dest = dest(roi);
                            cv::Mat img_mask;
                            cv::resize(dest, img_mask, src_origin.size(), cv::INTER_NEAREST);
                            cv::Rect temp_rect = cv::Rect{ rect.tl().x,rect.tl().y,rect.br().x - rect.tl().x,rect.br().y - rect.tl().y };
                            img_mask = img_mask(temp_rect) > 0.5f;

                            std::shared_ptr<InstanceSegmentMap> segPtr = std::make_shared<InstanceSegmentMap>(img_mask.cols, img_mask.rows);
                            std::memcpy(segPtr->data, img_mask.data, img_mask.cols* img_mask.rows);
                            result_object_box.seg = segPtr;  
                        }
                        output.emplace_back(result_object_box);
                    }
                    seg_ptr_data = NULL;
                    delete seg_ptr_data;
                }
            }
            Det_ptr = NULL;
            delete Det_ptr;
            Seg_ptr = NULL;
            delete Seg_ptr;
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

    Infer* loadraw_from_memory(const std::string& model_file, Type type, int batch_size, int width_size, int height_size, float confidence_threshold,
        float nms_threshold) {
        Infer* impl = new Infer();
        if (!impl->load_from_memory(model_file, type, batch_size, width_size, height_size, confidence_threshold, nms_threshold)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load_from_memory(const std::string& model_file, Type type,int batch_size,int width_size,int height_size, float confidence_threshold, float nms_threshold)
    {
        return std::shared_ptr<Infer>(
            (Infer*)loadraw_from_memory(model_file, type, batch_size, width_size, height_size, confidence_threshold, nms_threshold));
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
