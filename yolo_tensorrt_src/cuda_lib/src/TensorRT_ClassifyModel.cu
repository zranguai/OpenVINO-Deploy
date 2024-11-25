#include "TensorRT_ClassifyModel.hpp"
namespace Classify {

    //using namespace std;

#define GPU_BLOCK_THREADS 512
#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

    enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

    enum class ChannelType : int { None = 0, SwapRB = 1 };

    /* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
    struct Norm {
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
            ChannelType channel_type = ChannelType::None);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

        // None
        static Norm None();
    };

    Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
        ChannelType channel_type) {
        Norm out;
        out.type = NormType::MeanStd;
        out.alpha = alpha;
        out.channel_type = channel_type;
        memcpy(out.mean, mean, sizeof(out.mean));
        memcpy(out.std, std, sizeof(out.std));
        return out;
    }

    Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type) {
        Norm out;
        out.type = NormType::AlphaBeta;
        out.alpha = alpha;
        out.beta = beta;
        out.channel_type = channel_type;
        return out;
    }

    Norm Norm::None() { return Norm(); }

    const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class,
                                    // keepflag, row_index(output)
    const int MAX_IMAGE_BOXES = 1024;
    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }
    static __host__ __device__ void affine_project(float* matrix, float x, float y, float* ox,
        float* oy) {
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    

    static dim3 grid_dims(int numJobs) {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
    }

    static dim3 block_dims(int numJobs) {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }

    

    static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
        uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width,
        int dst_height, uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm) {
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx >= dst_width || dy >= dst_height) return;

        float m_x1 = warp_affine_matrix_2_3[0];
        float m_y1 = warp_affine_matrix_2_3[1];
        float m_z1 = warp_affine_matrix_2_3[2];
        float m_x2 = warp_affine_matrix_2_3[3];
        float m_y2 = warp_affine_matrix_2_3[4];
        float m_z2 = warp_affine_matrix_2_3[5];

        float src_x = m_x1 * dx + m_y1 * dy + m_z1;
        float src_y = m_x2 * dx + m_y2 * dy + m_z2;
        float c0, c1, c2;

        if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
            // out of range
            c0 = const_value_st;
            c1 = const_value_st;
            c2 = const_value_st;
        }
        else {  // 双线性插值计算
            int y_low = floorf(src_y);
            int x_low = floorf(src_x);
            int y_high = y_low + 1;
            int x_high = x_low + 1;
            uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
            float ly = src_y - y_low;
            float lx = src_x - x_low;
            float hy = 1 - ly;
            float hx = 1 - lx;
            float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            uint8_t* v1 = const_value;
            uint8_t* v2 = const_value;
            uint8_t* v3 = const_value;
            uint8_t* v4 = const_value;
            if (y_low >= 0) {
                if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;
                if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
            }
            if (y_high < src_height) {
                if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;
                if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
            }
            // same to opencv
            c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
            c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
            c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
        }

        if (norm.channel_type == ChannelType::SwapRB) {
            float t = c2;
            c2 = c0;
            c0 = t;
        }

        if (norm.type == NormType::MeanStd) {
            c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
            c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
            c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
        }
        else if (norm.type == NormType::AlphaBeta) {
            c0 = c0 * norm.alpha + norm.beta;
            c1 = c1 * norm.alpha + norm.beta;
            c2 = c2 * norm.alpha + norm.beta;
        }

        int area = dst_width * dst_height;
        float* pdst_c0 = dst + dy * dst_width + dx;
        float* pdst_c1 = pdst_c0 + area;
        float* pdst_c2 = pdst_c1 + area;
        *pdst_c0 = c0;
        *pdst_c1 = c1;
        *pdst_c2 = c2;
    }

    // 利用CUDA工具处理的能力，将输入图像进行仿射变换、双线性插值处理，把结果归一化到输出图像中
    static void warp_affine_bilinear_and_normalize_plane(uint8_t* src, int src_line_size, int src_width,
        int src_height, float* dst, int dst_width,
        int dst_height, float* matrix_2_3,
        uint8_t const_value, const Norm& norm,
        cudaStream_t stream) {
        dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);  // CUDA中的三维图形支持类型，用于定义网格和块的维度
        dim3 block(32, 32);  // 计算网格的维度，根据输出图像的宽度和高度计算所需的块数量，每个CUDA块的维度设置为32x32，表示每个块处理1024个线程

        checkKernel(warp_affine_bilinear_and_normalize_plane_kernel << <grid, block, 0, stream >> > (
            src, src_line_size, src_width, src_height, dst, dst_width, dst_height, const_value,
            matrix_2_3, norm));
    }






    struct AffineMatrix {
        float i2d[6];  // image to dst(network), 2x3 matrix
        float d2i[6];  // dst to image, 2x3 matrix

        void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to) {
            float scale_x = std::get<0>(to) / (float)std::get<0>(from);
            float scale_y = std::get<1>(to) / (float)std::get<1>(from);
            float scale = std::min(scale_x, scale_y);
            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 + scale * 0.5 - 0.5;

            double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            D = D != 0. ? double(1.) / D : double(0.);
            double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
            double b1 = -A11 * i2d[2] - A12 * i2d[5];
            double b2 = -A21 * i2d[2] - A22 * i2d[5];

            d2i[0] = A11;
            d2i[1] = A12;
            d2i[2] = b1;
            d2i[3] = A21;
            d2i[4] = A22;
            d2i[5] = b2;
        }
    };

    // 定义InferImpl类，继承Infer接口
    class InferImpl : public Infer {
    public:
        std::shared_ptr<trt::Infer> trt_;  // 指向TensorRT推理引擎的智能指针
        std::string engine_file_;
        int class_num;
        std::vector<std::shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;  // 用于推理之前预处理图像的缓冲区
        trt::Memory<float> input_buffer_, prob_predict_, output_probarray_;  // 输入数据和预测的缓冲区
        int network_input_width_, network_input_height_;  // 模型输入的维度
        Norm normalize_;  // 用于规范化输入图像的对象
        std::vector<int> class_dim;  // 输出类别的维度
        bool isdynamic_model_ = false;  // 指示模型是否可以处理动态批量大小的标志


        virtual ~InferImpl() = default;  // 默认析构函数；由于智能指针，TensorRT 对象将被自动清理

        // 根据给定的批次大小为输入和输出缓冲区分配内存,
        // 它检查的大小是否preprocess_buffers_小于batch_size，如果是，则向Memory对象添加新的共享指针
        void adjust_memory(int batch_size) {
            // the inference batch_size
            size_t input_numel = network_input_width_ * network_input_height_ * 3;
        


            input_buffer_.gpu(batch_size * input_numel);
            prob_predict_.gpu(batch_size * class_num);
            prob_predict_.cpu(batch_size * class_num);
            //output_probarray_.gpu(batch_size * (32 + 1));
            //output_probarray_.cpu(batch_size * (32 + 1));
            
            std::cout << batch_size * class_num * 3 << std::endl;
            if ((int)preprocess_buffers_.size() < batch_size) {
                for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                    preprocess_buffers_.push_back(std::make_shared<trt::Memory<unsigned char>>());
            }
        }

        // 准备图像以输入到模型中
        // 计算仿射变换以调整图像大小并对其进行标准化，然后将图像数据复制到GPU内存
        // 调用warp_affine_bilinear_and_normalize_plane，它可能处理图像以便为模型做准备
        void preprocess(int ibatch, const Image& image,
            std::shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
            void* stream = nullptr) {
            affine.compute(std::make_tuple(image.width, image.height),  // 图像计算的仿射变换，准备调整大小
                std::make_tuple(network_input_width_, network_input_height_));  // 目标尺寸为网络输入的宽度和高度

            // 上述compute通过输入图像尺寸和模型输入尺寸，得到映射关系

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            float* input_device = input_buffer_.gpu() + ibatch * input_numel;  // 计算输入的元素数量，并获取当前批次的输入设备指针
            size_t size_image = image.width * image.height * 3;  // 图像计算的总字节数
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);  // 计算仿射矩阵的大小，可能要对齐到32字节
            uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);  // 在GPU上为图像和分配矩阵工作空间
            // 获取设备指针，分别指向仿射矩阵和图像数据
            float* affine_matrix_device = (float*)gpu_workspace;  
            uint8_t* image_device = gpu_workspace + size_matrix;  // 获取设备指针，分别指向仿射矩阵和图像数据

            uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);  // 在CPU上为图像和分配矩阵工作空间
            // 获取CPU指针，分别指向仿射矩阵和图像数据
            float* affine_matrix_host = (float*)cpu_workspace;
            uint8_t* image_host = cpu_workspace + size_matrix;

            // speed up
            // 将输入的流指针转换为CUDA流
            cudaStream_t stream_ = (cudaStream_t)stream;
            // 将图像和仿射矩阵数据从主机复制到CPU工作空间
            memcpy(image_host, image.bgrptr, size_image);
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
            // 异步将图像数据从CPU复制到GPU
            checkRuntime(
                cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            // 平行将仿射矩阵数据从CPU复制到GPU
            checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                cudaMemcpyHostToDevice, stream_));
            // 调用函数进行仿射变换和归一化，将处理后的图像存储到输入设备指针
            warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
                image.height, input_device, network_input_width_,
                network_input_height_, affine_matrix_device, 114,
                normalize_, stream_);
        }


        // 从内存缓冲区加载模型并设置各种参数，包括输入维度和规范化
        // 检查模型是否加载成功
        bool load_from_memory(std::vector<uint8_t>& model_data, int class_num) {
            trt_ = trt::load_from_memory(model_data);
            if (trt_ == nullptr) return false;

            trt_->print();


            this->class_num = class_num;

            auto input_dim = trt_->static_dims(0);  // 获取输入的维度
            class_dim = trt_->static_dims(1);  // 获取类别的维度
            network_input_width_ = input_dim[3];
            network_input_height_ = input_dim[2];

           


            isdynamic_model_ = trt_->has_dynamic_dim();  // 检查模型是否支持动态维度
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);  // 设置图像标准化的参数，将像素值缩放到 [0, 1] 之间并进行通道转换
            return true;
        }


        // 推理
        virtual ConfInfoArray forward(const Image& image, void* stream = nullptr) override {
            auto output = forwards({ image }, stream);
            if (output.empty()) return {};
            return output[0];
        }


        // 推理函数
        virtual std::vector<ConfInfoArray> forwards(const std::vector<Image>& images, void* stream = nullptr) override {
            int num_image = images.size();
            if (num_image == 0) return {};

            // 从TensorRT引擎获取输入的静态维度，并获取批次大小
            auto input_dims = trt_->static_dims(0);
            int infer_batch_size = input_dims[0];

            if (infer_batch_size != num_image) {
                if (isdynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(0, input_dims)) return {};
                    
                }
                else {
                    if (infer_batch_size < num_image) {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, infer_batch_size);
                        return {};
                    }
                }
            }
            adjust_memory(infer_batch_size);  // 调整内存，确保推理分配有足够的输入和输出
            
            std::vector<AffineMatrix> affine_matrixs(num_image);  // 创建一个AffineMatrix对象的管理，用于存储每张图像的仿射变换矩阵
            cudaStream_t stream_ = (cudaStream_t)stream;  // 将输入的流转换为CUDA流，以便在后续操作中使用

            // 遍历所有输入图像，调用preprocess函数处理每张图像，包括调整大小和归一化
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            // 获取输出预测的GPU指针，并创建一个处理bindings,将输入和输出坐标的指针存储在其中，及时传递给TensorRT的推理函数。
            float* prob_output_device = prob_predict_.gpu();
            std::vector<void*> bindings{ input_buffer_.gpu(), prob_output_device };

  

            if (!trt_->forward(bindings, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            // 异步将预测结果从 GPU 复制到 CPU 内存
            checkRuntime(cudaMemcpyAsync(prob_predict_.cpu(), prob_predict_.gpu(),
                prob_predict_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
            checkRuntime(cudaStreamSynchronize(stream_));  // 确认所有操作完成，等待CUDA流同步

            std::vector<ConfInfoArray> arrout(num_image);  // 创建一个结果数据库，用于存储每张图像的预测信息
            //int imemory = 0;
            for (int ib = 0; ib < num_image; ++ib) {
                float* parray = prob_predict_.cpu() + ib * (this->class_num);  // 遍历每张图像，从CPU预测结果中获取当前图像的概率吞吐量

                int predict_label = std::max_element(parray, parray + this->class_num) - parray;  // 确定预测类别的下标

                float sum_prob = 0;
                for (int i = 0; i < this->class_num; i++) {
                    sum_prob += exp(parray[i]);
                }
                //auto predict_name = _class_names[predict_label];

                float confidence = exp(parray[predict_label]) / sum_prob; // 获得预测值的置信度
                ConfInfoArray& output = arrout[ib];
                ConfInfo Conf(confidence, predict_label);
                output.emplace_back(Conf);
            }
            return arrout;
        }
    };


    Infer* loadraw_from_memory(std::vector<uint8_t>& model_data, int class_num) {
        InferImpl* impl = new InferImpl();
        if (!impl->load_from_memory(model_data, class_num)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }






    std::shared_ptr<Infer> load_from_memory(std::vector<uint8_t>& model_data,int class_num) {
        return std::shared_ptr<InferImpl>(
            (InferImpl*)loadraw_from_memory(model_data, class_num));
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

};