#include "TensorRT_Infer.hpp"
namespace trt {

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

#define Assert(op)                 \
  do {                             \
    bool cond = !(!(op));          \
    if (!cond) {                   \
      INFO("Assert failed, " #op); \
      abort();                     \
    }                              \
  } while (0)

#define Assertf(op, ...)                             \
  do {                                               \
    bool cond = !(!(op));                            \
    if (!cond) {                                     \
      INFO("Assert failed, " #op " : " __VA_ARGS__); \
      abort();                                       \
    }                                                \
  } while (0)

    static std::string file_name(const std::string& path, bool include_suffix) {
        if (path.empty()) return "";

        int p = path.rfind('/');
        int e = path.rfind('\\');
        p = std::max(p, e);
        p += 1;

        // include suffix
        if (include_suffix) return path.substr(p);

        int u = path.rfind('.');
        if (u == -1) return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

    void __log_func(const char* file, int line, const char* fmt, ...) {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        std::string filename = file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
        fprintf(stdout, "%s\n", buffer);
    }

    static std::string format_shape(const nvinfer1::Dims& shape) {
        std::stringstream output;
        char buf[64];
        const char* fmts[] = { "%d", "x%d" };
        for (int i = 0; i < shape.nbDims; ++i) {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
            output << buf;
        }
        return output.str();
    }

    Timer::Timer() {
        checkRuntime(cudaEventCreate((cudaEvent_t*)&start_));
        checkRuntime(cudaEventCreate((cudaEvent_t*)&stop_));
    }

    Timer::~Timer() {
        checkRuntime(cudaEventDestroy((cudaEvent_t)start_));
        checkRuntime(cudaEventDestroy((cudaEvent_t)stop_));
    }

    void Timer::start(void* stream) {
        stream_ = stream;
        checkRuntime(cudaEventRecord((cudaEvent_t)start_, (cudaStream_t)stream_));
    }

    float Timer::stop(const char* prefix, bool print) {
        checkRuntime(cudaEventRecord((cudaEvent_t)stop_, (cudaStream_t)stream_));
        checkRuntime(cudaEventSynchronize((cudaEvent_t)stop_));

        float latency = 0;
        checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t)start_, (cudaEvent_t)stop_));

        if (print) {
            printf("[%s]: %.5f ms\n", prefix, latency);
        }
        return latency;
    }

    BaseMemory::BaseMemory(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes) {
        reference(cpu, cpu_bytes, gpu, gpu_bytes);
    }

    void BaseMemory::reference(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes) {
        release();

        if (cpu == nullptr || cpu_bytes == 0) {
            cpu = nullptr;
            cpu_bytes = 0;
        }

        if (gpu == nullptr || gpu_bytes == 0) {
            gpu = nullptr;
            gpu_bytes = 0;
        }

        this->cpu_ = cpu;
        this->cpu_capacity_ = cpu_bytes;
        this->cpu_bytes_ = cpu_bytes;
        this->gpu_ = gpu;
        this->gpu_capacity_ = gpu_bytes;
        this->gpu_bytes_ = gpu_bytes;

        this->owner_cpu_ = !(cpu && cpu_bytes > 0);
        this->owner_gpu_ = !(gpu && gpu_bytes > 0);
    }

    BaseMemory::~BaseMemory() { release(); }

    void* BaseMemory::gpu_realloc(size_t bytes) {
        if (gpu_capacity_ < bytes) {
            release_gpu();

            gpu_capacity_ = bytes;
            checkRuntime(cudaMalloc(&gpu_, bytes));
            // checkRuntime(cudaMemset(gpu_, 0, size));
        }
        gpu_bytes_ = bytes;
        return gpu_;
    }

    void* BaseMemory::cpu_realloc(size_t bytes) {
        if (cpu_capacity_ < bytes) {
            release_cpu();

            cpu_capacity_ = bytes;
            checkRuntime(cudaMallocHost(&cpu_, bytes));
            Assert(cpu_ != nullptr);
            // memset(cpu_, 0, size);
        }
        cpu_bytes_ = bytes;
        return cpu_;
    }

    void BaseMemory::release_cpu() {
        if (cpu_) {
            if (owner_cpu_) {
                checkRuntime(cudaFreeHost(cpu_));
            }
            cpu_ = nullptr;
        }
        cpu_capacity_ = 0;
        cpu_bytes_ = 0;
    }

    void BaseMemory::release_gpu() {
        if (gpu_) {
            if (owner_gpu_) {
                checkRuntime(cudaFree(gpu_));
            }
            gpu_ = nullptr;
        }
        gpu_capacity_ = 0;
        gpu_bytes_ = 0;
    }

    void BaseMemory::release() {
        release_cpu();
        release_gpu();
    }

    class __native_nvinfer_logger : public nvinfer1::ILogger {
    public:
        virtual void log(Severity severity, const char* msg) noexcept override {
            if (severity == Severity::kINTERNAL_ERROR) {
                INFO("NVInfer INTERNAL_ERROR: %s", msg);
                abort();
            }
            else if (severity == Severity::kERROR) {
                INFO("NVInfer: %s", msg);
            }
            // else  if (severity == Severity::kWARNING) {
            //     INFO("NVInfer: %s", msg);
            // }
            // else  if (severity == Severity::kINFO) {
            //     INFO("NVInfer: %s", msg);
            // }
            // else {
            //     INFO("%s", msg);
            // }
        }
    };
    static __native_nvinfer_logger gLogger;

    template <typename _T>
    static void destroy_nvidia_pointer(_T* ptr) {
        if (ptr) ptr->destroy();
    }

    static std::vector<uint8_t> load_file(const std::string& file) {
        std::ifstream in(file, std::ios::in | std::ios::binary);
        if (!in.is_open()) return {};

        in.seekg(0, std::ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0) {
            in.seekg(0, std::ios::beg);
            data.resize(length);

            in.read((char*)&data[0], length);
        }
        in.close();
        return data;
    }

    class __native_engine_context {
    public:
        virtual ~__native_engine_context() { destroy(); }

        bool construct(const void* pdata, size_t size) {
            destroy();

            if (pdata == nullptr || size == 0) return false;

            runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger), destroy_nvidia_pointer<nvinfer1::IRuntime>);
            if (runtime_ == nullptr) return false;

            engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr),
                destroy_nvidia_pointer<nvinfer1::ICudaEngine>);
            if (engine_ == nullptr) return false;

            context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                destroy_nvidia_pointer<nvinfer1::IExecutionContext>);
            return context_ != nullptr;
        }

    private:
        void destroy() {
            context_.reset();
            engine_.reset();
            runtime_.reset();
        }

    public:
        std::shared_ptr<nvinfer1::IExecutionContext> context_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    };

    class InferImpl : public Infer {
    public:
        std::shared_ptr<__native_engine_context> context_;
        std::unordered_map<std::string, int> binding_name_to_index_;

        virtual ~InferImpl() = default;

        bool construct(const void* data, size_t size) {
            context_ = std::make_shared<__native_engine_context>();
            if (!context_->construct(data, size)) {
                return false;
            }

            setup();
            return true;
        }

        bool load(const std::string& file) {
            auto data = load_file(file);
            if (data.empty()) {
                INFO("An empty file has been loaded. Please confirm your file path: %s", file.c_str());
                return false;
            }
            return this->construct(data.data(), data.size());
        }


        bool load_from_memory(std::vector<uint8_t>& data) {
            // auto data = load_file(file);
            if (data.empty()) {
                INFO("An empty file has been loaded. Please confirm your file path: %s", data);
                return false;
            }
            return this->construct(data.data(), data.size());
        }

        void setup() {
            auto engine = this->context_->engine_;
            int nbBindings = engine->getNbBindings();

            binding_name_to_index_.clear();
            for (int i = 0; i < nbBindings; ++i) {
                const char* bindingName = engine->getBindingName(i);
                binding_name_to_index_[bindingName] = i;
            }
        }

        virtual int index(const std::string& name) override {
            auto iter = binding_name_to_index_.find(name);
            Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s",
                name.c_str());
            return iter->second;
        }

        virtual bool forward(const std::vector<void*>& bindings, void* stream,
            void* input_consum_event) override {
            return this->context_->context_->enqueueV2((void**)bindings.data(), (cudaStream_t)stream,
                (cudaEvent_t*)input_consum_event);
        }

        virtual std::vector<int> run_dims(const std::string& name) override {
            return run_dims(index(name));
        }

        virtual std::vector<int> run_dims(int ibinding) override {
            auto dim = this->context_->context_->getBindingDimensions(ibinding);
            return std::vector<int>(dim.d, dim.d + dim.nbDims);
        }

        virtual std::vector<int> static_dims(const std::string& name) override {
            return static_dims(index(name));
        }

        virtual std::vector<int> static_dims(int ibinding) override {
            auto dim = this->context_->engine_->getBindingDimensions(ibinding);
            return std::vector<int>(dim.d, dim.d + dim.nbDims);
        }

        virtual int num_bindings() override { return this->context_->engine_->getNbBindings(); }

        virtual bool is_input(int ibinding) override {
            return this->context_->engine_->bindingIsInput(ibinding);
        }

        virtual bool set_run_dims(const std::string& name, const std::vector<int>& dims) override {
            return this->set_run_dims(index(name), dims);
        }

        virtual bool set_run_dims(int ibinding, const std::vector<int>& dims) override {
            nvinfer1::Dims d;
            memcpy(d.d, dims.data(), sizeof(int) * dims.size());
            d.nbDims = dims.size();
            return this->context_->context_->setBindingDimensions(ibinding, d);
        }

        virtual int numel(const std::string& name) override { return numel(index(name)); }

        virtual int numel(int ibinding) override {
            auto dim = this->context_->context_->getBindingDimensions(ibinding);
            return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int>());
        }

        virtual DType dtype(const std::string& name) override { return dtype(index(name)); }

        virtual DType dtype(int ibinding) override {
            return (DType)this->context_->engine_->getBindingDataType(ibinding);
        }

        virtual bool has_dynamic_dim() override {
            // check if any input or output bindings have dynamic shapes
            // code from ChatGPT
            int numBindings = this->context_->engine_->getNbBindings();
            for (int i = 0; i < numBindings; ++i) {
                nvinfer1::Dims dims = this->context_->engine_->getBindingDimensions(i);
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (dims.d[j] == -1) return true;
                }
            }
            return false;
        }

        virtual void print() override {
            INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

            int num_input = 0;
            int num_output = 0;
            auto engine = this->context_->engine_;
            for (int i = 0; i < engine->getNbBindings(); ++i) {
                if (engine->bindingIsInput(i))
                    num_input++;
                else
                    num_output++;
            }

            INFO("Inputs: %d", num_input);
            for (int i = 0; i < num_input; ++i) {
                auto name = engine->getBindingName(i);
                auto dim = engine->getBindingDimensions(i);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }

            INFO("Outputs: %d", num_output);
            for (int i = 0; i < num_output; ++i) {
                auto name = engine->getBindingName(i + num_input);
                auto dim = engine->getBindingDimensions(i + num_input);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }
        }
    };

    Infer* loadraw(const std::string& file) {
        InferImpl* impl = new InferImpl();
        if (!impl->load(file)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load(const std::string& file) {
        return std::shared_ptr<InferImpl>((InferImpl*)loadraw(file));
    }



    Infer* loadraw_from_memory(std::vector<uint8_t>& model_data) {
        InferImpl* impl = new InferImpl();
        if (!impl->load_from_memory(model_data)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load_from_memory(std::vector<uint8_t>& model_data) {
        return std::shared_ptr<InferImpl>((InferImpl*)loadraw_from_memory(model_data));
    }



    std::string format_shape(const std::vector<int>& shape) {
        std::stringstream output;
        char buf[64];
        const char* fmts[] = { "%d", "x%d" };
        for (int i = 0; i < (int)shape.size(); ++i) {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape[i]);
            output << buf;
        }
        return output.str();
    }
};