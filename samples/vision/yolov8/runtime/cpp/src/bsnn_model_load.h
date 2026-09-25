#ifndef __BSNN_MODEL_LOAD__
#define __BSNN_MODEL_LOAD__

#include "bsnn.h"
#include "bsnn_engine.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <vector>

class BSNN_MODEL
{
public:
    BSNN_MODEL(const std::string& model_dir, uint32_t batch_size = 1,
               int target_engine = BSNN_CORE_MASK_AUTO)
        : model_dir_(model_dir), target_engine_(target_engine), batch_size_(batch_size)
    {
        InitModel(model_dir_, batch_size_);
    }

    ~BSNN_MODEL()
    {
        ReleaseModel();
    }

    BSNN_MODEL(const BSNN_MODEL&) = delete;
    BSNN_MODEL& operator=(const BSNN_MODEL&) = delete;

    void InitModel(const std::string& model_dir, uint32_t batch_size)
    {
        ReleaseModel();

        const std::string meta_file = ResolveMetaFile(model_dir);
        if (meta_file.empty()) {
            printf("bsnn: no .meta file found in: %s\n", model_dir.c_str());
            return;
        }
        if (batch_size == 0) {
            printf("bsnn: invalid batch size %u!\n", batch_size);
            return;
        }

        // Match /userdata/new_bsnn AsyncPipeline (C1200 / A1000).
        const int flag = BSNN_FLAG_DISABLE_AUTO_FLUSH |
                         BSNN_FLAG_DISABLE_AUTO_INVLD |
                         BSNN_FLAG_ENABLE_BUFFER_OUTSIDE;
        const int n4_core_mask =
            target_engine_ & ~(BSNN_CORE_MASK_DIE_0 | BSNN_CORE_MASK_DIE_1);

        printf("bsnn: meta: %s\n", meta_file.c_str());
        printf("bsnn: target engine: %s (core_mask=0x%x)\n",
               bst_bsnn::engine_label(n4_core_mask), n4_core_mask);

        if (bsnn_init(&ctx_, meta_file.c_str(), flag, 0, n4_core_mask) != BSNN_SUCCESS) {
            printf("bsnn: init failed!\n");
            ctx_ = nullptr;
            return;
        }

        bsnn_sdk_version version{};
        if (bsnn_query(ctx_, BSNN_QUERY_SDK_VERSION, &version, sizeof(version)) == BSNN_SUCCESS) {
            printf("bsnn: u-space version: %s\n", version.so_version);
            printf("bsnn: k-space version: %s\n", version.ko_version);
            printf("bsnn: fw      version: %s\n", version.fw_version);
        }

        int inp_nums = 0;
        int out_nums = 0;
        int inst_batch = 0;
        uint64_t activation_size = 0;
        if (bsnn_query(ctx_, BSNN_QUERY_MODEL_INPUTS_NUMS, &inp_nums, sizeof(int)) != BSNN_SUCCESS ||
            bsnn_query(ctx_, BSNN_QUERY_MODEL_OUTPUTS_NUMS, &out_nums, sizeof(int)) != BSNN_SUCCESS) {
            printf("bsnn: query input/output num failed!\n");
            ReleaseModel();
            return;
        }
        if (bsnn_query(ctx_, BSNN_QUERY_MODEL_BATCH_SIZE, &inst_batch, sizeof(int)) != BSNN_SUCCESS) {
            inst_batch = 0;
        }
        if (bsnn_query(ctx_, BSNN_QUERY_MODEL_INTERNAL_SIZE, &activation_size,
                       sizeof(uint64_t)) != BSNN_SUCCESS) {
            activation_size = 0;
        }

        if (inp_nums <= 0 || out_nums <= 0) {
            printf("bsnn: invalid tensor nums in=%d out=%d\n", inp_nums, out_nums);
            ReleaseModel();
            return;
        }

        batch_size_ = inst_batch > 0 ? static_cast<uint32_t>(inst_batch) : batch_size;
        if (batch_size_ == 0) {
            printf("bsnn: invalid resolved batch size\n");
            ReleaseModel();
            return;
        }

        input_attrs_.resize(static_cast<size_t>(inp_nums));
        output_attrs_.resize(static_cast<size_t>(out_nums));
        if (bsnn_query(ctx_, BSNN_QUERY_MODEL_INPUTS_ATTR, input_attrs_.data(),
                       sizeof(bsnn_tensor_attr) * static_cast<size_t>(inp_nums)) != BSNN_SUCCESS ||
            bsnn_query(ctx_, BSNN_QUERY_MODEL_OUTPUTS_ATTR, output_attrs_.data(),
                       sizeof(bsnn_tensor_attr) * static_cast<size_t>(out_nums)) != BSNN_SUCCESS) {
            printf("bsnn: query tensor attrs failed!\n");
            ReleaseModel();
            return;
        }

        const size_t total_inputs =
            static_cast<size_t>(batch_size_) * static_cast<size_t>(inp_nums);
        const size_t total_outputs =
            static_cast<size_t>(batch_size_) * static_cast<size_t>(out_nums);

        inputs_.assign(total_inputs, nullptr);
        outputs_.assign(total_outputs, nullptr);
        internals_.assign(batch_size_, nullptr);

        for (uint32_t b = 0; b < batch_size_; ++b) {
            for (int i = 0; i < inp_nums; ++i) {
                BSNN_MEM mem = bsnn_mem_create(ctx_, input_attrs_[i].size);
                if (!mem) {
                    printf("bsnn: create input mem failed (b=%u, i=%d)\n", b, i);
                    ReleaseModel();
                    return;
                }
                inputs_[static_cast<size_t>(b) * static_cast<size_t>(inp_nums) +
                        static_cast<size_t>(i)] = mem;
                if (bsnn_set_input_mem(ctx_, i, static_cast<int>(b), mem) != BSNN_SUCCESS) {
                    printf("bsnn: set input mem failed (b=%u, i=%d)\n", b, i);
                    ReleaseModel();
                    return;
                }
            }
            for (int i = 0; i < out_nums; ++i) {
                BSNN_MEM mem = bsnn_mem_create(ctx_, output_attrs_[i].size);
                if (!mem) {
                    printf("bsnn: create output mem failed (b=%u, i=%d)\n", b, i);
                    ReleaseModel();
                    return;
                }
                outputs_[static_cast<size_t>(b) * static_cast<size_t>(out_nums) +
                         static_cast<size_t>(i)] = mem;
                if (bsnn_set_output_mem(ctx_, i, static_cast<int>(b), mem) != BSNN_SUCCESS) {
                    printf("bsnn: set output mem failed (b=%u, i=%d)\n", b, i);
                    ReleaseModel();
                    return;
                }
            }
            BSNN_MEM act = bsnn_mem_create(ctx_, activation_size);
            if (!act) {
                printf("bsnn: create internal mem failed (b=%u, size=%lu)\n",
                       b, static_cast<unsigned long>(activation_size));
                ReleaseModel();
                return;
            }
            internals_[b] = act;
            if (bsnn_set_internal_mem(ctx_, static_cast<int>(b), act) != BSNN_SUCCESS) {
                printf("bsnn: set internal mem failed (b=%u)\n", b);
                ReleaseModel();
                return;
            }
        }

        dataunit_ = bsnn_get_dataunit(ctx_);
        if (!dataunit_) {
            printf("bsnn: get dataunit failed!\n");
            ReleaseModel();
            return;
        }

        infer_ = {};
        infer_.non_block = 1;
        infer_.timeout_ms = 8000;
        infer_.fence_fd = 0;
        infer_.dataunit = dataunit_;

        // Warmup in Run() after valid input has been copied and synced.

        printf("bsnn: load model done!\n");
        model_init_succ_ = true;
    }

    void ReleaseModel()
    {
        if (ctx_ != nullptr) {
            if (dataunit_ != nullptr) {
                bsnn_put_dataunit(ctx_, dataunit_);
                dataunit_ = nullptr;
            }
            for (auto mem : inputs_) {
                if (mem) {
                    bsnn_mem_destroy(ctx_, mem);
                }
            }
            for (auto mem : outputs_) {
                if (mem) {
                    bsnn_mem_destroy(ctx_, mem);
                }
            }
            for (auto mem : internals_) {
                if (mem) {
                    bsnn_mem_destroy(ctx_, mem);
                }
            }
            bsnn_deinit(ctx_);
            ctx_ = nullptr;
        }
        inputs_.clear();
        outputs_.clear();
        internals_.clear();
        input_attrs_.clear();
        output_attrs_.clear();
        infer_ = {};
        warmed_up_ = false;
        last_infer_us_ = 0;
        model_init_succ_ = false;
    }

    bool Run(const std::vector<uint8_t*>& input_data, const std::vector<size_t>& input_sizes)
    {
        if (!model_init_succ_) {
            return false;
        }
        if (input_data.size() != input_sizes.size()) {
            printf("bsnn: input pointer count(%zu) != size count(%zu)!\n",
                   input_data.size(), input_sizes.size());
            return false;
        }
        if (input_data.size() != inputs_.size()) {
            printf("bsnn: expected %zu input tensors but got %zu!\n",
                   inputs_.size(), input_data.size());
            return false;
        }

        // Host->device copy/sync is outside NPU timing (matches README engine FPS).
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (!inputs_[i] || input_data[i] == nullptr) {
                printf("bsnn: null input at index %zu!\n", i);
                return false;
            }
            if (!inputs_[i]->va) {
                printf("bsnn: input mem va is null at index %zu!\n", i);
                return false;
            }
            if (input_sizes[i] > static_cast<size_t>(inputs_[i]->size)) {
                printf("bsnn: input size overflow at index %zu (size=%zu, cap=%lu)!\n",
                       i, input_sizes[i], static_cast<unsigned long>(inputs_[i]->size));
                return false;
            }
            std::memcpy(inputs_[i]->va, input_data[i], input_sizes[i]);
            if (bsnn_mem_sync(ctx_, inputs_[i], BSNN_MEM_SYNC_TO_DEVICE) != BSNN_SUCCESS) {
                printf("bsnn: sync input to device failed at index %zu!\n", i);
                return false;
            }
        }

        infer_.non_block = 1;
        infer_.timeout_ms = 8000;
        infer_.fence_fd = 0;
        infer_.dataunit = dataunit_;
        infer_.timestamp_start = 0;
        infer_.timestamp_finish = 0;

        if (!warmed_up_) {
            if (bsnn_run(ctx_, &infer_) != BSNN_SUCCESS ||
                bsnn_wait(ctx_, &infer_) != BSNN_SUCCESS) {
                printf("bsnn: warmup failed!!\n");
                return false;
            }
            warmed_up_ = true;
            infer_.timestamp_start = 0;
            infer_.timestamp_finish = 0;
        }

        const auto t0 = std::chrono::steady_clock::now();
        if (bsnn_run(ctx_, &infer_) != BSNN_SUCCESS) {
            printf("bsnn: run model failed!!\n");
            return false;
        }
        if (bsnn_wait(ctx_, &infer_) != BSNN_SUCCESS) {
            printf("bsnn: wait model failed!!\n");
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const int64_t wall_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        // Board-side monotonic clock around synchronous run+wait.
        last_infer_us_ = wall_us;

        for (size_t i = 0; i < outputs_.size(); ++i) {
            if (outputs_[i] &&
                bsnn_mem_sync(ctx_, outputs_[i], BSNN_MEM_SYNC_FROM_DEVICE) != BSNN_SUCCESS) {
                printf("bsnn: sync output from device failed at index %zu!\n", i);
                return false;
            }
        }
        return true;
    }

    bool Run(uint8_t* input_data, size_t len)
    {
        return Run(std::vector<uint8_t*>{input_data}, std::vector<size_t>{len});
    }

    const std::vector<BSNN_MEM>& GetInputs() const { return inputs_; }
    const std::vector<BSNN_MEM>& GetOutputs() const { return outputs_; }
    const std::vector<bsnn_tensor_attr>& GetInputAttrs() const { return input_attrs_; }
    const std::vector<bsnn_tensor_attr>& GetOutputAttrs() const { return output_attrs_; }

    uint32_t GetOutputTensorCount() const
    {
        return static_cast<uint32_t>(output_attrs_.size());
    }

    uint32_t GetBatchSize() const { return batch_size_; }

    const char* engine_label() const { return bst_bsnn::engine_label(target_engine_); }

    /** Last synchronous run+wait duration in microseconds (excludes memcpy / mem_sync / warmup). */
    int64_t LastInferUs() const { return last_infer_us_; }

    bool ReleaseOutputBuffer() { return true; }

    uint8_t* OutputVa(size_t idx) const
    {
        if (idx >= outputs_.size() || !outputs_[idx]) {
            return nullptr;
        }
        return reinterpret_cast<uint8_t*>(outputs_[idx]->va);
    }

    static std::vector<float> QuantScales(const bsnn_tensor_attr& attr)
    {
        std::vector<float> scales;
        if (attr.quant == nullptr || !attr.quant->quantized || attr.quant->channel_count <= 0) {
            scales.push_back(1.0f);
            return scales;
        }
        scales.resize(static_cast<size_t>(attr.quant->channel_count));
        for (int i = 0; i < attr.quant->channel_count; ++i) {
            scales[static_cast<size_t>(i)] = attr.quant->quant_infos[i].scale;
        }
        return scales;
    }

    static float QuantScale(const bsnn_tensor_attr& attr, int channel = 0)
    {
        return ScaleAt(QuantScales(attr), channel);
    }

    static float ScaleAt(const std::vector<float>& scales, int channel)
    {
        if (scales.empty()) {
            return 1.0f;
        }
        if (channel >= 0 && channel < static_cast<int>(scales.size())) {
            return scales[static_cast<size_t>(channel)];
        }
        return scales[0];
    }

public:
    std::string model_dir_;
    int target_engine_ = BSNN_CORE_MASK_AUTO;
    bool model_init_succ_ = false;
    uint32_t batch_size_ = 1;
    BSNN_CTX ctx_ = nullptr;
    BSNN_DATAUNIT dataunit_ = nullptr;
    bsnn_infer infer_{};
    std::vector<BSNN_MEM> inputs_;
    std::vector<BSNN_MEM> outputs_;
    std::vector<BSNN_MEM> internals_;
    std::vector<bsnn_tensor_attr> input_attrs_;
    std::vector<bsnn_tensor_attr> output_attrs_;

private:
    bool warmed_up_ = false;
    int64_t last_infer_us_ = 0;

    static std::string ResolveMetaFile(const std::string& path)
    {
        struct stat info {};
        if (stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode)) {
            return path;
        }
        for (const std::string& dir : {path, path + "/model"}) {
            DIR* entries = opendir(dir.c_str());
            if (!entries) {
                continue;
            }
            std::string meta;
            while (const dirent* entry = readdir(entries)) {
                const std::string name(entry->d_name);
                if (name.size() < 5 || name.compare(name.size() - 5, 5, ".meta") != 0) {
                    continue;
                }
                const std::string candidate = dir + "/" + name;
                if (stat(candidate.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
                    (meta.empty() || candidate < meta)) {
                    meta = candidate;
                }
            }
            closedir(entries);
            if (!meta.empty()) {
                return meta;
            }
        }
        return {};
    }
};

#endif
