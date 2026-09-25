#include "image_process.h"
#include "bsnn_model_load.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

using namespace std;

// ============================================================
// 前处理
//
// 板端 WHD 布局说明 (bsnnrt 0.13.0 tensor attr):
//   dims[0] = W, dims[1] = H, dims[2] = D(channels), dims[3] = N
//   pitches[0] = W 维度对齐后的元素数 (>= W)
//   总 buffer 大小 = D * H * pitches[0] * elem_size  (约等于 attr.size)
// ============================================================
void cls_preprocess(const cv::Mat& src,
                    std::vector<uint8_t>& output_buf,
                    int pitch, int img_w, int img_h, int img_d,
                    int dtype)
{
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(img_w, img_h), 0.0, 0.0, cv::INTER_LINEAR);

    if (dtype == BSNN_TENSOR_UINT8) {
        size_t total_bytes = (size_t)img_d * img_h * pitch;
        output_buf.assign(total_bytes, 0);

        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);  // channels[0]=B, [1]=G, [2]=R

        for (int d = 0; d < img_d; d++) {
            const cv::Mat& ch = channels[d];
            for (int h = 0; h < img_h; h++) {
                const uint8_t* row_ptr = ch.ptr<uint8_t>(h);
                size_t offset = (size_t)d * img_h * pitch + (size_t)h * pitch;
                std::memcpy(output_buf.data() + offset, row_ptr, img_w);
            }
        }

        printf("-> Preprocess: UINT8/BGR path, buf=%zu bytes\n", total_bytes);
        printf("   pixel[0,0]: ch0=%d ch1=%d ch2=%d\n",
               (int)channels[0].at<uint8_t>(0,0),
               (int)channels[1].at<uint8_t>(0,0),
               (int)channels[2].at<uint8_t>(0,0));
    } else {
        cv::Mat bgr_float;
        resized.convertTo(bgr_float, CV_32FC3, 1.0 / 255.0);

        static const float BGR_MEAN[3] = {CIFAR10_MEAN[2], CIFAR10_MEAN[1], CIFAR10_MEAN[0]};
        static const float BGR_STD[3]  = {CIFAR10_STD[2],  CIFAR10_STD[1],  CIFAR10_STD[0]};

        std::vector<cv::Mat> channels(3);
        cv::split(bgr_float, channels);
        for (int c = 0; c < 3; c++) {
            channels[c] = (channels[c] - BGR_MEAN[c]) / BGR_STD[c];
        }

        size_t total_bytes = (size_t)img_d * img_h * pitch;
        output_buf.assign(total_bytes, 0);

        int w_bytes = img_w * (int)sizeof(float);
        for (int d = 0; d < img_d; d++) {
            const cv::Mat& ch = channels[d];
            for (int h = 0; h < img_h; h++) {
                const float* row_ptr = ch.ptr<float>(h);
                size_t offset = (size_t)d * img_h * pitch + (size_t)h * pitch;
                std::memcpy(output_buf.data() + offset, row_ptr, w_bytes);
            }
        }
        printf("-> Preprocess: FLOAT32/BGR path, buf=%zu bytes\n", total_bytes);
    }
}

static float dequant_elem(const void* data, int offset, bsnn_tensor_dtype dtype, float scale)
{
    switch (dtype) {
    case BSNN_TENSOR_INT8:
        return static_cast<const int8_t*>(data)[offset] * scale;
    case BSNN_TENSOR_UINT8:
        return static_cast<const uint8_t*>(data)[offset] * scale;
    case BSNN_TENSOR_INT16:
        return static_cast<const int16_t*>(data)[offset] * scale;
    case BSNN_TENSOR_FP16:
        return 0.0f;  // not used by classification models
    case BSNN_TENSOR_FP32:
        return static_cast<const float*>(data)[offset];
    default:
        return static_cast<const int8_t*>(data)[offset] * scale;
    }
}

void cls_postprocess(const BSNN_MODEL& model, ClsResult& result)
{
    if (model.GetOutputs().empty() || model.GetOutputAttrs().empty()) {
        std::cerr << "Failed to get output tensor." << std::endl;
        result.class_id = -1;
        result.score = 0.0f;
        result.class_name = "unknown";
        return;
    }

    const bsnn_tensor_attr& out_attr = model.GetOutputAttrs()[0];
    const BSNN_MEM out_mem = model.GetOutputs()[0];
    if (!out_mem || !out_mem->va) {
        std::cerr << "Output memory is null." << std::endl;
        result.class_id = -1;
        result.score = 0.0f;
        result.class_name = "unknown";
        return;
    }

    int num_classes = static_cast<int>(out_attr.dims[0]);
    int out_w = out_attr.dims_size > 1 ? static_cast<int>(out_attr.dims[1]) : 1;
    int out_h = out_attr.dims_size > 2 ? static_cast<int>(out_attr.dims[2]) : 1;
    int out_pitch = static_cast<int>(out_attr.pitches[0]);
    if (out_pitch < num_classes) {
        out_pitch = num_classes;
    }
    if (num_classes <= 0) {
        num_classes = NUM_CLASSES;
    }

    const float scale = BSNN_MODEL::QuantScale(out_attr, 0);

    printf("-> Output tensor: D=%d, W=%d, H=%d, pitch=%d, dtype=%d, size=%zu\n",
           num_classes, out_w, out_h, out_pitch, static_cast<int>(out_attr.dtype),
           out_attr.size);
    printf("-> Scale (per-tensor): %.6f\n", scale);

    const void* data = out_mem->va;
    printf("-> Raw values: ");
    for (int c = 0; c < std::min(num_classes, 16); c++) {
        if (out_attr.dtype == BSNN_TENSOR_INT8) {
            printf("%4d ", (int)static_cast<const int8_t*>(data)[c]);
        } else if (out_attr.dtype == BSNN_TENSOR_FP32) {
            printf("%.3f ", static_cast<const float*>(data)[c]);
        } else {
            printf("%4d ", (int)static_cast<const uint8_t*>(data)[c]);
        }
    }
    printf("\n");

    std::vector<float> logits(num_classes);
    for (int c = 0; c < num_classes; c++) {
        logits[c] = dequant_elem(data, c, out_attr.dtype, scale);
    }

    printf("-> Logits:    ");
    for (int c = 0; c < num_classes; c++) {
        printf("%.3f ", logits[c]);
    }
    printf("\n");

    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(num_classes);
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        probs[c] = std::exp(logits[c] - max_logit);
        sum_exp += probs[c];
    }
    for (int c = 0; c < num_classes; c++) {
        probs[c] /= sum_exp;
    }

    int best_cls = 0;
    float best_prob = probs[0];
    for (int c = 1; c < num_classes; c++) {
        if (probs[c] > best_prob) {
            best_prob = probs[c];
            best_cls = c;
        }
    }

    result.class_id = best_cls;
    result.score = best_prob;
    if (best_cls >= 0 && best_cls < NUM_CLASSES) {
        result.class_name = CIFAR10_CLASSES[best_cls];
    } else {
        result.class_name = "unknown";
    }

    std::vector<int> indices(num_classes);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + std::min(5, num_classes),
                      indices.end(),
                      [&probs](int a, int b) { return probs[a] > probs[b]; });

    printf("-> Top-5 predictions:\n");
    for (int i = 0; i < std::min(5, num_classes); i++) {
        int idx = indices[i];
        const char* name = (idx >= 0 && idx < NUM_CLASSES) ? CIFAR10_CLASSES[idx] : "unknown";
        printf("   [%d] %s: %.4f (%.2f%%)\n", idx, name, probs[idx], probs[idx] * 100.0f);
    }
}
