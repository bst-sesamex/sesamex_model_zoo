#include "image_process.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

#define USE_LIGHTSTEREO_FAST_EXP
// Macro wrapper to choose between std::exp (default) and fast_exp (optional)
#ifndef LIGHTSTEREO_EXP
#ifdef USE_LIGHTSTEREO_FAST_EXP
#define LIGHTSTEREO_EXP(x) fast_exp(x)
#else
#define LIGHTSTEREO_EXP(x) std::exp(x)
#endif
#endif

// forward declaration so LIGHTSTEREO_EXP can use fast_exp before it's defined later in this file
inline float fast_exp(float x);

// LightStereo预处理函数
void lightstereo_preprocess(const cv::Mat& left_img, const cv::Mat& right_img, 
                           const ModelMeta meta,
                           uint8_t* left_buffer, uint8_t* right_buffer)
{
    if (meta.input_tensors.size() < 2) {
        std::cerr << "Error: Expected 2 input tensors for LightStereo" << std::endl;
        return;
    }
    
    const auto& input_tensor = meta.input_tensors[0]; // 假设两个输入张量规格相同
    int target_width = input_tensor.width;
    int target_height = input_tensor.height;
    int channels = input_tensor.channels;
    
    std::cout << "Target size: " << target_width << "x" << target_height << "x" << channels << std::endl;
    
    // 处理左图像
    cv::Mat left_resized;
    cv::resize(left_img, left_resized, cv::Size(target_width, target_height));
    
    // 处理右图像  
    cv::Mat right_resized;
    cv::resize(right_img, right_resized, cv::Size(target_width, target_height));
    
    // 转换为RGB格式并存储为UINT8（不需要归一化，硬件完成）
    // 数据格式：WHD (按meta文件的dorder)
    size_t img_size = target_width * target_height * channels;
    
    // 左图像数据转换 (HWC -> CHW)
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < target_height; h++) {
            for (int w = 0; w < target_width; w++) {
                int src_idx = h * target_width * channels + w * channels + (2 - c); // BGR -> RGB
                int dst_idx = c * target_width * target_height + h * target_width + w;
                left_buffer[dst_idx] = left_resized.data[src_idx];
            }
        }
    }
    
    // 右图像数据转换 (HWC -> CHW)
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < target_height; h++) {
            for (int w = 0; w < target_width; w++) {
                int src_idx = h * target_width * channels + w * channels + (2 - c); // BGR -> RGB
                int dst_idx = c * target_width * target_height + h * target_width + w;
                right_buffer[dst_idx] = right_resized.data[src_idx];
            }
        }
    }
    
    std::cout << "LightStereo preprocessing completed" << std::endl;
}

// softmax函数实现
void out_softmax(float* data, int depth, int width, int height)
{
    int hw = width * height;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int base_idx = h * width + w;
            
            // 找到最大值
            float max_val = data[base_idx];
            for (int d = 1; d < depth; d++) {
                float val = data[d * hw + base_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // 计算exp和sum
            float sum = 0.0f;
            for (int d = 0; d < depth; d++) {
                float val = LIGHTSTEREO_EXP(data[d * hw + base_idx] - max_val);
                data[d * hw + base_idx] = val;
                sum += val;
            }
            
            // 归一化
            for (int d = 0; d < depth; d++) {
                data[d * hw + base_idx] /= sum;
            }
        }
    }
}
inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

inline float fast_sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
}


static void stereo_output_layout(const bsnn_tensor_attr& attr, int& depth, int& width,
                                 int& height, int& pitch_d, int& pitch_w)
{
    const bool dwh = (attr.dorder == BSNN_TENSOR_DWH);
    if (dwh) {
        depth = static_cast<int>(attr.dims[0]);
        width = static_cast<int>(attr.dims[1]);
        height = static_cast<int>(attr.dims[2]);
        pitch_d = static_cast<int>(attr.pitches[0]);
        pitch_w = static_cast<int>(attr.pitches[1]);
        if (pitch_d < depth) pitch_d = depth;
        if (pitch_w < width) pitch_w = width;
    } else {
        width = static_cast<int>(attr.dims[0]);
        height = static_cast<int>(attr.dims[1]);
        depth = static_cast<int>(attr.dims[2]);
        pitch_w = static_cast<int>(attr.pitches[0]);
        pitch_d = depth;
        if (pitch_w < width) pitch_w = width;
    }
}

static size_t stereo_output_index(bool dwh, int d, int w, int h, int height,
                                  int pitch_d, int pitch_w)
{
    if (dwh) {
        return static_cast<size_t>(h) * static_cast<size_t>(pitch_w) * static_cast<size_t>(pitch_d) +
               static_cast<size_t>(w) * static_cast<size_t>(pitch_d) +
               static_cast<size_t>(d);
    }
    return (static_cast<size_t>(d) * static_cast<size_t>(height) + static_cast<size_t>(h)) *
               static_cast<size_t>(pitch_w) +
           static_cast<size_t>(w);
}

static std::vector<float> stereo_dequant_scales(const bsnn_tensor_attr& attr,
                                                const std::vector<float>& meta_scales,
                                                int depth)
{
    std::vector<float> runtime = BSNN_MODEL::QuantScales(attr);
    const bool runtime_missing = runtime.empty() ||
                                 (runtime.size() == 1 && runtime[0] == 1.0f);
    const std::vector<float>& src = (runtime_missing && !meta_scales.empty()) ? meta_scales : runtime;
    std::vector<float> scales(static_cast<size_t>(depth), 1.0f);
    if (src.empty()) {
        return scales;
    }
    if (src.size() == 1) {
        std::fill(scales.begin(), scales.end(), src[0]);
        return scales;
    }
    for (int d = 0; d < depth; ++d) {
        scales[static_cast<size_t>(d)] = (d < static_cast<int>(src.size())) ? src[static_cast<size_t>(d)] : src[0];
    }
    return scales;
}

void lightstereo_process_output(const BSNN_MODEL& model, cv::Mat& disparity_map)
{
    if (model.GetOutputTensorCount() < 2) {
        std::cerr << "Error: Expected 2 output tensors for LightStereo" << std::endl;
        return;
    }

    const auto& output0 = model.GetOutputAttrs()[0];
    const auto& output1 = model.GetOutputAttrs()[1];
    const uint8_t* out0_data = model.OutputVa(0);
    const uint8_t* out1_data = model.OutputVa(1);
    if (!out0_data || !out1_data) {
        std::cerr << "Error: LightStereo output va is NULL" << std::endl;
        return;
    }

    int depth = 0, width = 0, height = 0, pitchD0 = 0, pitchW0 = 0;
    int depth1 = 0, width1 = 0, height1 = 0, pitchD1 = 0, pitchW1 = 0;
    stereo_output_layout(output0, depth, width, height, pitchD0, pitchW0);
    stereo_output_layout(output1, depth1, width1, height1, pitchD1, pitchW1);
    if (depth1 < depth) depth = depth1;
    if (width1 < width) width = width1;
    if (height1 < height) height = height1;

    std::cout << "Output tensor size: " << depth << "x" << width << "x" << height
              << " dorder=" << output0.dorder << " pitchD=" << pitchD0
              << " pitchW=" << pitchW0 << std::endl;

    const bool dwh0 = (output0.dorder == BSNN_TENSOR_DWH);
    const bool dwh1 = (output1.dorder == BSNN_TENSOR_DWH);
    const auto& meta = model.GetModelMeta();
    const std::vector<float> empty_scales;
    const std::vector<float>& meta0 =
        (!meta.output_tensors.empty()) ? meta.output_tensors[0].dequant_scale : empty_scales;
    const std::vector<float>& meta1 =
        (meta.output_tensors.size() > 1) ? meta.output_tensors[1].dequant_scale : empty_scales;
    const std::vector<float> scales0 = stereo_dequant_scales(output0, meta0, depth);
    const std::vector<float> scales1 = stereo_dequant_scales(output1, meta1, depth);

    std::vector<float> lut0(static_cast<size_t>(depth) * 256);
    std::vector<float> lut1(static_cast<size_t>(depth) * 256);
    for (int d = 0; d < depth; ++d) {
        for (int u = 0; u < 256; ++u) {
            const int8_t q = static_cast<int8_t>(u);
            lut0[static_cast<size_t>(d) * 256 + static_cast<size_t>(u)] =
                static_cast<float>(q) * scales0[static_cast<size_t>(d)];
            lut1[static_cast<size_t>(d) * 256 + static_cast<size_t>(u)] =
                static_cast<float>(q) * scales1[static_cast<size_t>(d)];
        }
    }

    std::vector<float> v0(static_cast<size_t>(depth));
    std::vector<float> v1(static_cast<size_t>(depth));
    disparity_map = cv::Mat::zeros(height, width, CV_32F);

    for (int h = 0; h < height; ++h) {
        float* dst_row = disparity_map.ptr<float>(h);
        for (int w = 0; w < width; ++w) {
            float max0 = -std::numeric_limits<float>::infinity();
            float max1 = -std::numeric_limits<float>::infinity();
            for (int d = 0; d < depth; ++d) {
                const size_t idx0 = stereo_output_index(dwh0, d, w, h, height, pitchD0, pitchW0);
                const size_t idx1 = stereo_output_index(dwh1, d, w, h, height1, pitchD1, pitchW1);
                const float vv0 = lut0[static_cast<size_t>(d) * 256 + out0_data[idx0]];
                const float vv1 = lut1[static_cast<size_t>(d) * 256 + out1_data[idx1]];
                v0[static_cast<size_t>(d)] = vv0;
                v1[static_cast<size_t>(d)] = vv1;
                if (vv0 > max0) max0 = vv0;
                if (vv1 > max1) max1 = vv1;
            }

            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sumProd = 0.0f;
            for (int d = 0; d < depth; ++d) {
                const float e0 = LIGHTSTEREO_EXP(v0[static_cast<size_t>(d)] - max0);
                const float e1 = LIGHTSTEREO_EXP(v1[static_cast<size_t>(d)] - max1);
                sum0 += e0;
                sum1 += e1;
                sumProd += e0 * e1;
            }

            const float dot = (sum0 > 0.0f && sum1 > 0.0f) ? (sumProd / (sum0 * sum1)) : 0.0f;
            dst_row[w] = dot * 192.0f;
        }
    }

    std::cout << "LightStereo post-processing completed" << std::endl;
}

// 视差图颜色映射函数 (参考Python代码的disp_to_color和disp_map函数)
cv::Mat create_disparity_colormap(const cv::Mat& disparity, float max_disp)
{
    cv::Mat colormap(disparity.rows, disparity.cols, CV_8UC3);
    
    // 基于颜色直方图的颜色映射
    float bin_map[8][4] = {
        {0,0,0,114},
        {0,0,1,185}, 
        {1,0,0,114},
        {1,0,1,174},
        {0,1,0,114},
        {0,1,1,185},
        {1,1,0,114},
        {1,1,1,0}
    };
    
    float bins[7] = {114.0f, 185.0f, 114.0f, 174.0f, 114.0f, 185.0f, 114.0f};
    float cbins[7];
    
    // 累积和
    cbins[0] = bins[0];
    for (int i = 1; i < 7; i++) {
        cbins[i] = cbins[i-1] + bins[i];
    }
    
    // 归一化
    float total = cbins[6];
    for (int i = 0; i < 7; i++) {
        bins[i] /= total;
        if (i < 6) cbins[i] /= total;
    }
    
    for (int i = 0; i < disparity.rows; i++) {
        for (int j = 0; j < disparity.cols; j++) {
            float disp_val = disparity.at<float>(i, j) / max_disp;
            disp_val = std::max(0.0f, std::min(1.0f, disp_val));
            
            // 找到对应的bin
            int bin_idx = 0;
            while (bin_idx < 6 && disp_val > cbins[bin_idx]) {
                bin_idx++;
            }
            
            // 计算插值比例
            float prev_cbin = (bin_idx == 0) ? 0.0f : cbins[bin_idx - 1];
            float ratio = (disp_val - prev_cbin) / bins[bin_idx];
            
            // 插值颜色
            cv::Vec3b color;
            for (int c = 0; c < 3; c++) {
                float val = bin_map[bin_idx][c] * (1.0f - ratio) + bin_map[bin_idx + 1][c] * ratio;
                color[2-c] = (uint8_t)(val * 255.0f); // RGB -> BGR
            }
            
            colormap.at<cv::Vec3b>(i, j) = color;
        }
    }
    
    return colormap;
}
