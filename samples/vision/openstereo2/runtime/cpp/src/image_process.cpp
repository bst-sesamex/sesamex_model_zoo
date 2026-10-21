#include "image_process.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

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
    // 8.3.1-rebuilt package (A1000B0): graph optimization fused the dual softmax
    // branches into the network, so the model emits a single disparity tensor
    // disp_pred [W,H,1,1] WHD UINT8, dequant_scale 1/256 (normalized 0..1).
    // Restore pixel disparity (x max_disp 192) to keep the legacy colormap contract.
    if (model.GetOutputTensorCount() < 1) {
        std::cerr << "Error: Expected at least 1 output tensor for LightStereo" << std::endl;
        return;
    }

    const auto& output0 = model.GetOutputAttrs()[0];
    const uint8_t* out0_data = model.OutputVa(0);
    if (!out0_data) {
        std::cerr << "Error: LightStereo output va is NULL" << std::endl;
        return;
    }

    // The decoder supports UINT8 and 16-bit (INT16/UINT16/FP16) packed
    // outputs; 16-bit samples are addressed as int16 elements, not bytes.
    // The dtype is declared per tensor in the model meta / runtime attrs.
    size_t elem_size = 1;
    const bool dtype_16bit = (output0.dtype == BSNN_TENSOR_INT16 ||
                              output0.dtype == BSNN_TENSOR_UINT16 ||
                              output0.dtype == BSNN_TENSOR_FP16);
    if (dtype_16bit) elem_size = 2;

    int depth = 0, width = 0, height = 0, pitchD0 = 0, pitchW0 = 0;
    stereo_output_layout(output0, depth, width, height, pitchD0, pitchW0);
    (void)depth; (void)pitchD0;

    std::cout << "Output tensor size: " << width << "x" << height
              << " dorder=" << output0.dorder << " pitchW=" << pitchW0
              << " dtype=" << output0.dtype << " elem_size=" << elem_size
              << " bytes=" << output0.size << std::endl;

    // Row stride in elements: pitches may be reported in bytes or elements.
    // Validate against the total buffer size and prefer the interpretation
    // that fits; default to the element reading (true for contiguous WHD).
    int row_stride = pitchW0;
    if (static_cast<size_t>(pitchW0) * static_cast<size_t>(height) * elem_size >
        static_cast<size_t>(output0.size)) {
        row_stride = static_cast<int>(static_cast<size_t>(pitchW0) / elem_size);
    }

    const bool dwh0 = (output0.dorder == BSNN_TENSOR_DWH);
    (void)dwh0; // WHD expected; index math identical for the [W,H,1,1] case
    const auto& meta = model.GetModelMeta();
    const std::vector<float> empty_scales;
    const std::vector<float>& meta0 =
        (!meta.output_tensors.empty()) ? meta.output_tensors[0].dequant_scale : empty_scales;
    const std::vector<float> scales0 = stereo_dequant_scales(output0, meta0, 1);
    const float dequant = scales0.empty() ? (1.0f / 256.0f) : scales0[0];
    constexpr float kMaxDisp = 192.0f;

    disparity_map.create(height, width, CV_32F);
    const int16_t* out0_s16 = reinterpret_cast<const int16_t*>(out0_data);
    for (int h = 0; h < height; ++h) {
        float* dst_row = disparity_map.ptr<float>(h);
        for (int w = 0; w < width; ++w) {
            const size_t idx = static_cast<size_t>(h) * static_cast<size_t>(row_stride) +
                               static_cast<size_t>(w);
            float raw = dtype_16bit ? static_cast<float>(out0_s16[idx])
                                    : static_cast<float>(out0_data[idx]);
            dst_row[w] = raw * dequant * kMaxDisp;
        }
    }

    std::cout << "LightStereo post-processing completed (single-output disparity)" << std::endl;
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
