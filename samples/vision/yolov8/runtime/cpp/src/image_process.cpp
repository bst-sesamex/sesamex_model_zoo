#include "image_process.h"
#include <iostream>
#include <vector>
#include "coco_info.h"
#include <cmath>

using namespace std;

inline float normal_sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float inverse_normal_sigmoid(float y)
{
    return -std::log(1.0f / y - 1.0f);
}

void preprocess(const cv::Mat& src, cv::Mat& dst)
{
    // Resize and pad image while meeting stride-multiple constraints
    double width = src.cols;
    double height = src.rows;

    if (width == IMG_WIDTH && height == IMG_HEIGHT) {
        dst = src.clone();
        return;
    }

    // Scale ratio (new / old)
    double r = MIN(IMG_WIDTH / width, IMG_HEIGHT / height);
    
    int scale_width = (int)(std::round(width * r));
    int scale_height = (int)(std::round(height * r));
    int dw = (IMG_WIDTH - scale_width) / 2;
    int dh = (IMG_HEIGHT - scale_height) / 2;

    cv::resize(src, dst, cv::Size(scale_width, scale_height), 0.0, 0.0, cv::INTER_LINEAR);
    int top = (int)(std::round(dh - 0.1));
    int bottom = (int)(std::round(dh + 0.1));
    int left = (int)(std::round(dw - 0.1));
    int right = (int)(std::round(dw + 0.1));

    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

void generate_bboxes(const int8_t* bsnn_out, const float* anchors, int out_h, int out_w, int stride,
                     float dequant_scale, std::vector<ObjInfo> &objects)
{
    for (int y = 0; y < out_h; y++)
    {
        for (int x = 0; x < out_w; x++)
        {
            auto start_pos = ((y * out_w) + x) * OUT_CHANNELS;
            for (int q = 0; q < 3; q++)
            {
                float anchor_w = anchors[q * 2];
                float anchor_h = anchors[q * 2 + 1];
                auto idx = start_pos + q * PER_ANCHORS_CHN;

                //find class index with max score
                int class_index = 0;
                int8_t class_score_ = std::numeric_limits<int8_t>::lowest();
                for (int k = 0; k < NUM_CLASSES; k++)
                {
                    int8_t score = bsnn_out[idx + 5 + k];
                    if (score > class_score_)
                    {
                        class_score_ = score;
                        class_index = k + 1;
                    }
                }

                float class_score = class_score_ * dequant_scale; //
                float box_score = bsnn_out[idx + 4] * dequant_scale;//
                float confidence = sigmoid(box_score) * sigmoid(class_score);
                float sigb = sigmoid(box_score);
                float sigc = sigmoid(class_score);

                if (confidence >= CONF_THRESHOLD)
                {
                    float dx = sigmoid(bsnn_out[idx]      * dequant_scale);
                    float dy = sigmoid(bsnn_out[idx + 1]  * dequant_scale);
                    float dw = sigmoid(bsnn_out[idx + 2]  * dequant_scale);
                    float dh = sigmoid(bsnn_out[idx + 3]  * dequant_scale);

                    float pb_cx = (dx * 2.f -0.5f + x) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + y) * stride;
                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;
                    ObjInfo obj;
                    obj.classId = class_index;
                    obj.score = confidence;
                    obj.x1 = x0;
                    obj.y1 = y0;
                    obj.x2 = x1;
                    obj.y2 = y1;
                    objects.emplace_back(obj);
                }
            }
        }
    }
}

void yolov5_process_output(const BSNN_MODEL& model, std::vector<ObjInfo> &result)
{
    int strides[] = {8, 16, 32};
    //int strides[] = {32, 16, 8};
    std::vector<ObjInfo> det_objs;
    // stride 8
    {
        float anchors[] = {10.f, 13.f, 16.f, 30.f, 33.f, 23.f};
        //float anchors[] = {1.25, 1.625, 2, 3.75, 4.125, 2.875};
        //float anchors[] = {2.3046875f, 4.61328125f, 3.052734375f, 6.86328125f, 4.08984375f, 9.5078125f};
        int out_h = IMG_HEIGHT / strides[0];
        int out_w = IMG_WIDTH / strides[0];
        std::vector<int8_t> ts_out_i8(out_h * out_w * OUT_CHANNELS);

        // DWH格式下，在（i， j）位置包含D个数据，将i, j拉平和进行处理
        int8_t* out_buf_i8 = (int8_t*)model.OutputVa(2);
        for (int i = 0; i < out_h * out_w; i++) {
            for (int j = 0; j < OUT_CHANNELS; j++) {
                ts_out_i8[i * OUT_CHANNELS + j] = out_buf_i8[i * REG_PITCH + j];
            }
        }

        generate_bboxes(ts_out_i8.data(), anchors, out_h, out_w, strides[0],
                        BSNN_MODEL::QuantScale(model.GetOutputAttrs()[2], 0), det_objs);        
    }

    // stride 16
    {
        float anchors[] = {30.f, 61.f, 62.f, 45.f, 59.f, 119.f};
        //float anchors[] = {1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375};
        // float anchors[] = {5.15625f, 12.0546875f, 6.32421875f, 15.3984375f, 7.75390625f, 18.984375f};
    int out_h = IMG_HEIGHT / strides[1];
    int out_w = IMG_WIDTH / strides[1];
    std::vector<int8_t> ts_out_i8(out_h * out_w * OUT_CHANNELS);

        // DWH格式下，在（i， j）位置包含D个数据，将i, j拉平和进行处理
        int8_t* out_buf_i8 = (int8_t*)model.OutputVa(1);
        for (int i = 0; i < out_h * out_w; i++)
        {
            for (int j = 0; j < OUT_CHANNELS; j++)
            {
                ts_out_i8[i * OUT_CHANNELS + j] = out_buf_i8[i * REG_PITCH + j];
            }
        }

        generate_bboxes(ts_out_i8.data(), anchors, out_h, out_w, strides[1],
                        BSNN_MODEL::QuantScale(model.GetOutputAttrs()[1], 0), det_objs);
    }

    // stride 32
    {
        float anchors[] = {116.f, 90.f, 156.f, 198.f, 373.f, 326.f};
        //float anchors[] = {3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875};
        //float anchors[] = {10.3515625f, 25.28125f, 15.21875f, 35.90625f, 23.421875f, 52.125f};
    int out_h = IMG_HEIGHT / strides[2];
    int out_w = IMG_WIDTH / strides[2];
    std::vector<int8_t> ts_out_i8(out_h * out_w * OUT_CHANNELS);

        // DWH格式下，在（i， j）位置包含D个数据，将i, j拉平和进行处理
        int8_t* out_buf_i8 = (int8_t*)model.OutputVa(0);
        for (int i = 0; i < out_h * out_w; i++)
        {
            for (int j = 0; j < OUT_CHANNELS; j++)
            {
                ts_out_i8[i * OUT_CHANNELS + j] = out_buf_i8[i * REG_PITCH + j];
            }
        }

        generate_bboxes(ts_out_i8.data(), anchors, out_h, out_w, strides[2],
                        BSNN_MODEL::QuantScale(model.GetOutputAttrs()[0], 0), det_objs);
    }

    soft_nms(det_objs, result, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
}

std::vector<float> grid_cells;
std::vector<float> bbox_dfl;

template <typename Stype, typename Dtype>
inline void convert_hwc2chw(size_t ch, size_t w, size_t h, size_t pitch, const Stype* src, Dtype* dst) {
  const size_t hw_stride = w * h;
  for (size_t s = 0UL; s < hw_stride; ++s) {
    size_t stride_index = s;
    const Stype* src_pixel = src + s * pitch;
    for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
      dst[stride_index] = static_cast<Dtype>(src_pixel[c]);
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

inline float inverse_fast_exp(float y)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    v.f = y;
    return (v.i - 1064807160.56887296f) / 12102203.1616540672f;
}

inline float inverse_fast_sigmoid(float y)
{
    return -inverse_fast_exp(1.0f / y - 1.0f);
}

static inline void calc_gain_pad(int orig_w, int orig_h, float &gain, float &pad_w, float &pad_h)
{
    gain = std::min(IMG_WIDTH * 1.0f / orig_w, IMG_HEIGHT * 1.0f / orig_h);
    pad_w = (IMG_WIDTH - orig_w * gain) * 0.5f;
    pad_h = (IMG_HEIGHT - orig_h * gain) * 0.5f;
}

static inline cv::Scalar darker(const cv::Scalar &c, double factor = 0.8)
{
    return cv::Scalar(std::max(0.0, c[0] * factor),
                      std::max(0.0, c[1] * factor),
                      std::max(0.0, c[2] * factor));
}

static inline void letterbox_params(int orig_w, int orig_h, int &resize_w, int &resize_h, int &pad_left, int &pad_top)
{
    double r = std::min(IMG_WIDTH / static_cast<double>(orig_w), IMG_HEIGHT / static_cast<double>(orig_h));
    resize_w = static_cast<int>(std::round(orig_w * r));
    resize_h = static_cast<int>(std::round(orig_h * r));
    double dw = (IMG_WIDTH - resize_w) / 2.0;
    double dh = (IMG_HEIGHT - resize_h) / 2.0;
    pad_left = static_cast<int>(std::round(dw - 0.1));
    pad_top  = static_cast<int>(std::round(dh - 0.1));
    if (pad_left < 0) pad_left = 0;
    if (pad_top  < 0) pad_top  = 0;
}

void yolov8_postprocess(const BSNN_MODEL& model, std::vector<ObjInfo> &det_objs, bool fused_output, InferenceType infer_type)
{
    std::vector<ObjInfo> result;
    result.reserve(128); // Pre-allocate to avoid frequent reallocation
    bool nms_done = false;

    // 1. Query input tensor attributes to determine input image size
    if (model.GetInputAttrs().empty()) {
        std::cerr << "Failed to query input tensor attributes." << std::endl;
        return;
    }
    const bsnn_tensor_attr_t& in_attr = model.GetInputAttrs()[0];
    int img_w = in_attr.dims[0];
    int img_h = in_attr.dims[1];

    std::vector<float> bbox_dfl_local(4);
    std::vector<float> regdeq(16);
    float conf_thresh_inverse = inverse_normal_sigmoid(CONF_THRESHOLD);

    if (!fused_output && infer_type == INFER_DET) {
        // Legacy path: paired outputs (reg/cls) => 6 tensors for 3 heads.
        int output_num = model.GetOutputTensorCount();
        int head_num = output_num / 2;

        for (int index = 0; index < head_num; index++) {
            int reg_idx = index * 2;
            int cls_idx = index * 2 + 1;

            const bsnn_tensor_attr_t& reg_attr = model.GetOutputAttrs()[reg_idx];

            const bsnn_tensor_attr_t& cls_attr = model.GetOutputAttrs()[cls_idx];

            int reg_channels = reg_attr.dims[0];
            int out_w = reg_attr.dims[1];
            int out_h = reg_attr.dims[2];
            int cls_channels = cls_attr.dims[0];
            int num_classes = cls_channels;

            int reg_pitch = reg_attr.pitches[0];
            int cls_pitch = cls_attr.pitches[0];
            if (reg_pitch < reg_channels) reg_pitch = reg_channels;
            if (cls_pitch < cls_channels) cls_pitch = cls_channels;

            int stride = img_w / out_w; 
            int reg_max = reg_channels / 4;
            if (regdeq.size() < reg_max) regdeq.resize(reg_max);

            int8_t *reg_base = (int8_t*)model.OutputVa(reg_idx);
            int8_t *cls_base = (int8_t*)model.OutputVa(cls_idx);

            float quant_scale_reg = BSNN_MODEL::QuantScale(model.GetOutputAttrs()[reg_idx], 0);
            float quant_scale_cls = BSNN_MODEL::QuantScale(model.GetOutputAttrs()[cls_idx], 0);

            float threshold_i8_scale = conf_thresh_inverse / quant_scale_cls;
            int8_t threshold_i8 = (int8_t)std::max(-128.0f, std::min(127.0f, std::floor(threshold_i8_scale)));

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int pixel_offset = h * out_w + w;
                    int8_t* p_cls = cls_base + pixel_offset * cls_pitch;

                    int8_t max_cls_val_i8 = -128;
                    int cls_index = 0;
                    for (int cl = 0; cl < num_classes; cl++) {
                        if (p_cls[cl] > max_cls_val_i8) {
                            max_cls_val_i8 = p_cls[cl];
                            cls_index = cl;
                        }
                    }

                    if (max_cls_val_i8 > threshold_i8) {
                        float cls_max = normal_sigmoid(max_cls_val_i8 * quant_scale_cls);

                        int8_t* p_reg = reg_base + pixel_offset * reg_pitch;
                        float grid_x = w + 0.5f;
                        float grid_y = h + 0.5f;

                        for (int lc = 0; lc < 4; lc++) {
                            float sfsum = 0;
                            float locval = 0;
                            int8_t* p_dist = p_reg + lc * reg_max;

                            for (int df = 0; df < reg_max; df++) {
                                float locvaltemp = fast_exp(p_dist[df] * quant_scale_reg);
                                regdeq[df] = locvaltemp;
                                sfsum += locvaltemp;
                            }
                            for (int df = 0; df < reg_max; df++) {
                                locval += (regdeq[df] / sfsum) * df;
                            }
                            bbox_dfl_local[lc] = locval;
                        }

                        float xmin = (grid_x - bbox_dfl_local[0]) * stride;
                        float ymin = (grid_y - bbox_dfl_local[1]) * stride;
                        float xmax = (grid_x + bbox_dfl_local[2]) * stride;
                        float ymax = (grid_y + bbox_dfl_local[3]) * stride;

                        xmin = std::max(xmin, 0.0f);
                        ymin = std::max(ymin, 0.0f);
                        xmax = std::min(xmax, static_cast<float>(img_w));
                        ymax = std::min(ymax, static_cast<float>(img_h));

                        if (xmin < xmax && ymin < ymax) {
                            ObjInfo temp;
                            temp.x1 = xmin;
                            temp.y1 = ymin;
                            temp.x2 = xmax;
                            temp.y2 = ymax;
                            temp.classId = cls_index + 1;
                            temp.score = cls_max;
                            result.push_back(temp);
                        }
                    }
                }
            }
        }
    } 

    if (fused_output && infer_type == INFER_DET) {
        // Fused DET path
        int head_num = model.GetOutputTensorCount();

        for (int index = 0; index < head_num; index++) {
            if (index >= (int)model.GetOutputAttrs().size()) {
                std::cerr << "Failed to query output tensor attributes for head " << index << std::endl;
                continue;
            }
            const bsnn_tensor_attr_t& out_attr = model.GetOutputAttrs()[index];

            int channels = out_attr.dims[0];   // logical C
            int out_w = out_attr.dims[1];      // W
            int out_h = out_attr.dims[2];      // H

            int pitch = out_attr.pitches[0];     // padded C stride
            if (pitch < channels) pitch = channels;

            int cls_channels = NUM_CLASSES;
            int reg_channels = channels - cls_channels;
            if (reg_channels <= 0 || reg_channels % 4 != 0) {
                std::cerr << "Unexpected channel layout for head " << index << " (channels=" << channels << ")" << std::endl;
                continue;
            }
            int reg_max = reg_channels / 4;
            if (regdeq.size() < reg_max) regdeq.resize(reg_max);

            int stride = img_w / out_w;

            int8_t *base = (int8_t*)model.OutputVa(index);
            std::vector<float> per_ch_scale_vec = BSNN_MODEL::QuantScales(out_attr);
            auto per_ch_scales = [&](int c) { return BSNN_MODEL::ScaleAt(per_ch_scale_vec, c); };

            // For thresholding, use cls channel 0 scale as reference (all cls use per-ch scales below).
            float threshold_i8_scale = conf_thresh_inverse / per_ch_scales(reg_channels);
            int8_t threshold_i8 = (int8_t)std::max(-128.0f, std::min(127.0f, std::floor(threshold_i8_scale)));

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    // HWC with padded C. One pixel stride = pitch.
                    int pixel_offset = h * out_w + w;
                    int8_t* p = base + pixel_offset * pitch;

                    int8_t* p_cls = p + reg_channels;

                    int8_t max_cls_val_i8 = -128;
                    int cls_index = 0;
                    for (int cl = 0; cl < cls_channels; cl++) {
                        int8_t v = p_cls[cl];
                        if (v > max_cls_val_i8) {
                            max_cls_val_i8 = v;
                            cls_index = cl;
                        }
                    }

                    // per-class scale
                    float cls_scale = per_ch_scales(reg_channels + cls_index);
                    float threshold_i8_scale_cls = conf_thresh_inverse / cls_scale;
                    int8_t threshold_i8_cls = (int8_t)std::max(-128.0f, std::min(127.0f, std::floor(threshold_i8_scale_cls)));

                    if (max_cls_val_i8 > threshold_i8_cls) {
                        float cls_max = normal_sigmoid(max_cls_val_i8 * cls_scale);

                        int8_t* p_reg = p; // first part of tensor holds DFL bins
                        float grid_x = w + 0.5f;
                        float grid_y = h + 0.5f;

                        for (int lc = 0; lc < 4; lc++) {
                            float sfsum = 0;
                            float locval = 0;
                            int8_t* p_dist = p_reg + lc * reg_max;

                            for (int df = 0; df < reg_max; df++) {
                                float scale = per_ch_scales(lc * reg_max + df);
                                float locvaltemp = fast_exp(p_dist[df] * scale);
                                regdeq[df] = locvaltemp;
                                sfsum += locvaltemp;
                            }
                            for (int df = 0; df < reg_max; df++) {
                                locval += (regdeq[df] / sfsum) * df;
                            }
                            bbox_dfl_local[lc] = locval;
                        }

                        float xmin = (grid_x - bbox_dfl_local[0]) * stride;
                        float ymin = (grid_y - bbox_dfl_local[1]) * stride;
                        float xmax = (grid_x + bbox_dfl_local[2]) * stride;
                        float ymax = (grid_y + bbox_dfl_local[3]) * stride;

                        xmin = std::max(xmin, 0.0f);
                        ymin = std::max(ymin, 0.0f);
                        xmax = std::min(xmax, static_cast<float>(img_w));
                        ymax = std::min(ymax, static_cast<float>(img_h));

                        if (xmin < xmax && ymin < ymax) {
                            ObjInfo temp;
                            temp.x1 = xmin;
                            temp.y1 = ymin;
                            temp.x2 = xmax;
                            temp.y2 = ymax;
                            temp.classId = cls_index + 1;
                            temp.score = cls_max;
                            result.push_back(temp);
                        }
                    }
                }
            }
        }
    }

    // fused_output seg/pose handled below

    if (fused_output && infer_type == INFER_SEG) {
        // outputs: 0/1/2 detection heads (per-channel), 3/4/5 mask heads (per-tensor), 6 proto (per-tensor, UINT8)
        const int det_head_num = 3;
        const int mask_head_start = 3;
        const int proto_idx = 6;
        int head_num = det_head_num;
        result.clear();

        for (int index = 0; index < head_num; index++) {
            const bsnn_tensor_attr_t& det_attr = model.GetOutputAttrs()[index];
            const bsnn_tensor_attr_t& mask_attr = model.GetOutputAttrs()[mask_head_start + index];

            int channels = det_attr.dims[0];
            int out_w = det_attr.dims[1];
            int out_h = det_attr.dims[2];
            int pitch = det_attr.pitches[0];
            if (pitch < channels) pitch = channels;

            int cls_channels = NUM_CLASSES;
            int reg_channels = channels - cls_channels;
            int reg_max = reg_channels / 4;
            if (reg_channels <= 0 || reg_channels % 4 != 0) continue;
            if (regdeq.size() < reg_max) regdeq.resize(reg_max);

            int stride = img_w / out_w;

            int8_t *det_base = (int8_t*)model.OutputVa(index);
            std::vector<float> det_scale_vec = BSNN_MODEL::QuantScales(det_attr);
            auto det_scales = [&](int c) { return BSNN_MODEL::ScaleAt(det_scale_vec, c); };

            int mask_pitch = mask_attr.pitches[0];
            int mask_channels = mask_attr.dims[0]; // should be 32
            if (mask_pitch < mask_channels) mask_pitch = mask_channels;
            int8_t *mask_base = (int8_t*)model.OutputVa(mask_head_start + index);
            float mask_scale = BSNN_MODEL::QuantScale(model.GetOutputAttrs()[mask_head_start + index], 0);

            float threshold_i8_scale = conf_thresh_inverse / det_scales(reg_channels);
            int8_t threshold_i8 = (int8_t)std::max(-128.0f, std::min(127.0f, std::floor(threshold_i8_scale)));

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int pixel_offset = h * out_w + w;
                    int8_t* p = det_base + pixel_offset * pitch;
                    int8_t* p_cls = p + reg_channels;

                    int8_t max_cls_val_i8 = -128;
                    int cls_index = 0;
                    for (int cl = 0; cl < cls_channels; cl++) {
                        int8_t v = p_cls[cl];
                        if (v > max_cls_val_i8) { max_cls_val_i8 = v; cls_index = cl; }
                    }

                    float cls_scale = det_scales(reg_channels + cls_index);
                    float threshold_i8_scale_cls = conf_thresh_inverse / cls_scale;
                    int8_t threshold_i8_cls = (int8_t)std::max(-128.0f, std::min(127.0f, std::floor(threshold_i8_scale_cls)));
                    if (max_cls_val_i8 <= threshold_i8_cls) continue;

                    float cls_max = normal_sigmoid(max_cls_val_i8 * cls_scale);

                    int8_t* p_reg = p;
                    float grid_x = w + 0.5f;
                    float grid_y = h + 0.5f;

                    for (int lc = 0; lc < 4; lc++) {
                        float sfsum = 0;
                        float locval = 0;
                        int8_t* p_dist = p_reg + lc * reg_max;
                        for (int df = 0; df < reg_max; df++) {
                            float scale = det_scales(lc * reg_max + df);
                            float locvaltemp = fast_exp(p_dist[df] * scale);
                            regdeq[df] = locvaltemp;
                            sfsum += locvaltemp;
                        }
                        for (int df = 0; df < reg_max; df++) {
                            locval += (regdeq[df] / sfsum) * df;
                        }
                        bbox_dfl_local[lc] = locval;
                    }

                    float xmin = (grid_x - bbox_dfl_local[0]) * stride;
                    float ymin = (grid_y - bbox_dfl_local[1]) * stride;
                    float xmax = (grid_x + bbox_dfl_local[2]) * stride;
                    float ymax = (grid_y + bbox_dfl_local[3]) * stride;

                    xmin = std::max(xmin, 0.0f);
                    ymin = std::max(ymin, 0.0f);
                    xmax = std::min(xmax, static_cast<float>(img_w));
                    ymax = std::min(ymax, static_cast<float>(img_h));

                    if (xmin < xmax && ymin < ymax) {
                        ObjInfo temp;
                        temp.x1 = xmin; temp.y1 = ymin; temp.x2 = xmax; temp.y2 = ymax;
                        temp.classId = cls_index + 1;
                        temp.score = cls_max;
                        temp.area = (xmax - xmin) * (ymax - ymin);
                        temp.mask_coeff.resize(mask_channels);

                        int8_t* p_mask = mask_base + pixel_offset * mask_pitch;
                        for (int mc = 0; mc < mask_channels; mc++) {
                            temp.mask_coeff[mc] = static_cast<float>(p_mask[mc]) * mask_scale;
                        }
                        result.push_back(temp);
                    }
                }
            }
        }

        soft_nms(result, det_objs, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
        nms_done = true;
    }

    if (fused_output && infer_type == INFER_POSE) {
        // ================================================================
        // YOLOv8-Pose fused postprocess (aligned with Python reference)
        //
        // Model outputs 6 tensors:
        //   0,1,2 = detection heads (per-channel quantized), C=65 (64 reg + 1 cls)
        //   3,4,5 = keypoint heads (per-tensor quantized),   C=51 (17 * 3)
        //
        // Python flow:
        //   1. DFL decode: softmax(reg[16]) → weighted_sum → dist (l,t,r,b)
        //   2. dist2bbox: anchor ± dist → xywh → * stride
        //   3. cls: sigmoid
        //   4. kpts_decode: (raw * 2.0 + (anchor - 0.5)) * stride, sigmoid(conf)
        //   5. NMS(iou=0.7, conf=0.25, nc=1)
        //   6. scale_boxes / scale_coords → original image space
        // ================================================================
        const int det_head_num = 3;
        const int kpt_head_start = 3;
        const int kpt_channels = 51; // 17 * 3
        result.clear();

        printf("  [POSE] output_ts_num=%d\n", model.GetOutputTensorCount());

        for (int index = 0; index < det_head_num; index++) {
            const bsnn_tensor_attr_t& det_attr = model.GetOutputAttrs()[index];
            const bsnn_tensor_attr_t& kpt_attr = model.GetOutputAttrs()[kpt_head_start + index];

            int channels = det_attr.dims[0];   // D dimension (65)
            int out_w    = det_attr.dims[1];   // W
            int out_h    = det_attr.dims[2];   // H
            int pitch    = det_attr.pitches[0];  // padded D stride
            if (pitch < channels) pitch = channels;

            int kpt_c     = kpt_attr.dims[0];
            int kpt_pitch = kpt_attr.pitches[0];
            if (kpt_pitch < kpt_c) kpt_pitch = kpt_c;

            printf("  [POSE] head %d: det[%d] C=%d W=%d H=%d pitch=%d | kpt[%d] C=%d W=%d H=%d pitch=%d\n",
                   index, index, channels, out_w, out_h, pitch,
                   kpt_head_start + index, kpt_c, kpt_attr.dims[1], kpt_attr.dims[2], kpt_pitch);

            int cls_channels = 1; // pose: only "person" class
            int reg_channels = channels - cls_channels;  // 64
            int reg_max = reg_channels / 4;              // 16
            if (reg_channels <= 0 || reg_channels % 4 != 0) {
                printf("  [POSE] ERROR: unexpected det channels=%d (expected 65)\n", channels);
                continue;
            }
            if (regdeq.size() < (size_t)reg_max) regdeq.resize(reg_max);

            int stride = img_w / out_w;

            int8_t *det_base = (int8_t*)model.OutputVa(index);
            std::vector<float> det_scale_vec = BSNN_MODEL::QuantScales(det_attr);
            auto det_scales = [&](int c) { return BSNN_MODEL::ScaleAt(det_scale_vec, c); };

            int8_t *kpt_base = (int8_t*)model.OutputVa(kpt_head_start + index);
            float   kpt_scale = BSNN_MODEL::QuantScale(model.GetOutputAttrs()[kpt_head_start + index], 0); // per-tensor

            // Pre-compute INT8 threshold for early rejection
            float cls_scale_ch = det_scales(reg_channels); // scale of the single cls channel
            float threshold_i8_f = conf_thresh_inverse / cls_scale_ch;
            int8_t threshold_i8 = (int8_t)std::max(-128.0f, std::min(127.0f, std::floor(threshold_i8_f)));

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int pixel_offset = h * out_w + w;
                    int8_t* p = det_base + pixel_offset * pitch;

                    // --- Classification: single channel (person) ---
                    int8_t cls_val = p[reg_channels];
                    if (cls_val <= threshold_i8) continue;
                    float cls_score = normal_sigmoid(cls_val * cls_scale_ch);
                    if (cls_score < CONF_THRESHOLD) continue;

                    // --- DFL decode (numerically stable softmax, matching torch.softmax) ---
                    float anchor_x = w + 0.5f;
                    float anchor_y = h + 0.5f;

                    for (int lc = 0; lc < 4; lc++) {
                        int8_t* p_dist = p + lc * reg_max;
                        // Step 1: dequantize and find max (for numerical stability)
                        float max_val = -1e30f;
                        for (int df = 0; df < reg_max; df++) {
                            regdeq[df] = p_dist[df] * det_scales(lc * reg_max + df);
                            if (regdeq[df] > max_val) max_val = regdeq[df];
                        }
                        // Step 2: exp(x - max) and sum (matches torch.softmax behavior)
                        float sfsum = 0.0f;
                        for (int df = 0; df < reg_max; df++) {
                            regdeq[df] = std::exp(regdeq[df] - max_val);
                            sfsum += regdeq[df];
                        }
                        // Step 3: weighted sum
                        float locval = 0.0f;
                        for (int df = 0; df < reg_max; df++) {
                            locval += (regdeq[df] / sfsum) * df;
                        }
                        bbox_dfl_local[lc] = locval;
                    }

                    // dist2bbox: anchor - lt, anchor + rb  (xyxy in model space)
                    float x1 = (anchor_x - bbox_dfl_local[0]) * stride;
                    float y1 = (anchor_y - bbox_dfl_local[1]) * stride;
                    float x2 = (anchor_x + bbox_dfl_local[2]) * stride;
                    float y2 = (anchor_y + bbox_dfl_local[3]) * stride;
                    x1 = std::max(x1, 0.0f);
                    y1 = std::max(y1, 0.0f);
                    x2 = std::min(x2, (float)img_w);
                    y2 = std::min(y2, (float)img_h);
                    if (!(x1 < x2 && y1 < y2)) continue;

                    // --- Keypoint decode (matching Python kpts_decode) ---
                    // Python: kx = (raw_x * 2.0 + (anchor_x - 0.5)) * stride
                    //         ky = (raw_y * 2.0 + (anchor_y - 0.5)) * stride
                    //         kc = sigmoid(raw_c)
                    int8_t* p_kpt = kpt_base + pixel_offset * kpt_pitch;
                    std::vector<float> kpts(kpt_channels);
                    for (int kp = 0; kp < 17; ++kp) {
                        float raw_x = static_cast<float>(p_kpt[kp * 3 + 0]) * kpt_scale;
                        float raw_y = static_cast<float>(p_kpt[kp * 3 + 1]) * kpt_scale;
                        float raw_c = static_cast<float>(p_kpt[kp * 3 + 2]) * kpt_scale;
                        kpts[kp * 3 + 0] = (raw_x * 2.0f + (anchor_x - 0.5f)) * stride;
                        kpts[kp * 3 + 1] = (raw_y * 2.0f + (anchor_y - 0.5f)) * stride;
                        kpts[kp * 3 + 2] = normal_sigmoid(raw_c);
                    }

                    ObjInfo temp;
                    temp.x1 = x1; temp.y1 = y1; temp.x2 = x2; temp.y2 = y2;
                    temp.classId = 1;  // person = coco_classes[1] in C++ (coco_classes[0]="__background__")
                    temp.score = cls_score;
                    temp.area = (x2 - x1) * (y2 - y1);
                    temp.keypoints = std::move(kpts);
                    result.push_back(temp);
                }
            }
        }

        soft_nms(result, det_objs, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
        nms_done = true;
    }

    if (!nms_done) {
        soft_nms(result, det_objs, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
    }
}

void yolov8_postprocess_tmp(const BSNN_MODEL& model, std::vector<ObjInfo> &det_objs)
{
    std::vector<ObjInfo> result;
    result.reserve(128); // Pre-allocate to avoid frequent reallocation
    int ret = 0;

    // 1. Query input tensor attributes to determine input image size
    if (model.GetInputAttrs().empty()) {
        std::cerr << "Failed to query input tensor attributes." << std::endl;
        return;
    }
    const bsnn_tensor_attr_t& in_attr = model.GetInputAttrs()[0];
    int img_w = in_attr.dims[0];
    int img_h = in_attr.dims[1];

    // 2. Get number of output tensors
    int output_num = model.GetOutputTensorCount();
    int head_num = output_num / 2;

    // Reusable buffers
    std::vector<float> bbox_dfl_local(4);
    std::vector<float> regdeq(16);
    // int8_t conf_threshold = s(1.0f / CONF_THRESHOLD);

    for (int index = 0; index < head_num; index++) {
        int reg_idx = index * 2;
        int cls_idx = index * 2 + 1;

        const bsnn_tensor_attr_t& reg_attr = model.GetOutputAttrs()[reg_idx];
        const bsnn_tensor_attr_t& cls_attr = model.GetOutputAttrs()[cls_idx];

        // Dynamic attributes
        int reg_channels = reg_attr.dims[0];
        int out_w = reg_attr.dims[1];
        int out_h = reg_attr.dims[2];
        
        int cls_channels = cls_attr.dims[0];
        int num_classes = cls_channels;

        // Use pitch from attributes if available, otherwise fallback to channels (compact)
        // Assuming pitches[0] represents the stride of a single pixel in bytes/elements
        int reg_pitch = reg_attr.pitches[0];
        int cls_pitch = cls_attr.pitches[0];
        
        // Safety check: pitch should be at least the number of channels
        if (reg_pitch < reg_channels) reg_pitch = reg_channels;
        if (cls_pitch < cls_channels) cls_pitch = cls_channels;

        int stride = img_w / out_w; 
        int reg_max = reg_channels / 4;
        if (regdeq.size() < reg_max) regdeq.resize(reg_max);

        // Direct pointer access (Zero-Copy)
        int8_t *reg_base = (int8_t*)model.OutputVa(reg_idx);
        int8_t *cls_base = (int8_t*)model.OutputVa(cls_idx);

        float quant_scale_reg = BSNN_MODEL::QuantScale(model.GetOutputAttrs()[reg_idx], 0);
        float quant_scale_cls = BSNN_MODEL::QuantScale(model.GetOutputAttrs()[cls_idx], 0);

        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                // Calculate offset for current pixel
                // Assuming layout is [H][W][C] or similar where pixels are sequential
                int pixel_offset = h * out_w + w;
                
                int8_t* p_cls = cls_base + pixel_offset * cls_pitch;

                // 1. Find max class score directly in HWC format (Cache Friendly)
                int8_t max_cls_val_i8 = -128; // int8 min
                int cls_index = 0;

                for (int cl = 0; cl < num_classes; cl++) {
                    if (p_cls[cl] > max_cls_val_i8) {
                        max_cls_val_i8 = p_cls[cl];
                        cls_index = cl;
                    }
                }

                // 2. Lazy Evaluation: Check threshold before processing regression
                float cls_max = fast_sigmoid(max_cls_val_i8 * quant_scale_cls);
                
                if (cls_max > CONF_THRESHOLD) {
                    // 3. Process Regression only for valid detections
                    int8_t* p_reg = reg_base + pixel_offset * reg_pitch;
                    
                    float grid_x = w + 0.5f;
                    float grid_y = h + 0.5f;

                    for (int lc = 0; lc < 4; lc++) {
                        float sfsum = 0;
                        float locval = 0;
                        int8_t* p_dist = p_reg + lc * reg_max;

                        for (int df = 0; df < reg_max; df++) {
                            float locvaltemp = fast_exp(p_dist[df] * quant_scale_reg);
                            regdeq[df] = locvaltemp;
                            sfsum += locvaltemp;
                        }
                        for (int df = 0; df < reg_max; df++) {
                            locval += (regdeq[df] / sfsum) * df;
                        }
                        bbox_dfl_local[lc] = locval;
                    }

                    float xmin = (grid_x - bbox_dfl_local[0]) * stride;
                    float ymin = (grid_y - bbox_dfl_local[1]) * stride;
                    float xmax = (grid_x + bbox_dfl_local[2]) * stride;
                    float ymax = (grid_y + bbox_dfl_local[3]) * stride;

                    xmin = std::max(xmin, 0.0f);
                    ymin = std::max(ymin, 0.0f);
                    xmax = std::min(xmax, static_cast<float>(img_w));
                    ymax = std::min(ymax, static_cast<float>(img_h));

                    if (xmin < xmax && ymin < ymax) {
                        ObjInfo temp;
                        temp.x1 = xmin;
                        temp.y1 = ymin;
                        temp.x2 = xmax;
                        temp.y2 = ymax;
                        temp.classId = cls_index + 1;
                        temp.score = cls_max;
                        result.push_back(temp);
                    }
                }
            }
        }
    }

    soft_nms(result, det_objs, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
}

void draw_bboxes(cv::Mat& img, const std::vector<ObjInfo>& result)
{
    float width = img.cols;
    float height = img.rows;
    float gain = MIN(IMG_WIDTH / width, IMG_HEIGHT / height);
    float pad_w = (IMG_WIDTH - width * gain) / 2;
    float pad_h = (IMG_HEIGHT - height * gain) / 2;
    
    stringstream str;
    for (int i = 0; i <  result.size(); i++)
    {
        int cls_id = std::min<int>(coco_classes.size() - 1, std::max(0, result[i].classId));
        std::string cls_name = coco_classes[cls_id];

        float score = result[i].score;
        std::string score_str = std::to_string(score);
        std::string confidence = score_str.substr(0, 4);

        int x1 = (int)((result[i].x1 - pad_w) / gain);
        int y1 = (int)((result[i].y1 - pad_h) / gain);
        int x2 = (int)((result[i].x2 - pad_w) / gain);
        int y2 = (int)((result[i].y2 - pad_h) / gain);

        cv::Scalar color = coco_colors[cls_id];
        cv::rectangle(img, cv::Rect(cv::Point2i(x1, y1), cv::Point2i(x2, y2)), color, 2);
        char buf[64];
        snprintf(buf, sizeof(buf), "%s %.2f", cls_name.c_str(), score);
        string label = buf;
        cv::putText(img, label, cv::Point2i(x1, y1 - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
    }
}

// Build proto matrix (mask basis) from tensor index (seg) into C x (H*W)
bool build_proto(const BSNN_MODEL& model, int proto_idx, cv::Mat &proto, int &proto_w, int &proto_h)
{
    if (proto_idx >= (int)model.GetOutputAttrs().size()) {
        std::cerr << "Failed to query proto tensor attributes." << std::endl;
        return false;
    }
            const bsnn_tensor_attr_t& proto_attr = model.GetOutputAttrs()[proto_idx];

    int c = proto_attr.dims[0];
    proto_w = proto_attr.dims[1];
    proto_h = proto_attr.dims[2];
    int pitch = proto_attr.pitches[0];
    if (pitch < c) pitch = c;

    float scale = BSNN_MODEL::QuantScale(model.GetOutputAttrs()[proto_idx], 0);
    int8_t *base = (int8_t*)model.OutputVa(proto_idx);

    proto = cv::Mat(c, proto_w * proto_h, CV_32F);
    for (int y = 0; y < proto_h; ++y) {
        for (int x = 0; x < proto_w; ++x) {
            int8_t* p = base + (y * proto_w + x) * pitch;
            int col = y * proto_w + x;
            for (int ch = 0; ch < c; ++ch) {
                proto.at<float>(ch, col) = static_cast<float>(p[ch]) * scale;
            }
        }
    }
    return true;
}

static inline float sigmoid_float(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}

// Decode masks and overlay
void draw_seg(cv::Mat& img, const std::vector<ObjInfo>& result, const cv::Mat& proto, int proto_w, int proto_h, int orig_w, int orig_h)
{
    if (result.empty() || proto.empty()) return;

    float gain, pad_w, pad_h;
    calc_gain_pad(orig_w, orig_h, gain, pad_w, pad_h);

    int resize_w = 0, resize_h = 0, pad_left = 0, pad_top = 0;
    letterbox_params(orig_w, orig_h, resize_w, resize_h, pad_left, pad_top);

    int mask_c = proto.rows;
    int proto_hw = proto.cols;
    if (proto_w * proto_h != proto_hw || proto_w <= 0 || proto_h <= 0) {
        std::cerr << "Invalid proto shape" << std::endl;
        return;
    }

    for (const auto &det : result) {
        if (det.mask_coeff.size() != (size_t)mask_c) continue;

        cv::Mat coeff(1, mask_c, CV_32F, const_cast<float*>(det.mask_coeff.data()));
        cv::Mat mask_flat = coeff * proto; // 1 x HW
        for (int i = 0; i < mask_flat.cols; ++i) {
            float &v = mask_flat.at<float>(0, i);
            v = sigmoid_float(v);
        }
        cv::Mat mask(proto_h, proto_w, CV_32F, mask_flat.data);

        // bbox in model coords (before removing pad)
        float x1m = det.x1;
        float y1m = det.y1;
        float x2m = det.x2;
        float y2m = det.y2;

        // downsample bbox to proto space (model space ratios)
        float width_ratio = proto_w / static_cast<float>(IMG_WIDTH);
        float height_ratio = proto_h / static_cast<float>(IMG_HEIGHT);
        int px1 = static_cast<int>(x1m * width_ratio);
        int py1 = static_cast<int>(y1m * height_ratio);
        int px2 = static_cast<int>(x2m * width_ratio);
        int py2 = static_cast<int>(y2m * height_ratio);
        px1 = std::max(0, std::min(px1, proto_w - 1));
        py1 = std::max(0, std::min(py1, proto_h - 1));
        px2 = std::max(px1 + 1, std::min(px2, proto_w));
        py2 = std::max(py1 + 1, std::min(py2, proto_h));

        // crop_like crop_mask: zero outside bbox
        cv::Mat mask_crop_full = cv::Mat::zeros(proto_h, proto_w, CV_32F);
        mask(cv::Rect(px1, py1, px2 - px1, py2 - py1)).copyTo(mask_crop_full(cv::Rect(px1, py1, px2 - px1, py2 - py1)));

        cv::Mat mask_crop = mask_crop_full;

        // upsample to model input size
        cv::Mat mask_up;
        cv::resize(mask_crop, mask_up, cv::Size(IMG_WIDTH, IMG_HEIGHT), 0, 0, cv::INTER_LINEAR);

        int crop_w = std::max(1, std::min(resize_w, IMG_WIDTH - pad_left));
        int crop_h = std::max(1, std::min(resize_h, IMG_HEIGHT - pad_top));
        cv::Rect roi(pad_left, pad_top, crop_w, crop_h);
        roi &= cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT);
        cv::Mat mask_nopad = mask_up(roi);

        cv::Mat mask_original;
        cv::resize(mask_nopad, mask_original, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

        // bbox mapped to original coords for visualization
        float x1o = (det.x1 - pad_w) / gain;
        float y1o = (det.y1 - pad_h) / gain;
        float x2o = (det.x2 - pad_w) / gain;
        float y2o = (det.y2 - pad_h) / gain;
        x1o = std::max(0.0f, x1o);
        y1o = std::max(0.0f, y1o);
        x2o = std::min(static_cast<float>(orig_w), x2o);
        y2o = std::min(static_cast<float>(orig_h), y2o);

        cv::Mat mask_bin_float;
        cv::threshold(mask_original, mask_bin_float, 0.5, 1.0, cv::THRESH_BINARY);
        cv::Mat mask_bin;
        mask_bin_float.convertTo(mask_bin, CV_8U);

        int cls_idx = std::min<int>(coco_colors.size() - 1, std::max(0, det.classId));
        cv::Scalar color = darker(coco_colors[cls_idx]);

        // colorize only masked pixels, keep backgrounds untouched
        cv::Mat overlay = img.clone();
        overlay.setTo(color, mask_bin);
        cv::addWeighted(overlay, 0.6, img, 0.4, 0.0, img);
        // draw bbox on top
        cv::rectangle(img, cv::Rect(cv::Point2i(static_cast<int>(x1o), static_cast<int>(y1o)),
                                    cv::Point2i(static_cast<int>(x2o), static_cast<int>(y2o))),
                      color, 2);
        // label with class name and confidence
        std::string cls_name = coco_classes[cls_idx];
        char buf[64];
        snprintf(buf, sizeof(buf), "%s %.2f", cls_name.c_str(), det.score);
        cv::putText(img, buf, cv::Point2i(static_cast<int>(x1o), static_cast<int>(y1o) - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
    }
}

// ================================================================
// Pose visualization — faithfully matching Python pose_plot()
// ================================================================

// skeleton: 19 limb connections, 1-indexed (from colors.py)
static const int skeleton[19][2] = {
    {16,14},{14,12},{17,15},{15,13},{12,13},{6,12},{7,13},{6,7},
    {6,8},{7,9},{8,10},{9,11},{2,3},{1,2},{1,3},{2,4},{3,5},{4,6},{5,7}
};

// pose_palette from colors.py (RGB values, but cv2 draws in BGR, same as Python)
// Python defines them as RGB but passes directly to cv2 (which interprets as BGR)
// We replicate the EXACT same numeric values, so the visual result matches.
//
// palette indices: [0]=[255,128,0] [7]=[255,51,255] [9]=[51,153,255] [16]=[0,255,0]
//
// kpt_color = pose_palette[[16,16,16,16,16, 0,0,0,0,0,0, 9,9,9,9,9,9]]
static const cv::Scalar kpt_color[17] = {
    {0,255,0},   {0,255,0},   {0,255,0},   {0,255,0},   {0,255,0},     // kp 0-4: palette[16]
    {255,128,0}, {255,128,0}, {255,128,0}, {255,128,0}, {255,128,0}, {255,128,0}, // kp 5-10: palette[0]
    {51,153,255},{51,153,255},{51,153,255},{51,153,255},{51,153,255},{51,153,255}   // kp 11-16: palette[9]
};

// limb_color = pose_palette[[9,9,9,9, 7,7,7, 0,0,0,0,0, 16,16,16,16,16,16,16]]
static const cv::Scalar limb_color[19] = {
    {51,153,255},{51,153,255},{51,153,255},{51,153,255},                             // limb 0-3: palette[9]
    {255,51,255},{255,51,255},{255,51,255},                                          // limb 4-6: palette[7]
    {255,128,0}, {255,128,0}, {255,128,0}, {255,128,0}, {255,128,0},                // limb 7-11: palette[0]
    {0,255,0},   {0,255,0},   {0,255,0},   {0,255,0},   {0,255,0},{0,255,0},{0,255,0} // limb 12-18: palette[16]
};


void draw_pose(cv::Mat& img, const std::vector<ObjInfo>& result, int orig_w, int orig_h)
{
    if (result.empty()) return;

    // Python: gain = min(640/H, 640/W)
    float gain = std::min((float)IMG_WIDTH / orig_w, (float)IMG_HEIGHT / orig_h);

    // Python scale_boxes pad: round((640 - dim*gain)/2 - 0.1)  (integer, with -0.1 offset)
    float box_pad_x = std::round(((float)IMG_WIDTH  - orig_w * gain) / 2.0f - 0.1f);
    float box_pad_y = std::round(((float)IMG_HEIGHT - orig_h * gain) / 2.0f - 0.1f);

    // Python scale_coords pad: (640 - dim*gain)/2  (exact float, no rounding)
    float kpt_pad_x = ((float)IMG_WIDTH  - orig_w * gain) / 2.0f;
    float kpt_pad_y = ((float)IMG_HEIGHT - orig_h * gain) / 2.0f;

    printf("  [draw_pose] gain=%.6f box_pad=(%.1f, %.1f) kpt_pad=(%.4f, %.4f) orig=%dx%d\n",
           gain, box_pad_x, box_pad_y, kpt_pad_x, kpt_pad_y, orig_w, orig_h);

    for (size_t di = 0; di < result.size(); di++) {
        const auto &det = result[di];
        if (det.keypoints.size() < 51) continue;

        // ---- Scale keypoints: Python scale_coords (float pad, no rounding) ----
        // Python: coords[...,0] -= pad[0]; coords[...,0] /= gain; clamp(0, W)
        // Python: coords[...,1] -= pad[1]; coords[...,1] /= gain; clamp(0, H)
        float kpts_scaled[17][3]; // [x, y, conf] in original image coords (float)
        for (int k = 0; k < 17; k++) {
            float mx = det.keypoints[k*3 + 0]; // model space
            float my = det.keypoints[k*3 + 1]; // model space
            float mc = det.keypoints[k*3 + 2]; // sigmoid confidence

            float ox = (mx - kpt_pad_x) / gain;
            float oy = (my - kpt_pad_y) / gain;
            // clip_coords: clamp to [0, orig_w] / [0, orig_h]
            ox = std::max(0.0f, std::min(ox, (float)orig_w));
            oy = std::max(0.0f, std::min(oy, (float)orig_h));

            kpts_scaled[k][0] = ox;
            kpts_scaled[k][1] = oy;
            kpts_scaled[k][2] = mc;
        }

        // ---- Draw keypoint circles ----
        // Python:
        //   for idx, k in enumerate(keypoints):
        //       x_coord, y_coord = k[0], k[1]
        //       if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
        //           if len(k)==3 and k[2] < 0.5: continue
        //           cv2.circle(orig_img, (int(x_coord), int(y_coord)), 5, color, -1, LINE_AA)
        for (int k = 0; k < 17; k++) {
            float x = kpts_scaled[k][0];
            float y = kpts_scaled[k][1];
            float c = kpts_scaled[k][2];

            // Python boundary filter: x_coord % shape[1] != 0  (skip if x is multiple of orig_w, i.e. 0 or orig_w)
            int ix = (int)x;
            int iy = (int)y;
            bool at_boundary = (ix % orig_w == 0) || (iy % orig_h == 0);
            if (at_boundary) continue;
            if (c < 0.5f) continue;

            cv::circle(img, cv::Point(ix, iy), 5, kpt_color[k], -1, cv::LINE_AA);
        }

        // ---- Draw skeleton lines ----
        // Python:
        //   for limb_idx, sk in enumerate(skeleton):
        //       pos1 = (int(kpts[sk[0]-1, 0]), int(kpts[sk[0]-1, 1]))
        //       pos2 = (int(kpts[sk[1]-1, 0]), int(kpts[sk[1]-1, 1]))
        //       conf1 = kpts[sk[0]-1, 2], conf2 = kpts[sk[1]-1, 2]
        //       if conf1 < 0.5 or conf2 < 0.5: continue
        //       if pos1[0] % W == 0 or pos1[1] % H == 0 or pos1[0] < 0 or pos1[1] < 0: continue
        //       if pos2[0] % W == 0 or pos2[1] % H == 0 or pos2[0] < 0 or pos2[1] < 0: continue
        //       cv2.line(...)
        for (int li = 0; li < 19; li++) {
            int a = skeleton[li][0] - 1;  // 1-indexed → 0-indexed
            int b = skeleton[li][1] - 1;

            float c1 = kpts_scaled[a][2];
            float c2 = kpts_scaled[b][2];
            if (c1 < 0.5f || c2 < 0.5f) continue;

            int ax = (int)kpts_scaled[a][0], ay = (int)kpts_scaled[a][1];
            int bx = (int)kpts_scaled[b][0], by = (int)kpts_scaled[b][1];

            // Python boundary filter
            if (ax % orig_w == 0 || ay % orig_h == 0 || ax < 0 || ay < 0) continue;
            if (bx % orig_w == 0 || by % orig_h == 0 || bx < 0 || by < 0) continue;

            cv::line(img, cv::Point(ax, ay), cv::Point(bx, by), limb_color[li], 2, cv::LINE_AA);
        }

        // ---- Draw bbox: Python scale_boxes (rounded pad) ----
        // Python: boxes[...,0] -= round(pad-0.1); boxes /= gain; clip; round()
        float bx1 = (det.x1 - box_pad_x) / gain;
        float by1 = (det.y1 - box_pad_y) / gain;
        float bx2 = (det.x2 - box_pad_x) / gain;
        float by2 = (det.y2 - box_pad_y) / gain;
        // clip_boxes
        bx1 = std::max(0.0f, std::min(bx1, (float)orig_w));
        by1 = std::max(0.0f, std::min(by1, (float)orig_h));
        bx2 = std::max(0.0f, std::min(bx2, (float)orig_w));
        by2 = std::max(0.0f, std::min(by2, (float)orig_h));
        // Python: .round() on final coords
        int ix1 = (int)std::round(bx1), iy1 = (int)std::round(by1);
        int ix2 = (int)std::round(bx2), iy2 = (int)std::round(by2);

        // Use coco_colors[classId] — classId=1 → person color (matching Python coco_colors[0])
        int cls_idx = std::min<int>((int)coco_colors.size() - 1, std::max(0, det.classId));
        cv::Scalar bbox_color = coco_colors[cls_idx];
        cv::rectangle(img, cv::Point(ix1, iy1), cv::Point(ix2, iy2), bbox_color, 2);

        // Label: "person (0.85)"
        const std::string& cls_name = coco_classes[cls_idx];
        char buf[64];
        snprintf(buf, sizeof(buf), "%s (%.2f)", cls_name.c_str(), det.score);
        cv::putText(img, buf, cv::Point(ix1, iy1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2);

        printf("  [draw_pose] det[%zu]: bbox_orig=(%d,%d,%d,%d) score=%.4f\n",
               di, ix1, iy1, ix2, iy2, det.score);
    }
}