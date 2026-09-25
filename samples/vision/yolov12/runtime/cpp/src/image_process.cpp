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


static inline cv::Scalar darker(const cv::Scalar &c, double factor = 0.8)
{
    return cv::Scalar(std::max(0.0, c[0] * factor),
                      std::max(0.0, c[1] * factor),
                      std::max(0.0, c[2] * factor));
}

void yolo_det_postprocess(const BSNN_MODEL& model, std::vector<ObjInfo> &det_objs, bool fused_output)
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

    if (!fused_output) {
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
    else {
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

    if (!nms_done) {
        soft_nms(result, det_objs, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
    }
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

        cv::Scalar color = darker(coco_colors[cls_id]);
        cv::rectangle(img, cv::Rect(cv::Point2i(x1, y1), cv::Point2i(x2, y2)), color, 1);
        string label = cls_name + " " + confidence;
        cv::putText(img, label, cv::Point2i(x1 + 10, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
    }
}
