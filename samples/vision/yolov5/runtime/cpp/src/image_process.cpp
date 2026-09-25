#include "image_process.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include "coco_info.h"

using namespace std;

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

void generate_bboxes(const int8_t* bsnn_out, const float* anchors, int out_h, int out_w, int stride,
                     float dequant_scale, std::vector<ObjInfo> &objects, int src_pitch = OUT_CHANNELS)
{
    // Build a small LUT to replace repeated int8->float*dequant multiplications.
    float lut[256];
    for (int u = 0; u < 256; ++u) {
        int8_t q = static_cast<int8_t>(u);
        lut[u] = static_cast<float>(q) * dequant_scale;
    }

    for (int y = 0; y < out_h; y++)
    {
        for (int x = 0; x < out_w; x++)
        {
            size_t loc = static_cast<size_t>(y) * static_cast<size_t>(out_w) + static_cast<size_t>(x);
            size_t base = loc * static_cast<size_t>(src_pitch);
            for (int q = 0; q < 3; q++)
            {
                float anchor_w = anchors[q * 2];
                float anchor_h = anchors[q * 2 + 1];
                size_t idx = base + static_cast<size_t>(q * PER_ANCHORS_CHN);

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

                // use LUT to get dequantized float values
                float class_score = lut[static_cast<uint8_t>(class_score_)];
                float box_score = lut[static_cast<uint8_t>(bsnn_out[idx + 4])];
                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= CONF_THRESHOLD)
                {
                    float dx = sigmoid(lut[static_cast<uint8_t>(bsnn_out[idx + 0])]);
                    float dy = sigmoid(lut[static_cast<uint8_t>(bsnn_out[idx + 1])]);
                    float dw = sigmoid(lut[static_cast<uint8_t>(bsnn_out[idx + 2])]);
                    float dh = sigmoid(lut[static_cast<uint8_t>(bsnn_out[idx + 3])]);

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
    std::vector<ObjInfo> det_objs;
    const uint32_t nout = model.GetOutputTensorCount();
    // stride 8
    {
        float anchors[] = {10.f, 13.f, 16.f, 30.f, 33.f, 23.f};
        int out_h = IMG_HEIGHT / strides[0];
        int out_w = IMG_WIDTH / strides[0];
        int pitch = REG_PITCH;
        if (nout > 0) {
            pitch = std::max(static_cast<int>(model.GetOutputAttrs()[0].pitches[0]), OUT_CHANNELS);
            out_w = static_cast<int>(model.GetOutputAttrs()[0].dims[1]);
            out_h = static_cast<int>(model.GetOutputAttrs()[0].dims[2]);
        }
        int8_t* out_buf_i8 = (int8_t*)model.OutputVa(0);
        generate_bboxes(out_buf_i8, anchors, out_h, out_w, strides[0],
                        BSNN_MODEL::QuantScale(model.GetOutputAttrs()[0], 0), det_objs, pitch);
    }

    // stride 16
    {
        float anchors[] = {30.f, 61.f, 62.f, 45.f, 59.f, 119.f};
        int out_h = IMG_HEIGHT / strides[1];
        int out_w = IMG_WIDTH / strides[1];
        int pitch = REG_PITCH;
        if (nout > 1) {
            pitch = std::max(static_cast<int>(model.GetOutputAttrs()[1].pitches[0]), OUT_CHANNELS);
            out_w = static_cast<int>(model.GetOutputAttrs()[1].dims[1]);
            out_h = static_cast<int>(model.GetOutputAttrs()[1].dims[2]);
        }
        int8_t* out_buf_i8 = (int8_t*)model.OutputVa(1);
        generate_bboxes(out_buf_i8, anchors, out_h, out_w, strides[1],
                        BSNN_MODEL::QuantScale(model.GetOutputAttrs()[1], 0), det_objs, pitch);
    }

    // stride 32
    {
        float anchors[] = {116.f, 90.f, 156.f, 198.f, 373.f, 326.f};
        int out_h = IMG_HEIGHT / strides[2];
        int out_w = IMG_WIDTH / strides[2];
        int pitch = REG_PITCH;
        if (nout > 2) {
            pitch = std::max(static_cast<int>(model.GetOutputAttrs()[2].pitches[0]), OUT_CHANNELS);
            out_w = static_cast<int>(model.GetOutputAttrs()[2].dims[1]);
            out_h = static_cast<int>(model.GetOutputAttrs()[2].dims[2]);
        }
        int8_t* out_buf_i8 = (int8_t*)model.OutputVa(2);
        generate_bboxes(out_buf_i8, anchors, out_h, out_w, strides[2],
                        BSNN_MODEL::QuantScale(model.GetOutputAttrs()[2], 0), det_objs, pitch);
    }

    soft_nms(det_objs, result, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
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
        int cls_id = result[i].classId;
        std::string cls_name = coco_classes[cls_id];
        cv::Scalar color = coco_colors[cls_id];

        float score = result[i].score;
        std::string score_str = std::to_string(score);
        std::string confidence = score_str.substr(0, 4);

        int x1 = (int)((result[i].x1 - pad_w) / gain);
        int y1 = (int)((result[i].y1 - pad_h) / gain);
        int x2 = (int)((result[i].x2 - pad_w) / gain);
        int y2 = (int)((result[i].y2 - pad_h) / gain);

        cv::rectangle(img, cv::Rect(cv::Point2i(x1, y1), cv::Point2i(x2, y2)), color, 1);
        string label = cls_name + " " + confidence;
        cv::putText(img, label, cv::Point2i(x1 + 10, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
    }
}
