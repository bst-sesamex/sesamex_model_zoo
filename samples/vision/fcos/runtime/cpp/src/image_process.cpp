#include "image_process.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "coco_info.h"

using namespace std;

// ============================================================
// sigmoid 工具函数
// ============================================================
inline float normal_sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// ============================================================
// 前处理: 与 YOLO 完全一致 (resize + letterbox, pad=114, uint8)
// ============================================================
void preprocess(const cv::Mat& src, cv::Mat& dst)
{
    double width = src.cols;
    double height = src.rows;
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

// ============================================================
// FCOS 后处理
//
// 模型有 15 个输出 tensor:
//   index 0 ~ 4  : classification (80 channels), 对应 FPN level 0~4
//   index 5 ~ 9  : bbox regression (4 channels: l, t, r, b)
//   index 10 ~ 14: centerness (1 channel)
//
// 每个 tensor 的内存布局是 DWH (D=channel 是最低维度), 有 pitch 对齐
//   dims[0] = D (channels)
//   dims[1] = W
//   dims[2] = H
//   pitches[0] = padded D stride (>= dims[0])
//
// 数据类型: INT8, 使用 per-channel 反量化:
//   float_val = int8_val * scale[channel]
//
// 解码逻辑 (对照 inference.py 的 _decode_fcos):
//   对每个 FPN level (stride in [8, 16, 32, 64, 128]):
//     对每个像素位置 (h, w):
//       loc_x = w * stride + stride / 2
//       loc_y = h * stride + stride / 2
//       cls_scores = sigmoid(cls_tensor[h, w, :])     // 80 个类别
//       centerness = sigmoid(ctr_tensor[h, w, 0])
//       score = cls_score * centerness
//       如果 score > PRE_NMS_THRESHOLD:
//         l, t, r, b = exp(box_tensor[h, w, :])       // 4 个回归值
//         x1 = loc_x - l
//         y1 = loc_y - t
//         x2 = loc_x + r
//         y2 = loc_y + b
//         clamp 到 [0, img_w] / [0, img_h]
//         最终 score = sqrt(cls_score * centerness)  // 与 inference.py 一致
// ============================================================
void fcos_det_postprocess(const BSNN_MODEL& model,
                          std::vector<ObjInfo> &det_objs,
                          float score_thr)
{
    std::vector<ObjInfo> result;
    result.reserve(256);
    std::vector<ObjInfo> level_candidates;  // 每个 level 的候选框
    level_candidates.reserve(512);

    // 查询输入 tensor 属性获取输入图像尺寸
    if (model.GetInputAttrs().empty()) {
        std::cerr << "Failed to query input tensor attributes." << std::endl;
        return;
    }
    const bsnn_tensor_attr_t& in_attr = model.GetInputAttrs()[0];

    // 输入 tensor 也是 DWH 布局: dims[0]=D(3), dims[1]=W, dims[2]=H
    // 但也可能是 WHD, 需要自动判断
    int img_w, img_h;
    if (in_attr.dims[2] == 3) {
        // WHD: dims[0]=W, dims[1]=H, dims[2]=D=3
        img_w = in_attr.dims[0];
        img_h = in_attr.dims[1];
    } else if (in_attr.dims[0] == 3) {
        // DWH: dims[0]=D=3, dims[1]=W, dims[2]=H
        img_w = in_attr.dims[1];
        img_h = in_attr.dims[2];
    } else {
        // 默认按 DWH
        img_w = in_attr.dims[1];
        img_h = in_attr.dims[2];
    }

    printf("-> FCOS input: %d x %d, output tensors: %d\n", img_w, img_h, model.GetOutputTensorCount());

    // 遍历 5 个 FPN level
    for (int level = 0; level < FCOS_LEVELS; level++) {
        int cls_idx = level;              // 0-4
        int box_idx = level + FCOS_LEVELS;     // 5-9
        int ctr_idx = level + FCOS_LEVELS * 2; // 10-14

        int stride = FCOS_STRIDES[level];

        // ---- 查询 cls tensor 属性 ----
        if (cls_idx >= (int)model.GetOutputAttrs().size()) {
            std::cerr << "Failed to query cls tensor " << cls_idx << std::endl;
            continue;
        }
        const bsnn_tensor_attr_t& cls_attr = model.GetOutputAttrs()[cls_idx];

        // ---- 查询 box tensor 属性 ----
        if (box_idx >= (int)model.GetOutputAttrs().size()) {
            std::cerr << "Failed to query box tensor " << box_idx << std::endl;
            continue;
        }
        const bsnn_tensor_attr_t& box_attr = model.GetOutputAttrs()[box_idx];

        // ---- 查询 centerness tensor 属性 ----
        if (ctr_idx >= (int)model.GetOutputAttrs().size()) {
            std::cerr << "Failed to query ctr tensor " << ctr_idx << std::endl;
            continue;
        }
        const bsnn_tensor_attr_t& ctr_attr = model.GetOutputAttrs()[ctr_idx];

        // DWH 布局: dims[0]=D(channels), dims[1]=W, dims[2]=H
        int cls_channels = cls_attr.dims[0];  // should be 80
        int out_w        = cls_attr.dims[1];
        int out_h        = cls_attr.dims[2];

        int box_channels = box_attr.dims[0];  // should be 4
        int ctr_channels = ctr_attr.dims[0];  // should be 1

        int cls_pitch = cls_attr.pitches[0];
        int box_pitch = box_attr.pitches[0];
        int ctr_pitch = ctr_attr.pitches[0];

        // pitch 至少要 >= channels
        if (cls_pitch < cls_channels) cls_pitch = cls_channels;
        if (box_pitch < box_channels) box_pitch = box_channels;
        if (ctr_pitch < ctr_channels) ctr_pitch = ctr_channels;

        // 获取数据指针
        uint8_t *cls_base = model.OutputVa(cls_idx);
        uint8_t *box_base = model.OutputVa(box_idx);
        uint8_t *ctr_base = model.OutputVa(ctr_idx);

        // per-channel 反量化 scale
        std::vector<float> cls_scale_vec = BSNN_MODEL::QuantScales(cls_attr);
        auto cls_scales = [&](int c) { return BSNN_MODEL::ScaleAt(cls_scale_vec, c); };
        std::vector<float> box_scale_vec = BSNN_MODEL::QuantScales(box_attr);
        auto box_scales = [&](int c) { return BSNN_MODEL::ScaleAt(box_scale_vec, c); };
        std::vector<float> ctr_scale_vec = BSNN_MODEL::QuantScales(ctr_attr);
        auto ctr_scales = [&](int c) { return BSNN_MODEL::ScaleAt(ctr_scale_vec, c); };
        // printf(" cls_scales(0...9): ");
        // for (int c = 0; c < std::min(cls_channels, 10); c++) {
        //     printf("%.4f ", cls_scales(c));
        // }
        // printf("...\n");

        // printf(" box_scales(0...3): ");
        // for (int c = 0; c < box_channels; c++) {
        //     printf("%.6f ", box_scales(c));
        // }
        // printf("...\n");

        // printf("ctr_scales: %.6f\n", ctr_scales(0));

        // 仅在第一个level打印调试信息
        if (level == 0) {
            printf("-> Level %d: cls_ch=%d, box_ch=%d, ctr_ch=%d, out_w=%d, out_h=%d, stride=%d\n",
                   level, cls_channels, box_channels, ctr_channels, out_w, out_h, stride);
            printf("   cls_pitch=%d, box_pitch=%d, ctr_pitch=%d\n",
                   cls_pitch, box_pitch, ctr_pitch);
        }

        // 遍历每个像素位置 (DWH: 内存排列是 H行 × W列, 每个像素占 pitch 字节)
        level_candidates.clear();
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                int pixel_offset = h * out_w + w;

                uint8_t* p_cls = cls_base + pixel_offset * cls_pitch;
                uint8_t* p_ctr = ctr_base + pixel_offset * ctr_pitch;

                // ---- 反量化 centerness 并 sigmoid ----
                float ctr_raw = (ctr_attr.dtype == BSNN_TENSOR_UINT8)
                                    ? static_cast<float>(p_ctr[0])
                                    : static_cast<float>(static_cast<int8_t>(p_ctr[0]));
                float ctr_val = normal_sigmoid(ctr_raw * ctr_scales(0));

                // ---- 找最大类别得分 ----
                // 先在 int8 域快速过滤
                int max_cls_raw = -129;
                int best_cls = 0;
                for (int c = 0; c < cls_channels; c++) {
                    int cls_value = (cls_attr.dtype == BSNN_TENSOR_UINT8)
                                        ? static_cast<int>(p_cls[c])
                                        : static_cast<int>(static_cast<int8_t>(p_cls[c]));
                    if (cls_value > max_cls_raw) {
                        max_cls_raw = cls_value;
                        best_cls = c;
                    }
                }

                // 反量化 + sigmoid
                float cls_score = normal_sigmoid(max_cls_raw * cls_scales(best_cls));

                // 复合得分 = cls * centerness
                float score = cls_score * ctr_val;
                if (score <= PRE_NMS_THRESHOLD) continue;

                // ---- 解码 bbox ----
                // inference.py: loc = (w*stride + stride//2, h*stride + stride//2)
                float loc_x = w * stride + stride / 2.0f;
                float loc_y = h * stride + stride / 2.0f;

                uint8_t* p_box = box_base + pixel_offset * box_pitch;

                // box regression 值: l, t, r, b
                // inference.py 中 reg = box_reg[0].permute(1,2,0).reshape(-1,4)
                // 然后 boxes = [loc_x - reg[:,0], loc_y - reg[:,1], loc_x + reg[:,2], loc_y + reg[:,3]]
                // FCOS 原始输出是 exp(x) 形式? 不, inference.py 中直接用 raw 值不做 exp
                // 所以板端也直接用反量化后的值
                auto box_value = [&](int channel) {
                    return (box_attr.dtype == BSNN_TENSOR_UINT8)
                               ? static_cast<float>(p_box[channel]) * box_scales(channel)
                               : static_cast<float>(static_cast<int8_t>(p_box[channel])) * box_scales(channel);
                };
                float l = box_value(0);
                float t = box_value(1);
                float r = box_value(2);
                float b = box_value(3);

                float x1 = loc_x - l;
                float y1 = loc_y - t;
                float x2 = loc_x + r;
                float y2 = loc_y + b;

                // clamp
                x1 = std::max(x1, 0.0f);
                y1 = std::max(y1, 0.0f);
                x2 = std::min(x2, (float)img_w);
                y2 = std::min(y2, (float)img_h);

                if (x1 >= x2 || y1 >= y2) continue;

                // inference.py: all_scores.append(torch.sqrt(per_scores))
                // 最终 score = sqrt(cls * centerness)
                float final_score = std::sqrt(score);

                ObjInfo obj;
                obj.x1 = x1;
                obj.y1 = y1;
                obj.x2 = x2;
                obj.y2 = y2;
                obj.classId = best_cls + 1;  // 1-based, 与 coco_classes 对齐 (index 0 = __background__)
                obj.score = final_score;
                level_candidates.push_back(obj);
            }
        }

        // per-level topk: 与 inference.py 的 pre_nms_topk=1000 一致
        if ((int)level_candidates.size() > PRE_NMS_TOPK) {
            std::partial_sort(level_candidates.begin(),
                              level_candidates.begin() + PRE_NMS_TOPK,
                              level_candidates.end(),
                              [](const ObjInfo& a, const ObjInfo& b) { return a.score > b.score; });
            level_candidates.resize(PRE_NMS_TOPK);
        }
        result.insert(result.end(), level_candidates.begin(), level_candidates.end());
    }

    printf("-> FCOS before NMS: %zu candidates\n", result.size());

    // ================================================================
    // 第1轮: per-class NMS (与 inference.py 的 batched_nms 对齐)
    // ================================================================
    std::vector<ObjInfo> nms_out;
    soft_nms(result, nms_out, 3, 0.5, NMS_THRESHOLD, CONF_THRESHOLD);
    printf("-> FCOS after per-class NMS: %zu\n", nms_out.size());

    // ================================================================
    // 第2轮: class-agnostic NMS
    // 原因: INT8 量化噪声导致相邻像素对同一目标选出不同 best_cls,
    //       per-class NMS 中不同类别框互不压制, 产生大量重复框.
    //       Python float32 模型不存在此问题, 因为类别预测很一致.
    //       追加一轮忽略类别的 NMS 去除跨类别重复框.
    // ================================================================
    std::sort(nms_out.begin(), nms_out.end(),
              [](const ObjInfo& a, const ObjInfo& b) { return a.score > b.score; });

    std::vector<bool> suppressed(nms_out.size(), false);
    std::vector<ObjInfo> agnostic_out;
    agnostic_out.reserve(nms_out.size());

    auto calc_iou = [](const ObjInfo& a, const ObjInfo& b) -> float {
        float inter_x1 = std::max(a.x1, b.x1);
        float inter_y1 = std::max(a.y1, b.y1);
        float inter_x2 = std::min(a.x2, b.x2);
        float inter_y2 = std::min(a.y2, b.y2);
        float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        float union_area = area_a + area_b - inter_area;
        return union_area > 0.0f ? inter_area / union_area : 0.0f;
    };

    for (int i = 0; i < (int)nms_out.size(); i++) {
        if (suppressed[i]) continue;
        agnostic_out.push_back(nms_out[i]);
        for (int j = i + 1; j < (int)nms_out.size(); j++) {
            if (!suppressed[j] && calc_iou(nms_out[i], nms_out[j]) > NMS_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }
    printf("-> FCOS after class-agnostic NMS: %zu\n", agnostic_out.size());

    // ================================================================
    // post_nms_topk + score_thr 过滤
    // ================================================================
    det_objs.clear();
    for (int i = 0; i < (int)agnostic_out.size() && (int)det_objs.size() < POST_NMS_TOPK; i++) {
        if (agnostic_out[i].score >= score_thr) {
            det_objs.push_back(agnostic_out[i]);
        }
    }

    printf("-> FCOS final detections (score>=%.2f): %zu\n", score_thr, det_objs.size());
}

// ============================================================
// 画框: 将检测结果还原到原图坐标并绘制
// 坐标还原逻辑 (与 inference.py 一致):
//   x_orig = (x_model - pad_left) / scale
//   y_orig = (y_model - pad_top)  / scale
// ============================================================
static inline cv::Scalar darker(const cv::Scalar &c, double factor = 0.8)
{
    return cv::Scalar(std::max(0.0, c[0] * factor),
                      std::max(0.0, c[1] * factor),
                      std::max(0.0, c[2] * factor));
}

void draw_bboxes_fcos(cv::Mat& img, const std::vector<ObjInfo>& result,
                      const LetterBoxInfo& lb_info)
{
    float gain = lb_info.scale;
    float pad_w = lb_info.pad_left;
    float pad_h = lb_info.pad_top;

    for (int i = 0; i < (int)result.size(); i++)
    {
        int cls_id = std::min<int>(coco_classes.size() - 1, std::max(0, result[i].classId));
        std::string cls_name = coco_classes[cls_id];

        float score = result[i].score;
        std::string score_str = std::to_string(score);
        std::string confidence = score_str.substr(0, 4);

        // 还原到原图坐标
        int x1 = (int)((result[i].x1 - pad_w) / gain);
        int y1 = (int)((result[i].y1 - pad_h) / gain);
        int x2 = (int)((result[i].x2 - pad_w) / gain);
        int y2 = (int)((result[i].y2 - pad_h) / gain);

        // clamp 到原图范围
        x1 = std::max(0, std::min(x1, lb_info.orig_w));
        y1 = std::max(0, std::min(y1, lb_info.orig_h));
        x2 = std::max(0, std::min(x2, lb_info.orig_w));
        y2 = std::max(0, std::min(y2, lb_info.orig_h));

        cv::Scalar color = darker(coco_colors[cls_id]);
        cv::rectangle(img, cv::Rect(cv::Point2i(x1, y1), cv::Point2i(x2, y2)), color, 2);
        string label = cls_name + " " + confidence;
        cv::putText(img, label, cv::Point2i(x1 + 2, std::max(15, y1 - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
}
