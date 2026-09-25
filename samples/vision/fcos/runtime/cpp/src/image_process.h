#ifndef __IMAGE_PROCESS_H__
#define __IMAGE_PROCESS_H__

#include <opencv2/opencv.hpp>
#include "zutils.h"
#include "bsnn.h"
#include "bsnn_model_load.h"

// ============================================================
// FCOS 模型参数
// ============================================================
// 输入图像尺寸 (与 inference.py 的 input_size 一致，默认 896x896)
#define  IMG_WIDTH          896
#define  IMG_HEIGHT         896

// COCO 80类
#define  NUM_CLASSES        80

// FCOS 有 5 个 FPN level
#define  FCOS_LEVELS        5
// 每个level有 cls/box/centerness 三组输出，共 15 个输出tensor
#define  FCOS_OUTPUTS       15

// FPN strides
static const int FCOS_STRIDES[FCOS_LEVELS] = {8, 16, 32, 64, 128};

// 后处理参数
#define  PRE_NMS_THRESHOLD  0.05f
#define  NMS_THRESHOLD      0.6f
#define  CONF_THRESHOLD     0.05f
#define  POST_NMS_TOPK      100
#define  PRE_NMS_TOPK       1000

// ============================================================
// LetterBox信息, 用于坐标还原
// ============================================================
struct LetterBoxInfo {
    float scale;        // 缩放比例 min(tw/w0, th/h0)
    int   pad_left;     // 左侧padding
    int   pad_top;      // 上侧padding
    int   input_w;      // 模型输入宽
    int   input_h;      // 模型输入高
    int   orig_w;       // 原始图像宽
    int   orig_h;       // 原始图像高
};

// ============================================================
// 函数声明
// ============================================================

// 前处理: 与 YOLO 完全一致 (resize + letterbox, pad=114, uint8输出)
void preprocess(const cv::Mat& src, cv::Mat& dst);

// FCOS 后处理: 解码 15 个输出 tensor，生成检测框
//   输出 tensor 排列：0-4 = cls(80ch), 5-9 = box(4ch), 10-14 = centerness(1ch)
//   per-channel 量化反量化
//   DWH 内存布局 (D=channel 是最低维度)
void fcos_det_postprocess(const BSNN_MODEL& model,
                          std::vector<ObjInfo> &result,
                          float score_thr = 0.3f);

// 画框: 将检测结果还原到原图坐标并绘制
void draw_bboxes_fcos(cv::Mat& img, const std::vector<ObjInfo>& result,
                      const LetterBoxInfo& lb_info);

#endif

