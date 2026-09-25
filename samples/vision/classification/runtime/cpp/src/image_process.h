#ifndef __IMAGE_PROCESS_H__
#define __IMAGE_PROCESS_H__

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include "bsnn.h"

class BSNN_MODEL;

// ============================================================
// 分类模型参数 (与 resnet34.py 一致)
// ============================================================
#define IMG_SIZE            224

// CIFAR-10 归一化参数
static const float CIFAR10_MEAN[3] = {0.4914f, 0.4822f, 0.4465f};  // RGB
static const float CIFAR10_STD[3]  = {0.2023f, 0.1994f, 0.2010f};  // RGB

#define NUM_CLASSES         10

// CIFAR-10 类别名称
static const char* CIFAR10_CLASSES[NUM_CLASSES] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

// ============================================================
// Timer 工具
// ============================================================
class Timer {
public:
    Timer() : _begin(std::chrono::high_resolution_clock::now()) {}
    void reset() { _begin = std::chrono::high_resolution_clock::now(); }

    int64_t elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - _begin).count();
    }
    int64_t elapsed_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - _begin).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _begin;
};

// ============================================================
// 分类结果
// ============================================================
struct ClsResult {
    int   class_id;
    float score;        // softmax 概率
    std::string class_name;
};

// ============================================================
// 函数声明
// ============================================================

// 前处理:
//   1. resize 到 IMG_SIZE x IMG_SIZE
//   2. BGR 通道顺序
//   3. UINT8: 直接存像素; FP32: /255 + normalize
//   4. WHD (W 为最低维度) + pitch 补 0 对齐
void cls_preprocess(const cv::Mat& src,
                    std::vector<uint8_t>& output_buf,
                    int pitch, int img_w, int img_h, int img_d,
                    int dtype);

// 后处理:
//   反量化 INT8 输出 -> softmax -> argmax
void cls_postprocess(const BSNN_MODEL& model, ClsResult& result);

#endif
