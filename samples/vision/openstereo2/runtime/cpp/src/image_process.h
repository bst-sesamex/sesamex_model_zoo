#ifndef __IMAGE_PROCESS_H__
#define __IMAGE_PROCESS_H__

#include <opencv2/opencv.hpp>
#include "zutils.h"
#include <thread>
#include "bsnn.h"
#include "bsnn_model_load.h"
#include "model_meta.h"

#define NEAR_FRAME          10  // Number of recent frames to save when using camera

// LightStereo预处理函数
void lightstereo_preprocess(const cv::Mat& left_img, const cv::Mat& right_img,
                           const ModelMeta meta,
                           uint8_t* left_buffer, uint8_t* right_buffer);

// LightStereo后处理函数
void lightstereo_process_output(const BSNN_MODEL& model,
                               cv::Mat& disparity_map);

// 视差图颜色映射
cv::Mat create_disparity_colormap(const cv::Mat& disparity, float max_disp = 192.0f);

#endif
