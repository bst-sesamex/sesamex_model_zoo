#ifndef __IMAGE_PROCESS_H__
#define __IMAGE_PROCESS_H__

#include <opencv2/opencv.hpp>
#include "zutils.h"
#include <thread>
#include "bsnn.h"
#include "bsnn_model_load.h"



#define  IMG_WIDTH          640
#define  IMG_HEIGHT         640

#define  REG_PITCH          256  //FOR Yolov8 64
// #define  REG_CHANNELS       64  //FOR Yolov8
// #define  CLS_PITCH          80  //FOR Yolov8
// #define  CLS_CHANNELS       80 //FOR Yolov8

#define  OUT_CHANNELS     255 //FOR Yolov5
#define  PER_ANCHORS_CHN    85
#define  NUM_CLASSES        80

#define CONF_THRESHOLD      0.25f
#define NMS_THRESHOLD       0.5f

#define HEAD_NUM  3 

void preprocess(const cv::Mat& src, cv::Mat& dst);

void yolov5_process_output(const BSNN_MODEL& model, std::vector<ObjInfo> &result);
void yolov8_process_output(const BSNN_MODEL& model, std::vector<ObjInfo> &result);

void draw_bboxes(cv::Mat& img, const std::vector<ObjInfo>& result);

#endif