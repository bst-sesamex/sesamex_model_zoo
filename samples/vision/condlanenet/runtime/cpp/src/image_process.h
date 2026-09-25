#ifndef __IMAGE_PROCESS_H__
#define __IMAGE_PROCESS_H__

#include <opencv2/opencv.hpp>
#include "zutils.h"
#include <thread>
#include "bsnn.h"
#include "bsnn_model_load.h"


#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <fstream>

#define  IMG_WIDTH_PITCH    832
#define  IMG_WIDTH          800
#define  IMG_HEIGHT         320
#define OUTPUT_TENSORS  5

struct TensorInfo {
    int pitch,expected_c, expected_h, expected_w;
    cv::Mat* target_mat;
    std::string name;
    bool is_uint8;  // true for uint8, false for int8
};

struct Seed {
    std::vector<int> coord;  // [x, y]
    int id_class;
    float score;
    cv::Mat reg;            // regression tensor
    cv::Mat mask;           // mask tensor
    cv::Mat range;          // range tensor
};

struct Lane {
    int id_class;
    std::vector<cv::Point2f> points;
    float score;
    Seed seed;
};

void preprocess(const cv::Mat& src, cv::Mat& dst);
cv::Mat crop_and_resize(const cv::Mat& src);

// 处理 std::vector<T>
template<typename T>
void save_to_bin(const std::vector<T>& data, 
                 const std::string& filename,
                 const std::vector<int>& shape = {},
                 bool save_shape = false);

// 处理 cv::Mat
void save_to_bin(const cv::Mat& mat, 
                 const std::string& filename,
                 const std::vector<int>& shape = {},
                 bool save_shape = false);

// 处理原始指针和计数
template<typename T>
void save_to_bin(T* data_ptr, 
                 size_t count,
                 const std::string& filename,
                 const std::vector<int>& shape = {},
                 bool save_shape = false);

cv::Mat loadTensorFromBin(const std::string& filename, 
                         int height, 
                         int width, 
                         int channels);

class CondLanePostProcessor {
public:
    CondLanePostProcessor(float hm_thr = 0.5f, int min_points = 5, float nms_thr = 3.0f, 
                         bool use_offset = true, float hm_downscale = 16.0f, 
                         float mask_downscale = 8.0f);
    
    std::vector<Lane> process(const std::vector<Seed>& output, float downscale);
    
private:
    float hm_thr_;
    int min_points_;
    float nms_thr_;
    bool use_offset_;
    float hm_downscale_;
    float mask_downscale_;
    
    std::vector<Seed> nms_seeds_tiny(const std::vector<Seed>& seeds);
    std::vector<Lane> process_lanes(const std::vector<Seed>& seeds, float downscale);
    std::vector<std::vector<int>> get_range(const std::vector<cv::Mat>& ranges);
    float cal_dis(const cv::Point2f& p1, const cv::Point2f& p2);
};

class LaneDetectionPostProcessor {
public:
    cv::Mat img_tensor_ ;
    LaneDetectionPostProcessor(const std::vector<int>& crop_bbox = {0, 160, 1280, 720});

    // Load ONNX output from binary files
    bool load_outputs(const BSNN_MODEL& model,
                         uint32_t tensors_per_batch = 1,
                         bool a1000_layout = false);
    bool load_reference_outputs();
    // Post-process the loaded output
    std::vector<std::vector<cv::Point2f>> postprocess();

    // Visualize results on original image
    void visualize(const std::string& original_image_path, 
                   const std::string& output_image_path,
                   const std::vector<std::vector<cv::Point2f>>& lanes);
    
private:
    std::vector<int> crop_bbox_;
    cv::Mat  hm_, hm_maxpool_, regs_, masks_, feat_range_;

    std::vector<Seed> ctdet_decode();
    std::vector<int> parse_pos(const std::vector<Seed>& seeds, int h, int w);
    std::vector<std::vector<cv::Point2f>> adjust_result(const std::vector<Lane>& lanes);
    
    // Helper to load tensor from bin file
    bool load_tensor_from_bin(const std::string& filename, cv::Mat& tensor, std::vector<int>& shape);
};


#endif