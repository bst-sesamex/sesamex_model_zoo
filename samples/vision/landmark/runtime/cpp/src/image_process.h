#ifndef __IMAGE_PROCESS_H__
#define __IMAGE_PROCESS_H__

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include "bsnn.h"
#include "bsnn_model_load.h"

#define LANDMARK_NUM_POINTS 68
#define LANDMARK_NUM_VALUES (LANDMARK_NUM_POINTS * 2)

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

struct LandmarkResult {
	std::vector<cv::Point2f> points_norm;   // [0,1] 坐标
	std::vector<cv::Point2f> points_pixel;  // 输入图像像素坐标
	std::vector<float> output_quant_flat;    // 解码后的量化值(按dOrder+pitch展开)
	std::vector<float> output_dequant_flat;  // 解码后的反量化值

	float min_norm = 0.0f;
	float max_norm = 0.0f;
	float dequant_scale = 1.0f;
	int output_dtype = -1;
	bool valid = false;
};

void landmark_preprocess(const cv::Mat& src,
						 std::vector<uint8_t>& output_buf,
						 int pitch,
						 int img_w,
						 int img_h,
						 int img_d,
						 int dtype,
						 int dorder,
						 int pitch1 = 0,
						 int pitch2 = 0);

void landmark_postprocess(const BSNN_MODEL& model,
						  int input_w,
						  int input_h,
						  LandmarkResult &result);

cv::Mat draw_landmarks(const cv::Mat& img,
					   const std::vector<cv::Point2f>& points,
					   const cv::Scalar& color = cv::Scalar(0, 255, 0),
					   int radius = 2,
					   bool draw_indices = false);

#endif


