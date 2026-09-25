#ifndef __UNET_IMAGE_PROCESS_H__
#define __UNET_IMAGE_PROCESS_H__

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bsnn.h"
#include "bsnn_model_load.h"

class Timer {
public:
	Timer() : begin_(std::chrono::high_resolution_clock::now()) {}

	void reset() { begin_ = std::chrono::high_resolution_clock::now(); }

	int64_t elapsed_ms() const {
		return std::chrono::duration_cast<std::chrono::milliseconds>(
				   std::chrono::high_resolution_clock::now() - begin_)
			.count();
	}

	int64_t elapsed_us() const {
		return std::chrono::duration_cast<std::chrono::microseconds>(
				   std::chrono::high_resolution_clock::now() - begin_)
			.count();
	}

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> begin_;
};

struct UnetSegResult {
	cv::Mat class_mask;   // CV_8UC1, model output size
	cv::Mat color_mask;   // CV_8UC3, model output size
	int out_w = 0;
	int out_h = 0;
	int num_classes = 0;
	int out_dtype = -1;
	float dequant_scale = 1.0f;
	bool valid = false;
};

// 输入: 仅 resize，不做归一化，重点处理 dtype + pitch + dOrder
void unet_preprocess(const cv::Mat& src,
					 std::vector<uint8_t>& output_buf,
					 int pitch0,
					 int img_w,
					 int img_h,
					 int img_d,
					 int dtype,
					 int dorder,
					 int pitch1 = 0,
					 int pitch2 = 0);

// 输出: 支持 INT8/UINT8/INT16/UINT16/INT32/UINT32/FP32，反量化后 argmax
bool unet_decode_output(const BSNN_MODEL& model,
					UnetSegResult& result,
					int out_idx = 0,
					bool debug = false);

// Cityscapes 19类配色；超出部分使用灰色
void unet_apply_colormap(const cv::Mat& class_mask, cv::Mat& color_mask);

const char* unet_dtype_to_string(int dtype);
int unet_elem_bytes_from_dtype(int dtype);

#endif


