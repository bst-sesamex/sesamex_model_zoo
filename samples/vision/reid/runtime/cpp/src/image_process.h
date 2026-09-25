#ifndef __REID_IMAGE_PROCESS_H__
#define __REID_IMAGE_PROCESS_H__

#include <chrono>
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

struct ReidFeatureResult {
	std::vector<float> embedding;
	float dequant_scale = 1.0f;
	std::vector<float> dequant_scales;
	uint32_t dequant_scale_num = 1;
	int output_dtype = -1;
	bool valid = false;
};

void reid_preprocess(const cv::Mat& src,
					 std::vector<uint8_t>& output_buf,
					 int pitch0,
					 int img_w,
					 int img_h,
					 int img_d,
					 int dtype,
					 int dorder,
					 int pitch1 = 0,
					 int pitch2 = 0);

bool reid_decode_output_feature(const BSNN_MODEL& model,
								ReidFeatureResult& result,
								int out_idx = 0,
								bool debug = false);

float reid_compute_distance(std::vector<float> f1,
							std::vector<float> f2,
							const std::string& metric,
							bool normalize_feature);

void reid_l2_normalize(std::vector<float>& v, float eps = 1e-12f);

const char* reid_dtype_to_string(int dtype);

#endif


