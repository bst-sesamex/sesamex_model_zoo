#include "image_process.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

inline int elem_bytes_from_dtype(int dtype) {
	switch (dtype) {
	case BSNN_TENSOR_INT8:
	case BSNN_TENSOR_UINT8:
		return 1;
	case BSNN_TENSOR_INT16:
	case BSNN_TENSOR_UINT16:
		return 2;
	case BSNN_TENSOR_INT32:
	case BSNN_TENSOR_UINT32:
	case BSNN_TENSOR_FP32:
		return 4;
	default:
		return 1;
	}
}

inline float read_dequant_scale(const BSNN_MODEL& model, int out_idx) {
	if (out_idx >= (int)model.GetOutputAttrs().size()) {
		return 1.0f;
	}
	return BSNN_MODEL::QuantScale(model.GetOutputAttrs()[out_idx], 0);
}

inline float dequant_value_from_raw(uint8_t* base,
								int out_dtype,
								float scale) {
	if (out_dtype == BSNN_TENSOR_FP32) {
		return *reinterpret_cast<float*>(base);
	}
	if (out_dtype == BSNN_TENSOR_INT8) {
		return static_cast<float>(*reinterpret_cast<int8_t*>(base)) * scale;
	}
	if (out_dtype == BSNN_TENSOR_UINT8) {
		return static_cast<float>(*reinterpret_cast<uint8_t*>(base)) * scale;
	}
	if (out_dtype == BSNN_TENSOR_INT16) {
		return static_cast<float>(*reinterpret_cast<int16_t*>(base)) * scale;
	}
	if (out_dtype == BSNN_TENSOR_UINT16) {
		return static_cast<float>(*reinterpret_cast<uint16_t*>(base)) * scale;
	}
	if (out_dtype == BSNN_TENSOR_INT32) {
		return static_cast<float>(*reinterpret_cast<int32_t*>(base)) * scale;
	}
	if (out_dtype == BSNN_TENSOR_UINT32) {
		return static_cast<float>(*reinterpret_cast<uint32_t*>(base)) * scale;
	}
	return static_cast<float>(*reinterpret_cast<int8_t*>(base)) * scale;
}

} // namespace

void reid_preprocess(const cv::Mat& src,
					 std::vector<uint8_t>& output_buf,
					 int pitch0,
					 int img_w,
					 int img_h,
					 int img_d,
					 int dtype,
					 int dorder,
					 int pitch1,
					 int pitch2) {
	(void)pitch2;
	cv::Mat resized;
	cv::resize(src, resized, cv::Size(img_w, img_h), 0.0, 0.0, cv::INTER_LINEAR);

	const int elem_bytes = elem_bytes_from_dtype(dtype);

	// WHD layout strides (NCHW style): W fastest, then H, then D
	const int stride_h_whd = std::max(pitch0, img_w * elem_bytes);
	const int stride_d_whd =
		std::max(1, (pitch1 > 0 ? pitch1 : img_h)) * stride_h_whd;

	// DWH layout strides (NHWC style): D fastest, then W, then H
	const int stride_w_dwh = std::max(pitch0, img_d * elem_bytes);
	const int stride_h_dwh =
		std::max(1, (pitch1 > 0 ? pitch1 : img_w)) * stride_w_dwh;

	// 统一策略：不做归一化。
	// 1) INT8/UINT8: 保持 BGR uint8 写入
	// 2) FP32: 写入 BGR 原始像素值（float，范围约 0~255）
	if (dtype == BSNN_TENSOR_INT8 || dtype == BSNN_TENSOR_UINT8) {
		const size_t total_bytes = (dorder == BSNN_TENSOR_DWH)
									   ? (size_t)img_h * (size_t)stride_h_dwh
									   : (size_t)img_d * (size_t)stride_d_whd;
		output_buf.assign(total_bytes, 0);

		std::vector<cv::Mat> channels;
		cv::split(resized, channels); // B G R

		if (dorder == BSNN_TENSOR_DWH) {
			for (int h = 0; h < img_h; ++h) {
				for (int w = 0; w < img_w; ++w) {
					const size_t base =
						(size_t)h * (size_t)stride_h_dwh + (size_t)w * (size_t)stride_w_dwh;
					for (int d = 0; d < img_d; ++d) {
						output_buf[base + (size_t)d] = channels[d].at<uint8_t>(h, w);
					}
				}
			}
		} else {
			for (int d = 0; d < img_d; ++d) {
				const cv::Mat& ch = channels[d];
				for (int h = 0; h < img_h; ++h) {
					const uint8_t* row_ptr = ch.ptr<uint8_t>(h);
					const size_t offset =
						(size_t)d * (size_t)stride_d_whd + (size_t)h * (size_t)stride_h_whd;
					std::memcpy(output_buf.data() + offset, row_ptr, (size_t)img_w);
				}
			}
		}

		std::printf("-> Preprocess: UINT8/INT8(BGR), dOrder=%d, pitch0=%d, pitch1=%d, buf=%zu bytes\n",
					dorder,
					pitch0,
					pitch1,
					total_bytes);
		return;
	}

	cv::Mat bgr_f;
	resized.convertTo(bgr_f, CV_32FC3);

	std::vector<cv::Mat> channels;
	cv::split(bgr_f, channels); // B G R

	const size_t total_bytes = (dorder == BSNN_TENSOR_DWH)
								   ? (size_t)img_h * (size_t)stride_h_dwh
								   : (size_t)img_d * (size_t)stride_d_whd;
	output_buf.assign(total_bytes, 0);

	if (dorder == BSNN_TENSOR_DWH) {
		for (int h = 0; h < img_h; ++h) {
			for (int w = 0; w < img_w; ++w) {
				const size_t base =
					(size_t)h * (size_t)stride_h_dwh + (size_t)w * (size_t)stride_w_dwh;
				for (int d = 0; d < img_d; ++d) {
					const float v = channels[d].at<float>(h, w);
					std::memcpy(output_buf.data() + base + (size_t)d * sizeof(float),
								&v,
								sizeof(float));
				}
			}
		}
	} else {
		const int row_bytes = img_w * (int)sizeof(float);
		for (int d = 0; d < img_d; ++d) {
			const cv::Mat& ch = channels[d];
			for (int h = 0; h < img_h; ++h) {
				const float* row_ptr = ch.ptr<float>(h);
				const size_t offset =
					(size_t)d * (size_t)stride_d_whd + (size_t)h * (size_t)stride_h_whd;
				std::memcpy(output_buf.data() + offset, row_ptr, (size_t)row_bytes);
			}
		}
	}

	std::printf("-> Preprocess: FP32(BGR raw, no normalize), dOrder=%d, pitch0=%d, pitch1=%d, buf=%zu bytes\n",
				dorder,
				pitch0,
				pitch1,
				total_bytes);
}

bool reid_decode_output_feature(const BSNN_MODEL& model,
								ReidFeatureResult& result,
								int out_idx,
								bool debug) {
	result = ReidFeatureResult{};

	if (out_idx >= (int)model.GetOutputAttrs().size() || !model.OutputVa(out_idx)) {
		std::cerr << "reid_decode_output_feature: invalid input pointers" << std::endl;
		return false;
	}

	const bsnn_tensor_attr_t& out_attr = model.GetOutputAttrs()[out_idx];

	const int out_d = out_attr.dims[0];
	const int out_w = out_attr.dims[1];
	const int out_h = out_attr.dims[2];
	const int out_pitch0 = out_attr.pitches[0];
	const int out_pitch1 = out_attr.pitches[1];
	const int out_dtype = out_attr.dtype;
	const int out_dorder = out_attr.dorder;

	const int elem_bytes = elem_bytes_from_dtype(out_dtype);

	const size_t stride_h_whd = (size_t)std::max(out_pitch0, out_w * elem_bytes);
	const size_t stride_d_whd =
		(size_t)std::max(1, (out_pitch1 > 0 ? out_pitch1 : out_h)) * stride_h_whd;

	const size_t stride_w_dwh = (size_t)std::max(out_pitch0, out_d * elem_bytes);
	const size_t stride_h_dwh =
		(size_t)std::max(1, (out_pitch1 > 0 ? out_pitch1 : out_w)) * stride_w_dwh;

	uint8_t* raw_ptr = model.OutputVa(out_idx);
	if (!raw_ptr) {
		std::cerr << "Output buffer is null." << std::endl;
		return false;
	}

	const int total_values = out_d * out_w * out_h;
	std::vector<float> embedding;
	embedding.reserve((size_t)total_values);

	std::vector<float> deq_scales = BSNN_MODEL::QuantScales(out_attr);
	float dequant_scale = deq_scales.empty() ? 1.0f : deq_scales[0];
	uint32_t deq_num = static_cast<uint32_t>(deq_scales.size());

	for (int h = 0; h < out_h; ++h) {
		for (int w = 0; w < out_w; ++w) {
			for (int d = 0; d < out_d; ++d) {
				uint8_t* base = nullptr;
				if (out_dorder == BSNN_TENSOR_DWH) {
					base = raw_ptr + (size_t)h * stride_h_dwh + (size_t)w * stride_w_dwh +
						   (size_t)d * (size_t)elem_bytes;
				} else {
					base = raw_ptr + (size_t)d * stride_d_whd + (size_t)h * stride_h_whd +
						   (size_t)w * (size_t)elem_bytes;
				}

				float sc = BSNN_MODEL::ScaleAt(deq_scales, d);
				float v = dequant_value_from_raw(base, out_dtype, sc);
				embedding.push_back(v);
			}
		}
	}

	result.embedding = std::move(embedding);
	result.dequant_scale = dequant_scale;
	result.dequant_scales = std::move(deq_scales);
	result.dequant_scale_num = deq_num;
	result.output_dtype = out_dtype;
	result.valid = true;

	if (debug) {
		std::printf("-> Output tensor[%d]: D=%d, W=%d, H=%d, dOrder=%d, pitch0=%d, pitch1=%d, dtype=%d\n",
					out_idx,
					out_d,
					out_w,
					out_h,
					out_dorder,
					out_pitch0,
					out_pitch1,
					out_dtype);
		std::printf("-> Dequant scale: %.8f, scale_num: %u, feature dim: %zu\n",
					dequant_scale,
					result.dequant_scale_num,
					result.embedding.size());
		std::printf("-> First 10 feature values: ");
		for (size_t i = 0; i < std::min<size_t>(10, result.embedding.size()); ++i) {
			std::printf("%.6f ", result.embedding[i]);
		}
		std::printf("\n");
	}

	return true;
}

void reid_l2_normalize(std::vector<float>& v, float eps) {
	if (v.empty()) {
		return;
	}
	double sum_sq = 0.0;
	for (float x : v) {
		sum_sq += static_cast<double>(x) * static_cast<double>(x);
	}
	const float n = std::sqrt(static_cast<float>(sum_sq));
	const float denom = std::max(n, eps);
	for (float& x : v) {
		x /= denom;
	}
}

float reid_compute_distance(std::vector<float> f1,
							std::vector<float> f2,
							const std::string& metric,
							bool normalize_feature) {
	if (f1.size() != f2.size() || f1.empty()) {
		return std::numeric_limits<float>::infinity();
	}

	// 对齐 python 脚本：可选的预归一化
	if (normalize_feature) {
		reid_l2_normalize(f1);
		reid_l2_normalize(f2);
	}

	if (metric == "cosine") {
		// 对齐 torchreid.metrics.cosine_distance: 内部总是先归一化
		reid_l2_normalize(f1);
		reid_l2_normalize(f2);
		double dot = 0.0;
		for (size_t i = 0; i < f1.size(); ++i) {
			dot += static_cast<double>(f1[i]) * static_cast<double>(f2[i]);
		}
		return 1.0f - static_cast<float>(dot);
	}

	// euclidean: 对齐 torchreid.metrics.euclidean_squared_distance，返回平方欧氏距离
	double sum = 0.0;
	for (size_t i = 0; i < f1.size(); ++i) {
		const double diff = static_cast<double>(f1[i]) - static_cast<double>(f2[i]);
		sum += diff * diff;
	}
	return static_cast<float>(sum);
}

const char* reid_dtype_to_string(int dtype) {
	switch (dtype) {
	case BSNN_TENSOR_INT8:
		return "INT8";
	case BSNN_TENSOR_UINT8:
		return "UINT8";
	case BSNN_TENSOR_INT16:
		return "INT16";
	case BSNN_TENSOR_UINT16:
		return "UINT16";
	case BSNN_TENSOR_INT32:
		return "INT32";
	case BSNN_TENSOR_UINT32:
		return "UINT32";
	case BSNN_TENSOR_FP32:
		return "FP32";
	default:
		return "UNKNOWN";
	}
}


