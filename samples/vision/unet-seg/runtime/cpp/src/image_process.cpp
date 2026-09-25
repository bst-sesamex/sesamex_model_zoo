#include "image_process.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

inline float read_dequant_scale(const BSNN_MODEL& model, int out_idx) {
	if (out_idx >= (int)model.GetOutputAttrs().size()) {
		return 1.0f;
	}
	return BSNN_MODEL::QuantScale(model.GetOutputAttrs()[out_idx], 0);
}

inline float read_value_dequant(uint8_t* base, int dtype, float scale) {
	switch (dtype) {
	case BSNN_TENSOR_INT8:
		return static_cast<float>(*reinterpret_cast<int8_t*>(base)) * scale;
	case BSNN_TENSOR_UINT8:
		return static_cast<float>(*reinterpret_cast<uint8_t*>(base)) * scale;
	case BSNN_TENSOR_INT16:
		return static_cast<float>(*reinterpret_cast<int16_t*>(base)) * scale;
	case BSNN_TENSOR_UINT16:
		return static_cast<float>(*reinterpret_cast<uint16_t*>(base)) * scale;
	case BSNN_TENSOR_INT32:
		return static_cast<float>(*reinterpret_cast<int32_t*>(base)) * scale;
	case BSNN_TENSOR_UINT32:
		return static_cast<float>(*reinterpret_cast<uint32_t*>(base)) * scale;
	case BSNN_TENSOR_FP32:
		return *reinterpret_cast<float*>(base);
	default:
		return static_cast<float>(*reinterpret_cast<int8_t*>(base)) * scale;
	}
}

} // namespace

int unet_elem_bytes_from_dtype(int dtype) {
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

void unet_preprocess(const cv::Mat& src,
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

	const int elem_bytes = unet_elem_bytes_from_dtype(dtype);

	// WHD: W fastest, then H, then D
	const int stride_h_whd = std::max(pitch0, img_w * elem_bytes);
	const int stride_d_whd = std::max(1, (pitch1 > 0 ? pitch1 : img_h)) * stride_h_whd;

	// DWH: D fastest, then W, then H
	const int stride_w_dwh = std::max(pitch0, img_d * elem_bytes);
	const int stride_h_dwh = std::max(1, (pitch1 > 0 ? pitch1 : img_w)) * stride_w_dwh;

	if (dtype == BSNN_TENSOR_UINT8 || dtype == BSNN_TENSOR_INT8) {
		const size_t total_bytes =
			(dorder == BSNN_TENSOR_DWH) ? (size_t)img_h * (size_t)stride_h_dwh
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
						const uint8_t v = channels[d].at<uint8_t>(h, w);
						if (dtype == BSNN_TENSOR_UINT8) {
							output_buf[base + (size_t)d] = v;
						} else {
							reinterpret_cast<int8_t*>(output_buf.data())[base + (size_t)d] =
								static_cast<int8_t>(v);
						}
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
					if (dtype == BSNN_TENSOR_UINT8) {
						std::memcpy(output_buf.data() + offset, row_ptr, (size_t)img_w);
					} else {
						int8_t* dst = reinterpret_cast<int8_t*>(output_buf.data() + offset);
						for (int x = 0; x < img_w; ++x) {
							dst[x] = static_cast<int8_t>(row_ptr[x]);
						}
					}
				}
			}
		}

		std::printf("-> Preprocess: %s(BGR raw,no normalize), dOrder=%d, pitch0=%d, pitch1=%d, buf=%zu bytes\n",
					dtype == BSNN_TENSOR_UINT8 ? "UINT8" : "INT8",
					dorder,
					pitch0,
					pitch1,
					total_bytes);
		return;
	}

	// FP32 等浮点路径：不归一化，直接写 0~255 的 BGR float
	cv::Mat bgr_f;
	resized.convertTo(bgr_f, CV_32FC3);

	std::vector<cv::Mat> channels;
	cv::split(bgr_f, channels);

	const size_t total_bytes =
		(dorder == BSNN_TENSOR_DWH) ? (size_t)img_h * (size_t)stride_h_dwh
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

	std::printf("-> Preprocess: FP32(BGR raw,no normalize), dOrder=%d, pitch0=%d, pitch1=%d, buf=%zu bytes\n",
				dorder,
				pitch0,
				pitch1,
				total_bytes);
}

bool unet_decode_output(const BSNN_MODEL& model,
					UnetSegResult& result,
					int out_idx,
					bool debug) {
	result = UnetSegResult{};

	if (out_idx >= (int)model.GetOutputAttrs().size() || !model.OutputVa(out_idx)) {
		std::cerr << "unet_decode_output: invalid input pointers" << std::endl;
		return false;
	}

	const bsnn_tensor_attr_t& out_attr = model.GetOutputAttrs()[out_idx];

	// 默认按 DWH: dims[0]=D(classes), dims[1]=W, dims[2]=H
	const int out_d = out_attr.dims[0];
	const int out_w = out_attr.dims[1];
	const int out_h = out_attr.dims[2];
	const int out_pitch0 = out_attr.pitches[0];
	const int out_pitch1 = out_attr.pitches[1];
	const int out_dtype = out_attr.dtype;
	const int out_dorder = out_attr.dorder;

	if (out_d <= 0 || out_w <= 0 || out_h <= 0) {
		std::cerr << "Invalid output shape: D=" << out_d << ", W=" << out_w << ", H=" << out_h
				  << std::endl;
		return false;
	}

	const int elem_bytes = unet_elem_bytes_from_dtype(out_dtype);

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

	const std::vector<float> deq_scales = BSNN_MODEL::QuantScales(out_attr);
	const float scale = deq_scales.empty() ? 1.0f : deq_scales[0];
	uint32_t deq_num = static_cast<uint32_t>(deq_scales.size());

	cv::Mat mask(out_h, out_w, CV_8UC1, cv::Scalar(0));

	for (int h = 0; h < out_h; ++h) {
		for (int w = 0; w < out_w; ++w) {
			float best_val = -std::numeric_limits<float>::infinity();
			int best_cls = 0;

			for (int d = 0; d < out_d; ++d) {
				uint8_t* base = nullptr;
				if (out_dorder == BSNN_TENSOR_DWH) {
					base = raw_ptr + (size_t)h * stride_h_dwh + (size_t)w * stride_w_dwh +
						   (size_t)d * (size_t)elem_bytes;
				} else {
					base = raw_ptr + (size_t)d * stride_d_whd + (size_t)h * stride_h_whd +
						   (size_t)w * (size_t)elem_bytes;
				}

				const float sc = deq_num > 1 ? deq_scales[(size_t)d % deq_scales.size()] : deq_scales[0];
				const float v = read_value_dequant(base, out_dtype, sc);
				if (v > best_val) {
					best_val = v;
					best_cls = d;
				}
			}

			mask.at<uint8_t>(h, w) = static_cast<uint8_t>(std::max(0, std::min(255, best_cls)));
		}
	}

	result.class_mask = mask;
	result.out_w = out_w;
	result.out_h = out_h;
	result.num_classes = out_d;
	result.out_dtype = out_dtype;
	result.dequant_scale = deq_scales.empty() ? scale : deq_scales[0];
	result.valid = true;

	unet_apply_colormap(result.class_mask, result.color_mask);

	if (debug) {
		std::printf("-> Output tensor[%d]: D=%d, W=%d, H=%d, dOrder=%d, pitch0=%d, pitch1=%d, dtype=%d(%s), scale_num=%u, scale0=%.8f\n",
					out_idx,
					out_d,
					out_w,
					out_h,
					out_dorder,
					out_pitch0,
					out_pitch1,
					out_dtype,
					unet_dtype_to_string(out_dtype),
					deq_num,
					result.dequant_scale);

		std::printf("-> First 8 dequant scales: ");
		for (size_t i = 0; i < std::min<size_t>(8, deq_scales.size()); ++i) {
			std::printf("%.8f ", deq_scales[i]);
		}
		std::printf("\n");

		int max_label = 0;
		for (int y = 0; y < result.class_mask.rows; ++y) {
			for (int x = 0; x < result.class_mask.cols; ++x) {
				max_label = std::max<int>(max_label, result.class_mask.at<uint8_t>(y, x));
			}
		}
		std::printf("-> Seg mask stats: shape=(%d,%d), max_label=%d\n",
					result.class_mask.cols,
					result.class_mask.rows,
					max_label);
	}

	return true;
}

void unet_apply_colormap(const cv::Mat& class_mask, cv::Mat& color_mask) {
	if (class_mask.empty()) {
		color_mask.release();
		return;
	}

	static const cv::Vec3b cityscapes_19[] = {
		{128, 64, 128}, {232, 35, 244}, {70, 70, 70},   {156, 102, 102}, {153, 153, 190},
		{153, 153, 153}, {30, 170, 250}, {0, 220, 220},  {35, 142, 107},  {152, 251, 152},
		{180, 130, 70},  {60, 20, 220},  {0, 0, 255},    {142, 0, 0},     {70, 0, 0},
		{100, 60, 0},    {100, 80, 0},   {230, 0, 0},    {32, 11, 119},
	};

	color_mask = cv::Mat::zeros(class_mask.rows, class_mask.cols, CV_8UC3);
	for (int y = 0; y < class_mask.rows; ++y) {
		for (int x = 0; x < class_mask.cols; ++x) {
			const uint8_t cls = class_mask.at<uint8_t>(y, x);
			if (cls < 19) {
				color_mask.at<cv::Vec3b>(y, x) = cityscapes_19[cls];
			} else {
				color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 128, 128);
			}
		}
	}
}

const char* unet_dtype_to_string(int dtype) {
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


