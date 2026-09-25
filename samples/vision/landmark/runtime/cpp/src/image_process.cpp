#include "bsnn_model_load.h"
#include "image_process.h"
#include <iostream>
#include <cstring>

using namespace std;

namespace {

inline int elem_bytes_from_dtype(int dtype) {
	if (dtype == BSNN_TENSOR_FP32) return 4;
	return 1;
}

inline float read_dequant_scale(const BSNN_MODEL& model, int out_idx) {
	if (out_idx >= (int)model.GetOutputAttrs().size()) return 1.0f;
	return BSNN_MODEL::QuantScale(model.GetOutputAttrs()[out_idx], 0);
}

}  // namespace

void landmark_preprocess(const cv::Mat& src,
						 std::vector<uint8_t>& output_buf,
						 int pitch,
						 int img_w,
						 int img_h,
						 int img_d,
						 int dtype,
						 int dorder,
						 int pitch1,
						 int pitch2)
{
	(void)pitch2;
	// cv::Mat resized;
	// cv::resize(src, resized, cv::Size(img_w, img_h), 0.0, 0.0, cv::INTER_LINEAR);

	const int elem_bytes = elem_bytes_from_dtype(dtype);
	const int pitch0_bytes = std::max(pitch, img_w * elem_bytes);
	const int stride_h_whd = pitch0_bytes;
	const int stride_d_whd = std::max(1, (pitch1 > 0 ? pitch1 : img_h)) * stride_h_whd;

	const int pitch_d_bytes = std::max(pitch, img_d * elem_bytes);
	const int stride_w_dwh = pitch_d_bytes;
	const int stride_h_dwh = std::max(1, (pitch1 > 0 ? pitch1 : img_w)) * stride_w_dwh;

	if (dtype == BSNN_TENSOR_UINT8 || dtype == BSNN_TENSOR_INT8) {
		// 量化输入路径：保持 BGR，按 dOrder + pitch 写入
		size_t total_bytes = (dorder == BSNN_TENSOR_DWH)
			? (size_t)img_h * stride_h_dwh
			: (size_t)img_d * stride_d_whd;
		output_buf.assign(total_bytes, 0);

		std::vector<cv::Mat> channels;
		cv::split(src, channels);  // B,G,R

		if (dorder == BSNN_TENSOR_DWH) {
			// NHWC: D fastest
			for (int h = 0; h < img_h; ++h) {
				for (int w = 0; w < img_w; ++w) {
					size_t base = (size_t)h * stride_h_dwh + (size_t)w * stride_w_dwh;
					for (int d = 0; d < img_d; ++d) {
						output_buf[base + d] = channels[d].at<uint8_t>(h, w);
					}
				}
			}
		} else {
			// NCHW: W fastest
			for (int d = 0; d < img_d; d++) {
				const cv::Mat& ch = channels[d];
				for (int h = 0; h < img_h; h++) {
					const uint8_t* row_ptr = ch.ptr<uint8_t>(h);
					size_t offset = (size_t)d * stride_d_whd + (size_t)h * stride_h_whd;
					std::memcpy(output_buf.data() + offset, row_ptr, img_w);
				}
			}
		}

		printf("-> Preprocess: UINT8/INT8(BGR), dOrder=%d, pitch0=%d, pitch1=%d, buf=%zu bytes\n",
			   dorder, pitch, pitch1, total_bytes);
		return;
	}

	cv::Mat resized;
	cv::resize(src, resized, cv::Size(img_w, img_h), 0.0, 0.0, cv::INTER_LINEAR);
	// 浮点输入路径：不做归一化，直接写入 BGR 原始像素值（float，范围约 0~255）
	cv::Mat bgr_f;
	resized.convertTo(bgr_f, CV_32FC3);

	std::vector<cv::Mat> channels;
	cv::split(bgr_f, channels); // B,G,R

	size_t total_bytes = (dorder == BSNN_TENSOR_DWH)
		? (size_t)img_h * stride_h_dwh
		: (size_t)img_d * stride_d_whd;
	output_buf.assign(total_bytes, 0);

	if (dorder == BSNN_TENSOR_DWH) {
		for (int h = 0; h < img_h; ++h) {
			for (int w = 0; w < img_w; ++w) {
				size_t base = (size_t)h * stride_h_dwh + (size_t)w * stride_w_dwh;
				for (int d = 0; d < img_d; ++d) {
					float v = channels[d].at<float>(h, w);
					std::memcpy(output_buf.data() + base + (size_t)d * sizeof(float), &v, sizeof(float));
				}
			}
		}
	} else {
		int w_bytes = img_w * (int)sizeof(float);
		for (int d = 0; d < img_d; d++) {
			const cv::Mat& ch = channels[d];
			for (int h = 0; h < img_h; h++) {
				const float* row_ptr = ch.ptr<float>(h);
				size_t offset = (size_t)d * stride_d_whd + (size_t)h * stride_h_whd;
				std::memcpy(output_buf.data() + offset, row_ptr, w_bytes);
			}
		}
	}

	printf("-> Preprocess: FP32(BGR raw, no normalize), dOrder=%d, pitch0=%d, pitch1=%d, buf=%zu bytes\n",
		   dorder, pitch, pitch1, total_bytes);
}

void landmark_postprocess(const BSNN_MODEL& model,
						  int input_w,
						  int input_h,
						  LandmarkResult &result)
{
	result = LandmarkResult{};

	if (model.GetOutputAttrs().empty() || model.GetOutputs().empty()) {
		std::cerr << "landmark_postprocess: invalid input pointers" << std::endl;
		return;
	}

	const bsnn_tensor_attr_t& out_attr = model.GetOutputAttrs()[0];

	const int out_d = out_attr.dims[0];
	const int out_w = out_attr.dims[1];
	const int out_h = out_attr.dims[2];
	const int out_pitch = out_attr.pitches[0];
	const int out_pitch1 = out_attr.pitches[1];
	const int out_dtype = out_attr.dtype;
	const int out_dorder = out_attr.dorder;

	const int elem_bytes = elem_bytes_from_dtype(out_dtype);
	const int pitch0_bytes = std::max(out_pitch, out_d * elem_bytes);
	const int pitch1_units = std::max(1, out_pitch1 > 0 ? out_pitch1 : out_w);
	const size_t stride_w_dwh = pitch0_bytes;
	const size_t stride_h_dwh = (size_t)pitch1_units * stride_w_dwh;

	const size_t stride_h_whd = (size_t)pitch0_bytes;
	const size_t stride_d_whd = (size_t)pitch1_units * stride_h_whd;

	const int total_values = out_d * out_w * out_h;
	if (total_values < LANDMARK_NUM_VALUES) {
		std::cerr << "Output values are insufficient for 68 landmarks: total="
				  << total_values << ", required=" << LANDMARK_NUM_VALUES << std::endl;
		return;
	}

	printf("-> Output tensor[0]: D=%d, W=%d, H=%d, dOrder=%d, pitch0=%d, pitch1=%d, dtype=%d\n",
		   out_d, out_w, out_h, out_dorder, out_pitch, out_pitch1, out_dtype);

	uint8_t *raw_ptr = model.OutputVa(0);
	if (!raw_ptr) {
		std::cerr << "Output buffer is null." << std::endl;
		return;
	}

	// 将 DWH (D 连续, W/H 通过 pitch 跨步) 展平到 float
	std::vector<float> out_quant_flat;
	std::vector<float> out_flat;
	out_quant_flat.reserve(total_values);
	out_flat.reserve(total_values);

	const float scale = read_dequant_scale(model, 0);
	result.dequant_scale = scale;
	result.output_dtype = out_dtype;

	for (int h = 0; h < out_h; ++h) {
		for (int w = 0; w < out_w; ++w) {
			for (int d = 0; d < out_d; ++d) {
				uint8_t* base = nullptr;
				if (out_dorder == BSNN_TENSOR_DWH) {
					base = raw_ptr + (size_t)h * stride_h_dwh + (size_t)w * stride_w_dwh + (size_t)d * elem_bytes;
				} else {
					base = raw_ptr + (size_t)d * stride_d_whd + (size_t)h * stride_h_whd + (size_t)w * elem_bytes;
				}

				float v = 0.0f;
				if (out_dtype == BSNN_TENSOR_INT8) {
					int8_t q = *reinterpret_cast<int8_t*>(base);
					out_quant_flat.push_back(static_cast<float>(q));
					v = static_cast<float>(q) * scale;
				} else if (out_dtype == BSNN_TENSOR_UINT8) {
					uint8_t q = *reinterpret_cast<uint8_t*>(base);
					out_quant_flat.push_back(static_cast<float>(q));
					v = static_cast<float>(q) * scale;
				} else if (out_dtype == BSNN_TENSOR_FP32) {
					out_quant_flat.push_back(*reinterpret_cast<float*>(base));
					v = *reinterpret_cast<float*>(base);
				} else {
					// 兜底：按 int8 处理
					int8_t q = *reinterpret_cast<int8_t*>(base);
					out_quant_flat.push_back(static_cast<float>(q));
					v = static_cast<float>(q) * scale;
				}
				out_flat.push_back(v);
			}
		}
	}

	result.output_quant_flat = std::move(out_quant_flat);
	result.output_dequant_flat = out_flat;

	// 解码前 136 个值: [x0,y0,x1,y1,...]，默认是 [0,1] 归一化坐标
	result.points_norm.resize(LANDMARK_NUM_POINTS);
	result.points_pixel.resize(LANDMARK_NUM_POINTS);

	float min_v = out_flat[0];
	float max_v = out_flat[0];
	for (int i = 0; i < LANDMARK_NUM_VALUES; ++i) {
		min_v = std::min(min_v, out_flat[i]);
		max_v = std::max(max_v, out_flat[i]);
	}
	result.min_norm = min_v;
	result.max_norm = max_v;

	for (int i = 0; i < LANDMARK_NUM_POINTS; ++i) {
		float x_norm = out_flat[i * 2 + 0];
		float y_norm = out_flat[i * 2 + 1];

		// 对齐 vis_onnx.py: pred_crop = landmarks_norm * input_size（不做clip）
		result.points_norm[i] = cv::Point2f(x_norm, y_norm);
		result.points_pixel[i] = cv::Point2f(x_norm * input_w, y_norm * input_h);
	}

	result.valid = true;

	printf("-> Landmark output decode done. values=%d, per-tensor scale=%.8f\n",
		   total_values, scale);
	printf("-> Landmark norm range (first %d vals): [%.6f, %.6f]\n",
		   LANDMARK_NUM_VALUES, min_v, max_v);

	printf("-> First 10 dequant values: ");
	for (int i = 0; i < std::min(10, LANDMARK_NUM_VALUES); ++i) {
		printf("%.5f ", out_flat[i]);
	}
	printf("\n");

	printf("-> First 5 points (norm -> pixel):\n");
	for (int i = 0; i < std::min(5, LANDMARK_NUM_POINTS); ++i) {
		printf("   [%d] (%.5f, %.5f) -> (%.2f, %.2f)\n",
			   i,
			   result.points_norm[i].x,
			   result.points_norm[i].y,
			   result.points_pixel[i].x,
			   result.points_pixel[i].y);
	}
}

cv::Mat draw_landmarks(const cv::Mat& img,
					   const std::vector<cv::Point2f>& points,
					   const cv::Scalar& color,
					   int radius,
					   bool draw_indices)
{
	cv::Mat out = img.clone();

	for (size_t i = 0; i < points.size(); ++i) {
		int x = (int)std::round(points[i].x);
		int y = (int)std::round(points[i].y);
		cv::circle(out, cv::Point(x, y), radius, color, -1);
		if (draw_indices) {
			cv::putText(out,
						std::to_string(i),
						cv::Point(x + 2, y - 2),
						cv::FONT_HERSHEY_SIMPLEX,
						0.3,
						color,
						1,
						cv::LINE_AA);
		}
	}

	return out;
}


