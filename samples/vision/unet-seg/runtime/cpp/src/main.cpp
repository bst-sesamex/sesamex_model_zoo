#include <cstdlib>
#include <cstring>
#include "portable_path.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bsnn_model_load.h"
#include "image_process.h"
#include "benchmark_timing.h"

namespace {

size_t tensor_buffer_bytes_from_attr(int d, int w, int h, int dorder, int pitch0, int pitch1, int elem_bytes) {
	if (dorder == BSNN_TENSOR_DWH) {
		const size_t stride_w = (size_t)std::max(pitch0, d * elem_bytes);
		const size_t stride_h = (size_t)std::max(1, (pitch1 > 0 ? pitch1 : w)) * stride_w;
		return (size_t)h * stride_h;
	}
	const size_t stride_h = (size_t)std::max(pitch0, w * elem_bytes);
	const size_t stride_d = (size_t)std::max(1, (pitch1 > 0 ? pitch1 : h)) * stride_h;
	return (size_t)d * stride_d;
}

bool dump_bin(const std::string& path, const void* data, size_t bytes) {
	std::ofstream ofs(path, std::ios::binary);
	if (!ofs.is_open()) {
		return false;
	}
	ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
	return ofs.good();
}

void parse_input_shape_from_attr(const bsnn_tensor_attr_t& in_attr,
						 int& in_w,
						 int& in_h,
						 int& in_d) {
	// 优先兼容常见情况：WHD(size[2]=3) 与 DWH(size[0]=3)
	if (in_attr.dims[2] == 3 || in_attr.dims[2] == 1) {
		in_w = in_attr.dims[0];
		in_h = in_attr.dims[1];
		in_d = in_attr.dims[2];
		return;
	}
	if (in_attr.dims[0] == 3 || in_attr.dims[0] == 1) {
		in_d = in_attr.dims[0];
		in_w = in_attr.dims[1];
		in_h = in_attr.dims[2];
		return;
	}

	// fallback
	in_w = in_attr.dims[0];
	in_h = in_attr.dims[1];
	in_d = in_attr.dims[2];
}

} // namespace

int main(int argc, char* argv[]) {
	std::string image_path = "./input.jpg";
	std::string model_dir = "./model";
	std::string out_path = "./seg_overlay.jpg";
	float overlay_alpha = 0.5f;
	bool debug = false;
	const char* engine_cli = nullptr;

	for (int i = 1; i < argc; ++i) {
		if (bst_bsnn::parse_engine_cli(argc, argv, i, &engine_cli)) {
			continue;
		} else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
			image_path = argv[++i];
		} else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
			model_dir = argv[++i];
		} else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
			out_path = argv[++i];
		} else if (std::strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
			overlay_alpha = std::strtof(argv[++i], nullptr);
		} else if (std::strcmp(argv[i], "--debug") == 0) {
			debug = true;
		}
	}

	overlay_alpha = std::max(0.0f, std::min(1.0f, overlay_alpha));

	std::cout << "-> Input image: " << image_path << std::endl;
	std::cout << "-> Model dir  : " << model_dir << std::endl;
	std::cout << "-> Output img : " << out_path << std::endl;
	std::cout << "-> Overlay alpha: " << overlay_alpha << std::endl;
	std::cout << "-> Debug: " << (debug ? "ON" : "OFF") << std::endl;

	const auto target_engine = bst_bsnn::resolve_target_engine(engine_cli);
	BSNN_MODEL bsnn_model(model_dir, 1, target_engine);
	if (!bsnn_model.model_init_succ_) {
		std::cerr << "Error: model initialization failed" << std::endl;
		return -1;
	}

	if (bsnn_model.GetInputAttrs().empty()) {
		std::cerr << "Error: failed to query input tensor attr" << std::endl;
		return -1;
	}
	const bsnn_tensor_attr_t& in_attr = bsnn_model.GetInputAttrs()[0];

	int in_w = 0, in_h = 0, in_d = 0;
	parse_input_shape_from_attr(in_attr, in_w, in_h, in_d);
	const int in_pitch0 = in_attr.pitches[0];
	const int in_pitch1 = in_attr.pitches[1];
	const int in_pitch2 = in_attr.pitches[2];
	const int in_dtype = in_attr.dtype;
	const int in_dorder = in_attr.dorder;

	std::cout << "-> Input tensor: W=" << in_w << ", H=" << in_h << ", D=" << in_d
			  << ", dtype=" << in_dtype << "(" << unet_dtype_to_string(in_dtype) << ")"
			  << ", dOrder=" << in_dorder << ", pitch0=" << in_pitch0 << ", pitch1="
			  << in_pitch1 << std::endl;

	if (bsnn_model.GetOutputAttrs().empty()) {
		std::cerr << "Error: failed to query output tensor attr" << std::endl;
		return -1;
	}
	const bsnn_tensor_attr_t& out_attr = bsnn_model.GetOutputAttrs()[0];

	const int out_d = out_attr.dims[0];
	const int out_w = out_attr.dims[1];
	const int out_h = out_attr.dims[2];
	const int out_dtype = out_attr.dtype;
	const int out_pitch0 = out_attr.pitches[0];
	const int out_pitch1 = out_attr.pitches[1];
	const int out_dorder = out_attr.dorder;

	std::cout << "-> Output tensor: D=" << out_d << ", W=" << out_w << ", H=" << out_h
			  << ", dtype=" << out_dtype << "(" << unet_dtype_to_string(out_dtype) << ")"
			  << ", dOrder=" << out_dorder << ", pitch0=" << out_pitch0
			  << ", pitch1=" << out_pitch1 << std::endl;

	cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cerr << "Error: failed to read image: " << image_path << std::endl;
		return -1;
	}
	const int orig_w = image.cols;
	const int orig_h = image.rows;

	std::string debug_dir;
	if (debug) {
		const std::string name = bst_path::stem(out_path);
		const std::string stem = name.empty() ? "seg_overlay" : name;
		debug_dir = bst_path::join(bst_path::parent(out_path), stem + "_debug");
		if (!bst_path::ensure_dir(debug_dir)) {
			std::cerr << "Warning: create debug dir failed: " << debug_dir << std::endl;
		}
	}

	Timer timer_total;
	Timer timer_step;

	std::vector<uint8_t> input_buf;
	timer_step.reset();
	unet_preprocess(image,
				input_buf,
				in_pitch0,
				in_w,
				in_h,
				in_d,
				in_dtype,
				in_dorder,
				in_pitch1,
				in_pitch2);
	std::cout << "-> Preprocess time: " << timer_step.elapsed_ms() << " ms" << std::endl;

	if (debug) {
		const auto p_in = bst_path::join(debug_dir, "input_tensor.bin");
		if (dump_bin(p_in, input_buf.data(), input_buf.size())) {
			std::cout << "-> Debug dump saved: " << p_in << " (" << input_buf.size()
					  << " bytes)" << std::endl;
		}
	}

	timer_step.reset();
	if (!bsnn_model.Run(input_buf.data(), input_buf.size())) {
		std::cerr << "Error: inference failed" << std::endl;
		return -1;
	}
	bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());

	if (debug && bsnn_model.OutputVa(0)) {
		const int out_elem_bytes = unet_elem_bytes_from_dtype(out_dtype);
		const size_t out_raw_bytes =
			tensor_buffer_bytes_from_attr(out_d, out_w, out_h, out_dorder, out_pitch0, out_pitch1, out_elem_bytes);

		const void* out_raw_ptr = bsnn_model.OutputVa(0);
		const auto p_out = bst_path::join(debug_dir, "output_tensor_raw.bin");
		if (dump_bin(p_out, out_raw_ptr, out_raw_bytes)) {
			std::cout << "-> Debug dump saved: " << p_out << " (" << out_raw_bytes
					  << " bytes)" << std::endl;
		}
	}

	UnetSegResult seg_result;
	timer_step.reset();
	if (!unet_decode_output(bsnn_model, seg_result, 0, debug)) {
		std::cerr << "Error: decode output failed" << std::endl;
		bsnn_model.ReleaseOutputBuffer();
		return -1;
	}
	std::cout << "-> Postprocess time: " << timer_step.elapsed_ms() << " ms" << std::endl;

	if (!bsnn_model.ReleaseOutputBuffer()) {
		std::cerr << "Error: release output buffer failed" << std::endl;
		return -1;
	}

	cv::Mat color_resized;
	cv::resize(seg_result.color_mask, color_resized, cv::Size(orig_w, orig_h), 0.0, 0.0, cv::INTER_NEAREST);

	cv::Mat overlay;
	cv::addWeighted(image, 1.0f - overlay_alpha, color_resized, overlay_alpha, 0.0, overlay);

	if (!cv::imwrite(out_path, overlay)) {
		std::cerr << "Error: failed to save output image: " << out_path << std::endl;
		return -1;
	}

	if (debug) {
		const auto p_mask = bst_path::join(debug_dir, "post_mask_u8.bin");
		const auto p_overlay = bst_path::join(debug_dir, "overlay_debug.jpg");
		dump_bin(p_mask,
			 seg_result.class_mask.data,
			 (size_t)seg_result.class_mask.rows * (size_t)seg_result.class_mask.cols);
		cv::imwrite(p_overlay, overlay);
		std::cout << "-> Debug dump saved: " << p_mask << std::endl;
		std::cout << "-> Debug image saved: " << p_overlay << std::endl;
	}

	std::cout << "========================================" << std::endl;
	std::cout << "  UNet Segmentation Summary" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "  image size      : " << orig_w << "x" << orig_h << std::endl;
	std::cout << "  model input     : " << in_w << "x" << in_h << std::endl;
	std::cout << "  model output    : " << seg_result.out_w << "x" << seg_result.out_h
			  << "x" << seg_result.num_classes << std::endl;
	std::cout << "  output dtype    : " << seg_result.out_dtype << "("
			  << unet_dtype_to_string(seg_result.out_dtype) << ")" << std::endl;
	std::cout << "  dequant scale   : " << seg_result.dequant_scale << std::endl;
	std::cout << "  total time      : " << timer_total.elapsed_ms() << " ms" << std::endl;
	std::cout << "  output image    : " << out_path << std::endl;
	std::cout << "========================================" << std::endl;

	std::cout << "SUCCESS" << std::endl;
	return 0;
}

