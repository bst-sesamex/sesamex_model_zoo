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

size_t tensor_buffer_bytes(const bsnn_tensor_attr_t& attr) {
	const int d = attr.dims[0];
	const int w = attr.dims[1];
	const int h = attr.dims[2];
	const int pitch0 = attr.pitches[0];
	const int pitch1 = attr.pitches[1];
	const int dorder = attr.dorder;
	const int elem_bytes = elem_bytes_from_dtype(attr.dtype);

	if (dorder == BSNN_TENSOR_DWH) {
		const size_t stride_w = (size_t)std::max(pitch0, d * elem_bytes);
		const size_t stride_h =
			(size_t)std::max(1, (pitch1 > 0 ? pitch1 : w)) * stride_w;
		return (size_t)h * stride_h;
	}

	const size_t stride_h = (size_t)std::max(pitch0, w * elem_bytes);
	const size_t stride_d =
		(size_t)std::max(1, (pitch1 > 0 ? pitch1 : h)) * stride_h;
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

} // namespace

int main(int argc, char* argv[]) {
	std::string img1_path = "./input1.jpg";
	std::string img2_path = "./input2.jpg";
	std::string model_dir = "./model";
	std::string metric = "cosine";
	float threshold = 0.65f;
	bool normalize_feature = false;
	bool debug = false;
	const char* engine_cli = nullptr;

	for (int i = 1; i < argc; ++i) {
		if (bst_bsnn::parse_engine_cli(argc, argv, i, &engine_cli)) {
			continue;
		} else if (std::strcmp(argv[i], "-i1") == 0 && i + 1 < argc) {
			img1_path = argv[++i];
		} else if (std::strcmp(argv[i], "-i2") == 0 && i + 1 < argc) {
			img2_path = argv[++i];
		} else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
			model_dir = argv[++i];
		} else if (std::strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
			metric = argv[++i];
		} else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
			threshold = std::strtof(argv[++i], nullptr);
		} else if (std::strcmp(argv[i], "--normalize-feature") == 0) {
			normalize_feature = true;
		} else if (std::strcmp(argv[i], "--debug") == 0) {
			debug = true;
		}
	}

	if (metric != "cosine" && metric != "euclidean") {
		std::cerr << "Error: --metric must be cosine or euclidean" << std::endl;
		return -1;
	}

	std::cout << "-> Image1: " << img1_path << std::endl;
	std::cout << "-> Image2: " << img2_path << std::endl;
	std::cout << "-> Model : " << model_dir << std::endl;
	std::cout << "-> Metric: " << metric << std::endl;
	std::cout << "-> Threshold: " << threshold << std::endl;
	std::cout << "-> Normalize feature: " << (normalize_feature ? "ON" : "OFF") << std::endl;
	std::cout << "-> Debug: " << (debug ? "ON" : "OFF") << std::endl;

	std::string debug_dir;
	if (debug) {
		debug_dir = bst_path::join(model_dir, "debug_dump");
		if (!bst_path::ensure_dir(debug_dir)) {
			std::cerr << "Warning: failed to create debug dump dir: " << debug_dir << std::endl;
		}
	}

	const auto target_engine = bst_bsnn::resolve_target_engine(engine_cli);
	Timer total_timer;
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

	const int in_w = in_attr.dims[0];
	const int in_h = in_attr.dims[1];
	const int in_d = in_attr.dims[2];
	const int in_pitch0 = in_attr.pitches[0];
	const int in_pitch1 = in_attr.pitches[1];
	const int in_pitch2 = in_attr.pitches[2];
	const int in_dtype = in_attr.dtype;
	const int in_dorder = in_attr.dorder;

	if (bsnn_model.GetOutputAttrs().empty()) {
		std::cerr << "Error: failed to query output tensor attr" << std::endl;
		return -1;
	}
	const bsnn_tensor_attr_t& out_attr = bsnn_model.GetOutputAttrs()[0];

	std::cout << "-> Input tensor: W=" << in_w << ", H=" << in_h << ", D=" << in_d
			  << ", dtype=" << in_dtype << ", dOrder=" << in_dorder << ", pitch0="
			  << in_pitch0 << ", pitch1=" << in_pitch1 << std::endl;

	cv::Mat img1 = cv::imread(img1_path);
	cv::Mat img2 = cv::imread(img2_path);
	if (img1.empty() || img2.empty()) {
		std::cerr << "Error: failed to read input images" << std::endl;
		return -1;
	}

	std::vector<uint8_t> input_buf1;
	std::vector<uint8_t> input_buf2;

	Timer step_timer;
	reid_preprocess(img1,
					input_buf1,
					in_pitch0,
					in_w,
					in_h,
					in_d,
					in_dtype,
					in_dorder,
					in_pitch1,
					in_pitch2);
	std::cout << "-> Preprocess image1 time: " << step_timer.elapsed_ms() << " ms" << std::endl;

	step_timer.reset();
	reid_preprocess(img2,
					input_buf2,
					in_pitch0,
					in_w,
					in_h,
					in_d,
					in_dtype,
					in_dorder,
					in_pitch1,
					in_pitch2);
	std::cout << "-> Preprocess image2 time: " << step_timer.elapsed_ms() << " ms" << std::endl;

	if (debug) {
		const auto p1 = bst_path::join(debug_dir, "input_img1.bin");
		const auto p2 = bst_path::join(debug_dir, "input_img2.bin");
		if (dump_bin(p1, input_buf1.data(), input_buf1.size())) {
			std::cout << "-> Debug dump saved: " << p1 << " (" << input_buf1.size()
					  << " bytes)" << std::endl;
		} else {
			std::cerr << "Warning: failed to save " << p1 << std::endl;
		}
		if (dump_bin(p2, input_buf2.data(), input_buf2.size())) {
			std::cout << "-> Debug dump saved: " << p2 << " (" << input_buf2.size()
					  << " bytes)" << std::endl;
		} else {
			std::cerr << "Warning: failed to save " << p2 << std::endl;
		}
	}

	step_timer.reset();
	if (!bsnn_model.Run(input_buf1.data(), input_buf1.size())) {
		std::cerr << "Error: inference image1 failed" << std::endl;
		return -1;
	}
	bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());

	if (debug && bsnn_model.OutputVa(0)) {
		const size_t raw_bytes = tensor_buffer_bytes(out_attr);
		const auto raw1 = bst_path::join(debug_dir, "raw_output_img1.bin");
		const void* raw_ptr1 = bsnn_model.OutputVa(0);
		std::cout << "-> Debug raw1 info: D=" << out_attr.dims[0] << ", W=" << out_attr.dims[1]
				  << ", H=" << out_attr.dims[2] << ", dtype=" << out_attr.dtype
				  << ", dOrder=" << out_attr.dorder << ", pitch0=" << out_attr.pitches[0]
				  << ", pitch1=" << out_attr.pitches[1] << ", bytes=" << raw_bytes
				  << std::endl;
		if (dump_bin(raw1, raw_ptr1, raw_bytes)) {
			std::cout << "-> Debug dump saved: " << raw1 << " (" << raw_bytes << " bytes)"
					  << std::endl;
		} else {
			std::cerr << "Warning: failed to save " << raw1 << std::endl;
		}
	}

	ReidFeatureResult f1;
	step_timer.reset();
	if (!reid_decode_output_feature(bsnn_model, f1, 0, debug)) {
		std::cerr << "Error: decode output feature1 failed" << std::endl;
		bsnn_model.ReleaseOutputBuffer();
		return -1;
	}
	std::cout << "-> Postprocess image1 time: " << step_timer.elapsed_ms() << " ms" << std::endl;
	std::cout << "-> Output image1 dtype(meta): " << f1.output_dtype << " ("
			  << reid_dtype_to_string(f1.output_dtype) << "), deq_scale_num="
			  << f1.dequant_scale_num << ", deq_scale0=" << f1.dequant_scale
			  << std::endl;

	if (debug && !f1.embedding.empty()) {
		const auto emb1 = bst_path::join(debug_dir, "embedding_img1.bin");
		const size_t bytes = f1.embedding.size() * sizeof(float);
		if (dump_bin(emb1, f1.embedding.data(), bytes)) {
			std::cout << "-> Debug dump saved: " << emb1 << " (" << bytes << " bytes)"
					  << std::endl;
		} else {
			std::cerr << "Warning: failed to save " << emb1 << std::endl;
		}
	}

	if (!bsnn_model.ReleaseOutputBuffer()) {
		std::cerr << "Error: release output buffer1 failed" << std::endl;
		return -1;
	}

	step_timer.reset();
	if (!bsnn_model.Run(input_buf2.data(), input_buf2.size())) {
		std::cerr << "Error: inference image2 failed" << std::endl;
		return -1;
	}
	const long _infer_us2 = step_timer.elapsed_us();
	const double _infer_ms2 = static_cast<double>(_infer_us2) / 1000.0;
	std::cout << "-> Inference image2 time: " << std::fixed << std::setprecision(3)
			  << _infer_ms2 << " ms" << std::endl;

	if (debug && bsnn_model.OutputVa(0)) {
		const size_t raw_bytes = tensor_buffer_bytes(out_attr);
		const auto raw2 = bst_path::join(debug_dir, "raw_output_img2.bin");
		const void* raw_ptr2 = bsnn_model.OutputVa(0);
		std::cout << "-> Debug raw2 info: D=" << out_attr.dims[0] << ", W=" << out_attr.dims[1]
				  << ", H=" << out_attr.dims[2] << ", dtype=" << out_attr.dtype
				  << ", dOrder=" << out_attr.dorder << ", pitch0=" << out_attr.pitches[0]
				  << ", pitch1=" << out_attr.pitches[1] << ", bytes=" << raw_bytes
				  << std::endl;
		if (dump_bin(raw2, raw_ptr2, raw_bytes)) {
			std::cout << "-> Debug dump saved: " << raw2 << " (" << raw_bytes << " bytes)"
					  << std::endl;
		} else {
			std::cerr << "Warning: failed to save " << raw2 << std::endl;
		}
	}

	ReidFeatureResult f2;
	step_timer.reset();
	if (!reid_decode_output_feature(bsnn_model, f2, 0, debug)) {
		std::cerr << "Error: decode output feature2 failed" << std::endl;
		bsnn_model.ReleaseOutputBuffer();
		return -1;
	}
	std::cout << "-> Postprocess image2 time: " << step_timer.elapsed_ms() << " ms" << std::endl;
	std::cout << "-> Output image2 dtype(meta): " << f2.output_dtype << " ("
			  << reid_dtype_to_string(f2.output_dtype) << "), deq_scale_num="
			  << f2.dequant_scale_num << ", deq_scale0=" << f2.dequant_scale
			  << std::endl;

	if (debug && !f2.embedding.empty()) {
		const auto emb2 = bst_path::join(debug_dir, "embedding_img2.bin");
		const size_t bytes = f2.embedding.size() * sizeof(float);
		if (dump_bin(emb2, f2.embedding.data(), bytes)) {
			std::cout << "-> Debug dump saved: " << emb2 << " (" << bytes << " bytes)"
					  << std::endl;
		} else {
			std::cerr << "Warning: failed to save " << emb2 << std::endl;
		}
	}

	if (!bsnn_model.ReleaseOutputBuffer()) {
		std::cerr << "Error: release output buffer2 failed" << std::endl;
		return -1;
	}

	step_timer.reset();
	const float dist =
		reid_compute_distance(f1.embedding, f2.embedding, metric, normalize_feature);
	const bool is_same = dist <= threshold;
	std::cout << "-> Distance compute time: " << step_timer.elapsed_us() / 1000.0 << " ms"
			  << std::endl;

	std::cout << "========================================" << std::endl;
	std::cout << "  ReID Pair Match Summary" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "  feature_dim img1 : " << f1.embedding.size() << std::endl;
	std::cout << "  feature_dim img2 : " << f2.embedding.size() << std::endl;
	std::cout << "  metric           : " << metric << "_distance" << std::endl;
	std::cout << "  score(dist)      : " << dist << std::endl;
	std::cout << "  threshold        : " << threshold << std::endl;
	std::cout << "  match            : " << (is_same ? "True" : "False") << std::endl;
	std::cout << "  total time       : " << total_timer.elapsed_ms() << " ms" << std::endl;
	std::cout << "========================================" << std::endl;

	// 对齐 python：最终给出 True/False
	std::cout << (is_same ? "True" : "False") << std::endl;
	return 0;
}


