#include <iostream>
#include <fstream>
#include "portable_path.h"
#include <cmath>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include "bsnn_model_load.h"
#include "image_process.h"
#include "benchmark_timing.h"

using namespace std;

int main(int argc, char* argv[])
{
	std::string in_image_path = "./input.jpg";
	std::string bsnn_model_path = "./model";
	std::string out_image_path = "./inferresult.jpg";
	int input_size = -1;  // <=0: 使用模型输入尺寸
	const int vis_size = 224;  // 统一可视化尺寸
	bool draw_indices = false;
	bool debug = false;
	const char* engine_cli = nullptr;

	for (int i = 1; i < argc; i++) {
		if (bst_bsnn::parse_engine_cli(argc, argv, i, &engine_cli)) {
			continue;
		} else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
			in_image_path = argv[++i];
		} else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
			bsnn_model_path = argv[++i];
		} else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
			out_image_path = argv[++i];
		} else if (strcmp(argv[i], "--input_size") == 0 && i + 1 < argc) {
			input_size = std::atoi(argv[++i]);
		} else if (strcmp(argv[i], "--indices") == 0) {
			draw_indices = true;
		} else if (strcmp(argv[i], "--debug") == 0) {
			debug = true;
		}
	}

	// 兼容旧参数
	if (argc > 1 && argv[1][0] != '-') {
		if (2 <= argc) in_image_path = argv[1];
		if (3 <= argc) bsnn_model_path = argv[2];
		if (4 <= argc) out_image_path = argv[3];
	}

	printf("-> Input image: %s\n", in_image_path.c_str());
	printf("-> Model dir:   %s\n", bsnn_model_path.c_str());
	printf("-> Input size arg: %d (<=0 means auto by model)\n", input_size);
	printf("-> Vis size: %d\n", vis_size);
	printf("-> Debug: %s\n", debug ? "ON" : "OFF");
	printf("-> Output img:  %s\n", out_image_path.c_str());

	const auto target_engine = bst_bsnn::resolve_target_engine(engine_cli);
	BSNN_MODEL bsnn_model(bsnn_model_path, 1, target_engine);
	if (!bsnn_model.model_init_succ_) {
		cerr << "Error: Model initialization failed" << endl;
		return -1;
	}

	if (bsnn_model.GetInputAttrs().empty()) {
		cerr << "Error: Failed to query input tensor attributes" << endl;
		return -1;
	}
	const bsnn_tensor_attr_t& in_attr = bsnn_model.GetInputAttrs()[0];

	int in_w = in_attr.dims[0];
	int in_h = in_attr.dims[1];
	int in_d = in_attr.dims[2];
	int in_pitch0 = in_attr.pitches[0];
	int in_pitch1 = in_attr.pitches[1];
	int in_dtype = in_attr.dtype;
	int in_dorder = in_attr.dorder;
	int model_w = in_w;
	int model_h = in_h;

	if (input_size > 0) {
		if (input_size != model_w || input_size != model_h) {
			cerr << "Error: --input_size(" << input_size
				 << ") does not match model input(" << model_w << "x" << model_h << ")" << endl;
			cerr << "Please keep --input_size consistent with compiled model input." << endl;
			return -1;
		}
		in_w = input_size;
		in_h = input_size;
	}

	printf("-> Input tensor: W=%d, H=%d, D=%d, pitch=%d, dtype=%d\n",
		   in_w, in_h, in_d, in_pitch0, in_dtype);
	printf("-> Model tensor: W=%d, H=%d\n", model_w, model_h);
	printf("-> Input dOrder=%d, pitch0=%d, pitch1=%d\n", in_dorder, in_pitch0, in_pitch1);

	int elem_bytes = (in_dtype == BSNN_TENSOR_FP32) ? 4 : 1;
	size_t input_buf_size = 0;
	if (in_dorder == BSNN_TENSOR_DWH) {
		int pitch_d_bytes = std::max(in_pitch0, in_d * elem_bytes);
		int pitch_w_units = std::max(1, in_pitch1 > 0 ? in_pitch1 : in_w);
		input_buf_size = (size_t)in_h * (size_t)pitch_w_units * (size_t)pitch_d_bytes;
	} else {
		int pitch_w_bytes = std::max(in_pitch0, in_w * elem_bytes);
		int pitch_h_units = std::max(1, in_pitch1 > 0 ? in_pitch1 : in_h);
		input_buf_size = (size_t)in_d * (size_t)pitch_h_units * (size_t)pitch_w_bytes;
	}

	Timer total_timer;

	cv::Mat img = cv::imread(in_image_path);
	if (img.empty()) {
		cerr << "Error: Failed to read image: " << in_image_path << endl;
		return -1;
	}
	printf("-> Image loaded: %d x %d\n", img.cols, img.rows);

	Timer step_timer;
	std::vector<uint8_t> input_buf;
	landmark_preprocess(img, input_buf, in_pitch0, in_w, in_h, in_d, in_dtype, in_dorder, in_pitch1, in_attr.pitches[2]);
	printf("-> Preprocess time: %ld ms\n", step_timer.elapsed_ms());

	std::string debug_dir;
	if (debug) {
		std::string stem = bst_path::stem(out_image_path);
		if (stem.empty()) stem = "inferresult";
		debug_dir = bst_path::join(bst_path::parent(out_image_path), stem + "_debug");
		if (!bst_path::ensure_dir(debug_dir)) {
			std::cerr << "Error creating debug directory: " << debug_dir << std::endl;
			return 1;
		}

		// cv::Mat input_resized;
		// cv::resize(img, input_resized, cv::Size(in_w, in_h), 0.0, 0.0, cv::INTER_LINEAR);
		// cv::imwrite(bst_path::join(debug_dir, "01_model_input_bgr.jpg"), input_resized);

		std::ofstream fin(bst_path::join(debug_dir, "02_model_input_tensor_u8_with_pitch.bin"), std::ios::binary);
		if (fin.is_open()) {
			fin.write(reinterpret_cast<const char*>(input_buf.data()), (std::streamsize)input_buf.size());
		}
	}

	if (input_buf.size() != input_buf_size) {
		printf("-> WARNING: input_buf size(%zu) != expected(%zu), adjusting...\n",
			   input_buf.size(), input_buf_size);
		input_buf.resize(input_buf_size, 0);
	}

	step_timer.reset();
	if (!bsnn_model.Run(input_buf.data(), input_buf_size)) {
		cerr << "Error: Model inference failed" << endl;
		return -1;
	}
	bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());

	if (debug) {
		const bsnn_tensor_attr_t& out_attr = bsnn_model.GetOutputAttrs()[0];
		{
			int out_d = out_attr.dims[0];
			int out_w = out_attr.dims[1];
			int out_h = out_attr.dims[2];
			int out_pitch0 = out_attr.pitches[0];
			int out_pitch1 = out_attr.pitches[1];
			int out_dtype = out_attr.dtype;
			int out_dorder = out_attr.dorder;

			int out_elem_bytes = (out_dtype == BSNN_TENSOR_FP32) ? 4 : 1;
			size_t out_bytes = 0;
			if (out_dorder == BSNN_TENSOR_DWH) {
				int pitch_d_bytes = std::max(out_pitch0, out_d * out_elem_bytes);
				int pitch_w_units = std::max(1, out_pitch1 > 0 ? out_pitch1 : out_w);
				out_bytes = (size_t)out_h * (size_t)pitch_w_units * (size_t)pitch_d_bytes;
			} else {
				int pitch_w_bytes = std::max(out_pitch0, out_w * out_elem_bytes);
				int pitch_h_units = std::max(1, out_pitch1 > 0 ? out_pitch1 : out_h);
				out_bytes = (size_t)out_d * (size_t)pitch_h_units * (size_t)pitch_w_bytes;
			}

			uint8_t* out_raw = bsnn_model.OutputVa(0);
			if (out_raw && out_bytes > 0) {
				std::ofstream fout(bst_path::join(debug_dir, "03_model_output_tensor0_raw_with_pitch.bin"), std::ios::binary);
				if (fout.is_open()) {
					fout.write(reinterpret_cast<const char*>(out_raw), (std::streamsize)out_bytes);
				}
			}

			printf("-> Debug model I/O dtype: input=%d (expect UINT8=3), output=%d (expect INT8=0)\n",
				   in_dtype,
				   out_dtype);
		}
	}

	step_timer.reset();
	LandmarkResult result;
	landmark_postprocess(bsnn_model, in_w, in_h, result);
	printf("-> Postprocess time: %.1f ms\n", step_timer.elapsed_us() / 1000.0);

	if (!result.valid) {
		cerr << "Error: Landmark postprocess failed" << endl;
		bsnn_model.ReleaseOutputBuffer();
		return -1;
	}

	if (debug) {
		// 后处理输出保存为bin，便于numpy直接读取
		if (!result.output_quant_flat.empty()) {
			std::vector<int8_t> post_quant_i8;
			post_quant_i8.reserve(result.output_quant_flat.size());
			for (float v : result.output_quant_flat) {
				int iv = (int)std::round(v);
				if (iv > 127) iv = 127;
				if (iv < -128) iv = -128;
				post_quant_i8.push_back((int8_t)iv);
			}
			std::ofstream fb(bst_path::join(debug_dir, "04_post_output_quant_i8.bin"), std::ios::binary);
			if (fb.is_open()) {
				fb.write(reinterpret_cast<const char*>(post_quant_i8.data()),
						 (std::streamsize)(post_quant_i8.size() * sizeof(int8_t)));
			}
		}

		if (!result.output_dequant_flat.empty()) {
			std::ofstream fb(bst_path::join(debug_dir, "05_post_output_dequant_f32.bin"), std::ios::binary);
			if (fb.is_open()) {
				fb.write(reinterpret_cast<const char*>(result.output_dequant_flat.data()),
						 (std::streamsize)(result.output_dequant_flat.size() * sizeof(float)));
			}
		}

		if (!result.points_norm.empty()) {
			std::vector<float> points_norm_xy;
			points_norm_xy.reserve(result.points_norm.size() * 2);
			for (const auto& p : result.points_norm) {
				points_norm_xy.push_back(p.x);
				points_norm_xy.push_back(p.y);
			}
			std::ofstream fb(bst_path::join(debug_dir, "06_post_points_norm_xy_f32.bin"), std::ios::binary);
			if (fb.is_open()) {
				fb.write(reinterpret_cast<const char*>(points_norm_xy.data()),
						 (std::streamsize)(points_norm_xy.size() * sizeof(float)));
			}
		}

		if (!result.points_pixel.empty()) {
			std::vector<float> points_pixel_xy;
			points_pixel_xy.reserve(result.points_pixel.size() * 2);
			for (const auto& p : result.points_pixel) {
				points_pixel_xy.push_back(p.x);
				points_pixel_xy.push_back(p.y);
			}
			std::ofstream fb(bst_path::join(debug_dir, "07_post_points_pixel_xy_f32.bin"), std::ios::binary);
			if (fb.is_open()) {
				fb.write(reinterpret_cast<const char*>(points_pixel_xy.data()),
						 (std::streamsize)(points_pixel_xy.size() * sizeof(float)));
			}
		}
	}

	// 对齐 vis_onnx.py: 先在 input_size 空间得到点，再统一缩放到 vis_size 可视化
	cv::Mat vis_input;
	cv::resize(img, vis_input, cv::Size(in_w, in_h), 0.0, 0.0, cv::INTER_LINEAR);
	cv::Mat vis;
	cv::resize(vis_input, vis, cv::Size(vis_size, vis_size), 0.0, 0.0, cv::INTER_CUBIC);

	std::vector<cv::Point2f> vis_points = result.points_pixel;
	float scale_x = (float)vis_size / (float)std::max(1, in_w);
	float scale_y = (float)vis_size / (float)std::max(1, in_h);
	for (auto& p : vis_points) {
		p.x *= scale_x;
		p.y *= scale_y;
	}

	int radius = std::max(2, (int)std::round((float)vis_size / 112.0f));
	vis = draw_landmarks(vis, vis_points, cv::Scalar(0, 255, 0), radius, draw_indices);
	cv::putText(vis,
				"Landmark 68pts (Green)",
				cv::Point(5, 16),
				cv::FONT_HERSHEY_SIMPLEX,
				0.45,
				cv::Scalar(255, 255, 255),
				1,
				cv::LINE_AA);

	if (!cv::imwrite(out_image_path, vis)) {
		cerr << "Error: Failed to save output image: " << out_image_path << endl;
		bsnn_model.ReleaseOutputBuffer();
		return -1;
	}

	printf("-> Saved image: %s\n", out_image_path.c_str());
	if (debug) {
		printf("-> Saved debug artifacts: %s\n", debug_dir.c_str());
	}

	printf("========================================\n");
	printf("  Landmark Result Summary\n");
	printf("========================================\n");
	printf("  Points Num : %zu\n", result.points_norm.size());
	printf("  Out DType  : %d\n", result.output_dtype);
	printf("  Deq Scale  : %.8f\n", result.dequant_scale);
	printf("  Norm Range : [%.6f, %.6f]\n", result.min_norm, result.max_norm);
	printf("  Total time : %ld ms\n", total_timer.elapsed_ms());
	printf("========================================\n");

	bsnn_model.ReleaseOutputBuffer();

	printf("SUCCESS\n");
	return 0;
}


