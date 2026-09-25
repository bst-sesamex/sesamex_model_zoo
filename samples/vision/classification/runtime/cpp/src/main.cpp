#include <iostream>
#include <opencv2/opencv.hpp>
#include "bsnn_model_load.h"
#include "image_process.h"
#include "benchmark_timing.h"
#include <string.h>
#include <vector>

using namespace std;

// ============================================================
// 分类模型板端推理主流程 (bsnnrt 0.13.0)
//   1. 图片预处理 (resize + normalize + WHD + pitch 对齐)
//   2. 模型推理
//   3. 后处理 (反量化 + softmax + argmax)
//   4. 输出分类结果
// ============================================================

int main(int argc, char* argv[])
{
    std::string in_image_path = "./input.jpg";
    std::string bsnn_model_path = "./model";
    const char* engine_cli = nullptr;

    for (int i = 1; i < argc; i++) {
        if (bst_bsnn::parse_engine_cli(argc, argv, i, &engine_cli)) {
            continue;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            in_image_path = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            bsnn_model_path = argv[++i];
        }
    }

    if (argc > 1 && argv[1][0] != '-') {
        if (2 <= argc) in_image_path = argv[1];
        if (3 <= argc) bsnn_model_path = argv[2];
    }

    printf("-> Input image: %s\n", in_image_path.c_str());
    printf("-> Model dir:   %s\n", bsnn_model_path.c_str());

    const int target_engine = bst_bsnn::resolve_target_engine(engine_cli);
    BSNN_MODEL bsnn_model(bsnn_model_path, 1, target_engine);
    if (!bsnn_model.model_init_succ_) {
        cerr << "Error: Model initialization failed" << endl;
        return -1;
    }

    if (bsnn_model.GetInputAttrs().empty()) {
        cerr << "Error: Failed to query input tensor attributes" << endl;
        return -1;
    }
    const bsnn_tensor_attr& in_attr = bsnn_model.GetInputAttrs()[0];

    // WHD 布局: dims[0]=W, dims[1]=H, dims[2]=D(channels)
    int in_w     = static_cast<int>(in_attr.dims[0]);
    int in_h     = static_cast<int>(in_attr.dims[1]);
    int in_d     = in_attr.dims_size > 2 ? static_cast<int>(in_attr.dims[2]) : 1;
    int in_pitch = static_cast<int>(in_attr.pitches[0]);
    int in_dtype = static_cast<int>(in_attr.dtype);
    if (in_pitch < in_w) {
        in_pitch = in_w;
    }

    printf("-> Input tensor: W=%d, H=%d, D=%d, pitch=%d, dtype=%d, size=%zu\n",
           in_w, in_h, in_d, in_pitch, in_dtype, in_attr.size);

    size_t input_buf_size = in_attr.size;
    if (input_buf_size == 0) {
        const size_t elem = (in_dtype == BSNN_TENSOR_FP32) ? sizeof(float) : 1;
        input_buf_size = (size_t)in_d * in_h * in_pitch * elem;
    }
    printf("-> Input buffer size: %zu bytes\n", input_buf_size);

    Timer total_timer;
    cv::Mat img = cv::imread(in_image_path);
    if (img.empty()) {
        cerr << "Error: Failed to read image: " << in_image_path << endl;
        return -1;
    }
    printf("-> Image loaded: %d x %d\n", img.cols, img.rows);

    Timer step_timer;
    std::vector<uint8_t> input_buf;
    cls_preprocess(img, input_buf, in_pitch, in_w, in_h, in_d, in_dtype);
    printf("-> Preprocess time: %ld ms\n", step_timer.elapsed_ms());

    if (input_buf.size() != input_buf_size) {
        printf("-> WARNING: input_buf size(%zu) != expected(%zu), adjusting...\n",
               input_buf.size(), input_buf_size);
        input_buf.resize(input_buf_size, 0);
    }

    step_timer.reset();
    if (!bsnn_model.Run(input_buf.data(), input_buf.size())) {
        cerr << "Error: Model inference failed" << endl;
        return -1;
    }
    bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());

    step_timer.reset();
    ClsResult result;
    cls_postprocess(bsnn_model, result);
    printf("-> Postprocess time: %.1f ms\n", step_timer.elapsed_us() / 1000.0);

    printf("\n");
    printf("========================================\n");
    printf("  Classification Result\n");
    printf("========================================\n");
    printf("  Class ID:   %d\n", result.class_id);
    printf("  Class Name: %s\n", result.class_name.c_str());
    printf("  Confidence: %.4f (%.2f%%)\n", result.score, result.score * 100.0f);
    printf("========================================\n");
    printf("  Total time: %ld ms\n", total_timer.elapsed_ms());
    printf("========================================\n");

    bsnn_model.ReleaseOutputBuffer();

    printf("SUCCESS\n");
    return 0;
}
