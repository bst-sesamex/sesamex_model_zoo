#include <iostream>
#include <opencv2/opencv.hpp>
#include "bsnn_model_load.h"
#include "image_process.h"
#include "benchmark_timing.h"
#include <string.h>
#include <vector>

using namespace std;

// ============================================================
// FCOS 板端推理主流程
// 参考 YOLO 的 main.cpp 结构:
//   1. 图片预处理 (resize + letterbox, pad=114, uint8)
//   2. 转换为 CHW 布局送入模型
//   3. 模型推理
//   4. FCOS 后处理 (解码 15 个输出 tensor)
//   5. 画框并保存
// ============================================================

bsnn_ret_t run_inference(cv::Mat& img, BSNN_MODEL& bsnn_model, Timer& time, Timer& fps,
                         const std::string& out_image_path, float score_thr = 0.3f) {
    cv::Mat input_image;
    time.reset();
    fps.reset();

    int orig_w = img.cols;
    int orig_h = img.rows;

    // step 1: 前处理 (与YOLO完全一致)
    preprocess(img, input_image);

    size_t input_img_len = 3 * IMG_HEIGHT * IMG_WIDTH;

    // 更改图片在内存中的存储方式为 CHW (与YOLO一致)
    if (!input_image.isContinuous())
        printf("-> input image is not continuous in memory...\n");

    std::vector<uchar> input_buf(input_img_len);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < IMG_HEIGHT; i++) {
            for (int j = 0; j < IMG_WIDTH; j++) {
                input_buf[c * IMG_HEIGHT * IMG_WIDTH + i * IMG_WIDTH + j] =
                    input_image.data[3 * (i * IMG_WIDTH + j) + c];
            }
        }
    }
    cout << "-> preprocess time : " << time.elapsed() << " ms" << endl;

    // step 2: 模型推理
    time.reset();
    if (!bsnn_model.Run(input_buf.data(), input_img_len)) {
        return BSNN_ERR_FAIL;
    }

    bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());

    // step 3: FCOS 后处理
    time.reset();
    std::vector<ObjInfo> result;
    fcos_det_postprocess(bsnn_model, result, score_thr);
    cout << "-> The number of objects detected: " << result.size() << endl;
    printf("-> post process time : %.1f ms\n", time.elapsed_micro() / 1000.0);

    // step 4: 画框 (需要 letterbox 信息来还原坐标)
    LetterBoxInfo lb_info;
    lb_info.orig_w  = orig_w;
    lb_info.orig_h  = orig_h;
    lb_info.input_w  = IMG_WIDTH;
    lb_info.input_h  = IMG_HEIGHT;
    lb_info.scale    = std::min((float)IMG_WIDTH / orig_w, (float)IMG_HEIGHT / orig_h);
    lb_info.pad_left = (IMG_WIDTH  - (int)std::round(orig_w * lb_info.scale)) / 2;
    lb_info.pad_top  = (IMG_HEIGHT - (int)std::round(orig_h * lb_info.scale)) / 2;

    draw_bboxes_fcos(img, result, lb_info);

    // 保存结果
    if (!out_image_path.empty()) {
        cv::imwrite(out_image_path, img);
        cout << "-> Saved result to: " << out_image_path << endl;
    }

    bsnn_model.ReleaseOutputBuffer();

    return BSNN_SUCC;
}

int main(int argc, char* argv[])
{
    std::string in_image_path = "./input.jpg";
    std::string bsnn_model_path = "./model";
    std::string out_image_path = "./inferresult.jpg";
    float score_thr = 0.3f;
    cv::Mat img;
    const char* engine_cli = nullptr;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (bst_bsnn::parse_engine_cli(argc, argv, i, &engine_cli)) {
            continue;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            in_image_path = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            bsnn_model_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_image_path = argv[++i];
        } else if (strcmp(argv[i], "--score-thr") == 0 && i + 1 < argc) {
            score_thr = std::stof(argv[++i]);
        }
    }

    // 兼容旧的参数解析方式
    if (argc > 1 && argv[1][0] != '-') {
        if (2 <= argc) {
            in_image_path = argv[1];
        }
        if (3 <= argc) {
            bsnn_model_path = argv[2];
        }
    }

    printf("-> Input:  %s\n", in_image_path.c_str());
    printf("-> Model:  %s\n", bsnn_model_path.c_str());
    printf("-> Output: %s\n", out_image_path.c_str());
    printf("-> Score threshold: %.2f\n", score_thr);

    const auto target_engine = bst_bsnn::resolve_target_engine(engine_cli);
    BSNN_MODEL bsnn_model(bsnn_model_path, 1, target_engine);
    if (!bsnn_model.model_init_succ_) {
        cerr << "Error: Model initialization failed" << endl;
        return -1;
    }

    Timer time, fps;
    bool executed = false;
    bsnn_ret_t ret = BSNN_SUCC;

    while (!executed)
    {
        img = cv::imread(in_image_path);
        if (img.empty())
        {
            printf("img.empty = %d\n", img.empty());
            ret = BSNN_ERR_INPUT_INVALID;
            break;
        }

        ret = run_inference(img, bsnn_model, time, fps, out_image_path, score_thr);
        if (ret != BSNN_SUCC) {
            break;
        }

        executed = true;
    }

    if (ret != BSNN_SUCC) {
        cerr << "Error: Program execution failed with code " << ret << endl;
        return -1;
    }

    printf("SUCCESS\n");
    return 0;
}

