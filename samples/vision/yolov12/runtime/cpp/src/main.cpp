#include <iostream>
#include <opencv2/opencv.hpp>
#include "bsnn_model_load.h"
//#include "drmshow.h"
#include "image_process.h"
#include "benchmark_timing.h"
#include <string.h>
#include <vector>


using namespace std;

bsnn_ret_t run_inference(cv::Mat& img, BSNN_MODEL& bsnn_model, Timer& time, Timer& fps, bool fused_output) {
    cv::Mat input_image;
    time.reset();
    fps.reset();
    int orig_w = img.cols;
    int orig_h = img.rows;
    preprocess(img, input_image);
    
    size_t input_img_len = 3 * IMG_HEIGHT * IMG_WIDTH;

    // 更改图片在内存中的存储方式为WHD
    if (!input_image.isContinuous())
        printf("-> input image is not continuous in memory...");
    std::vector<uchar> input_buf(input_img_len);
    for (int c = 0; c < 3; c++){
        for (int i = 0; i < IMG_HEIGHT; i++){
            for (int j = 0; j < IMG_WIDTH; j++){
                input_buf[c * IMG_HEIGHT * IMG_WIDTH + i * IMG_WIDTH + j] = input_image.data[3 * (i * IMG_WIDTH + j) + c];
            }
        }
    }
    cout << "-> preprocess time : " << time.elapsed() << " ms" << endl;
    // step 2 模型推理
    time.reset();
    if (!bsnn_model.Run(input_buf.data(), input_img_len)) {
        return BSNN_ERR_FAIL;
    }
    
    bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());
    
    // step 3 后处理
    time.reset();
    std::vector<ObjInfo> result;
    yolo_det_postprocess(bsnn_model, result, fused_output);
    cout << "-> The number of objects detected: " << result.size() << endl;
    // cout << "-> post process time : " << time.elapsed_micro() / 1000.0 << endl;
    // 修改为输出毫秒并保留一位小数
    printf("-> post process time : %.1f ms\n", time.elapsed_micro() / 1000.0);
    
    // plot box
    draw_bboxes(img, result);

    //add save
    // cv::imwrite("./inferresult.jpg",img);
    bsnn_model.ReleaseOutputBuffer();

    return BSNN_SUCC;
}

int main(int argc, char* argv[])
{
    std::string in_image_path = "./input.jpg";
    // std::string in_image_path = "./fad_video.avi";
    std::string bsnn_model_path = "./model";
    std::string out_image_path = "./inferresult.jpg";
    bool fused_output = true; // fused heads
    cv::Mat img;
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
        }
    }

    // 兼容旧的参数解析方式 (如果没有使用 -i/-m/-o 标识)
    if (argc > 1 && argv[1][0] != '-') {
        if(2 <= argc) {
            in_image_path = argv[1];
        }
        if(3 <= argc) {
            bsnn_model_path = argv[2];
        }
    }

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
        // step 1 图片预处理
        //capture >> img;
        img = cv::imread(in_image_path);
        if(img.empty())
        {
            printf("img.empty = %d\n",img.empty());
            ret = BSNN_ERR_INPUT_INVALID;
            break;
            //capture.release();
            //capture.open(in_image_path);
            // continue;
        }
        
        ret = run_inference(img, bsnn_model, time, fps, fused_output);
        if (ret != BSNN_SUCC) {
            break;
        }

        // Save result
        if (!out_image_path.empty()) {
            cv::imwrite(out_image_path, img);
            cout << "-> Saved result to: " << out_image_path << endl;
        }

        // printf("-> full flow FPS: %.2f\n", 1.0 / fps.elapsed() * 1000);
        //cv::Mat show_img;
        // cv::resize(img, show_img, cv::Size(1920, 1080));
        // cv::imshow("img", show_img);
        //cv::waitKey(1);
        executed = true;
    }
    
    if (ret != BSNN_SUCC) {
        cerr << "Error: Program execution failed with code " << ret << endl;
        return -1;
    }

    printf("SUCCESS\n");
    return 0;
}