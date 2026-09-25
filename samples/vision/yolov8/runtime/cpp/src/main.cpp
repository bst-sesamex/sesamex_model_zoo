/*******************************************************************************
 * Copyright (c) 2026, Black Sesame Technologies Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include "bsnn_model_load.h"
//#include "drmshow.h"
#include "image_process.h"
#include "benchmark_timing.h"
#include <string.h>
#include <vector>


using namespace std;

bsnn_ret_t run_inference(cv::Mat& img, BSNN_MODEL& bsnn_model, Timer& time, Timer& fps, InferenceType infer_type, bool fused_output) {
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
    yolov8_postprocess(bsnn_model, result, fused_output, infer_type);
    // yolov8_process_output(bsnn_output, result);
    cout << "-> The number of objects detected: " << result.size() << endl;
    printf("-> post process time : %.1f ms\n", time.elapsed_micro() / 1000.0);

    // 打印检测结果
    for (size_t i = 0; i < result.size(); i++) {
        printf("   [%zu] class=%d  score=%.4f  bbox=(%.1f, %.1f, %.1f, %.1f)\n",
               i, result[i].classId, result[i].score,
               result[i].x1, result[i].y1, result[i].x2, result[i].y2);
        // pose: 打印关键点（模型空间坐标）
        if (infer_type == INFER_POSE && result[i].keypoints.size() >= 51) {
            printf("   keypoints (model space):\n");
            for (int k = 0; k < 17; k++) {
                printf("     kp%2d: x=%7.1f  y=%7.1f  conf=%.4f\n",
                       k, result[i].keypoints[k*3+0], result[i].keypoints[k*3+1], result[i].keypoints[k*3+2]);
            }
        }
    }


    if (infer_type == INFER_SEG) {
        cv::Mat proto;
        int proto_w = 0, proto_h = 0;
        if (build_proto(bsnn_model, 6, proto, proto_w, proto_h)) {
            draw_seg(img, result, proto, proto_w, proto_h, orig_w, orig_h);
        }
        draw_bboxes(img, result);
    } else if (infer_type == INFER_POSE) {
        draw_pose(img, result, orig_w, orig_h);
    } else {
        draw_bboxes(img, result);
    }

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
    std::string infer_flag = "det"; // det | seg | pose
    InferenceType infer_type = INFER_DET;
    bool fused_output = true; // fused heads (det/seg/pose)
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
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--type") == 0) && i + 1 < argc) {
            infer_flag = argv[++i];
        }
    }

    if (infer_flag == "seg" || infer_flag == "SEG") {
        infer_type = INFER_SEG;
    } else if (infer_flag == "pose" || infer_flag == "POSE") {
        infer_type = INFER_POSE;
    } else {
        infer_type = INFER_DET;
        infer_flag = "det";
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
        
        ret = run_inference(img, bsnn_model, time, fps, infer_type, fused_output);
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