#include <iostream>
#include <opencv2/opencv.hpp>
#include "bsnn_model_load.h"
#include "image_process.h"
#include "benchmark_timing.h"
#include <algorithm>
#include <cstring>
#include <vector>

using namespace std;

int main(int argc, char* argv[])
{
    std::string in_image_path = "./0_clip_resize.jpg";
    std::string bsnn_model_path = "condlanenet_model_v3/model";
    const char* engine_cli = nullptr;

    for (int i = 1; i < argc; ) {
        if (bst_bsnn::parse_engine_cli(argc, argv, i, &engine_cli)) {
            continue;
        }
        if (i + 1 < argc && strcmp(argv[i], "-i") == 0) {
            in_image_path = argv[++i];
            ++i;
            continue;
        }
        if (i + 1 < argc && strcmp(argv[i], "-m") == 0) {
            bsnn_model_path = argv[++i];
            ++i;
            continue;
        }
        ++i;
    }

    if (2 <= argc && argv[1][0] != '-') {
        in_image_path = argv[1];
    }
    if (3 <= argc && argv[2][0] != '-') {
        bsnn_model_path = argv[2];
    }

    const auto target_engine = bst_bsnn::resolve_target_engine(engine_cli);
    BSNN_MODEL bsnn_model(bsnn_model_path, 1, target_engine);
    if (!bsnn_model.model_init_succ_) {
        std::cerr << "Error: Model initialization failed" << std::endl;
        return -1;
    }

    Timer time, fps;
    LaneDetectionPostProcessor processor({0, 160, 1280, 720});

    cv::Mat img = cv::imread(in_image_path);
    if (img.empty()) {
        printf("img.empty = %d\n", img.empty());
        return -1;
    }
    time.reset();
    fps.reset();
    cv::Mat resized = crop_and_resize(img);
    if (resized.empty()) {
        std::cerr << "Error: crop_and_resize failed" << std::endl;
        return -1;
    }

    int pitch = IMG_WIDTH_PITCH;
    size_t input_img_len = 3 * IMG_HEIGHT * IMG_WIDTH_PITCH;
    if (!bsnn_model.GetInputAttrs().empty()) {
        const auto& in_attr = bsnn_model.GetInputAttrs()[0];
        pitch = std::max(static_cast<int>(in_attr.pitches[0]), IMG_WIDTH);
        if (in_attr.size > 0) {
            input_img_len = in_attr.size;
        }
    }
    std::vector<uchar> input_buf(input_img_len, 0);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < IMG_HEIGHT; i++) {
            for (int j = 0; j < IMG_WIDTH; j++) {
                input_buf[c * IMG_HEIGHT * pitch + i * pitch + j] =
                    resized.data[3 * (i * IMG_WIDTH + j) + c];
            }
        }
    }
    cout << "-> preprocess time : " << time.elapsed() << endl;

    time.reset();
    if (!bsnn_model.Run(input_buf.data(), input_buf.size())) {
        std::cerr << "Error: bsnn inference failed" << std::endl;
        return -1;
    }
    bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());

    std::cout << "Starting C++ Lane Detection Post-Processor..." << std::endl;
    if (!processor.load_outputs(bsnn_model, OUTPUT_TENSORS,
                                bsnn_model_path.find("a1000") != std::string::npos)) {
        std::cerr << "Failed to load model outputs" << std::endl;
        return -1;
    }

    auto lanes = processor.postprocess();
    std::cout << "Detected " << lanes.size() << " lanes" << std::endl;
    processor.visualize(in_image_path, "vis_cpp.jpg", lanes);
    bsnn_model.ReleaseOutputBuffer();
    printf("SUCCESS\n");
    return 0;
}
