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
#include <memory>
#include "bsnn_model_load.h"
#include "image_process.h"
#include "benchmark_timing.h"
#include <string.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <cerrno>
#include <cctype>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;

bool file_exists(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0;
}

bool is_directory(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool is_regular_file(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

std::string get_file_extension(const std::string& filename) {
    const auto slash = filename.find_last_of('/');
    const auto dot = filename.find_last_of('.');
    return dot == std::string::npos || (slash != std::string::npos && dot < slash)
        ? std::string() : filename.substr(dot);
}

std::string get_filename_without_extension(const std::string& filepath) {
    const auto name = filepath.substr(filepath.find_last_of('/') == std::string::npos
        ? 0 : filepath.find_last_of('/') + 1);
    const auto dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

std::string get_filename(const std::string& filepath) {
    const auto slash = filepath.find_last_of('/');
    return filepath.substr(slash == std::string::npos ? 0 : slash + 1);
}

std::vector<std::string> list_directory_images(const std::string& dir_path) {
    std::vector<std::string> image_files;
    
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) {
        std::cerr << "Error reading directory " << dir_path << std::endl;
        return image_files;
    }
    while (const dirent* entry = readdir(dir)) {
        const std::string path = dir_path + "/" + entry->d_name;
        if (!is_regular_file(path)) continue;
        std::string ext = get_file_extension(path);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
            ext == ".bmp" || ext == ".tiff" || ext == ".tif" || ext == ".webp") {
            image_files.push_back(path);
        }
    }
    closedir(dir);
    
    return image_files;
}

bool create_directory_if_not_exists(const std::string& path) {
    if (path.empty()) return false;
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i != path.size() && path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (part.empty() || part == ".") continue;
        if (mkdir(part.c_str(), 0775) != 0 && errno != EEXIST) {
            std::cerr << "Error creating directory " << part << std::endl;
            return false;
        }
    }
    return is_directory(path);
}

// Structure to hold command line arguments
struct Config {
    std::string left_path = "./left";
    std::string right_path = "./right";
    std::string model_path = "./model";
    std::string output_path = "./output";
    int save_output = 1;
    int camera_id = -1;
    bool use_camera = false;
    bool show_help = false;
    std::string engine_cli;
    // Optional mask & gt
    std::string mask_path = ""; // supports .png/.jpg (grayscale) or .bin (uint8)
    std::string gt_path = "";   // supports .bin (float32)
};

// Function to print help information
void print_help() {
    cout << "Usage: program [options]\n"
         << "Options:\n"
         << "  -left <path>  Left image directory or file (default: ./left)\n"
         << "  -right <path> Right image directory or file (default: ./right)\n"
         << "  -m <path>     Model directory (default: ./model)\n"
         << "  -o <path>     Output directory (default: ./output)\n"
         << "  -save <0|1>   Save output images (default: 1, 0=no save, 1=save)\n"
         << "  -mask <path>  Optional mask file (.png/.jpg grayscale or .bin uint8 of H*W)\n"
         << "  -gt <path>    Optional ground-truth disparity (.bin float32 of H*W)\n"
         << "  -cam <id>     Camera ID for stereo camera capture (overrides -left and -right)\n"
         << "  -h            Show this help message\n"
         << "\nNotes:\n"
         << "- When using camera (-cam), -save defaults to 0 unless explicitly set to 1\n"
         << "- With camera and -save=1, only the last " << NEAR_FRAME << " frames are saved\n"
         << "- Left and right paths should contain matching image pairs\n"
         << "- If paths are files, they should be corresponding left/right images\n";
}

// Function to parse command line arguments
Config parse_arguments(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h") {
            config.show_help = true;
        } else if (arg == "-left" && i + 1 < argc) {
            config.left_path = argv[++i];
        } else if (arg == "-right" && i + 1 < argc) {
            config.right_path = argv[++i];
        } else if (arg == "-m" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "-save" && i + 1 < argc) {
            config.save_output = std::stoi(argv[++i]);
        } else if (arg == "-mask" && i + 1 < argc) {
            config.mask_path = argv[++i];
        } else if (arg == "-gt" && i + 1 < argc) {
            config.gt_path = argv[++i];
        } else if (arg == "-cam" && i + 1 < argc) {
            config.camera_id = std::stoi(argv[++i]);
            config.use_camera = true;
        } else if ((arg == "-e" || arg == "--engine") && i + 1 < argc) {
            config.engine_cli = argv[++i];
        }
    }
    
    // If using camera and save parameter wasn't explicitly set, default to 0
    if (config.use_camera) {
        bool save_explicitly_set = false;
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "-save") {
                save_explicitly_set = true;
                break;
            }
        }
        if (!save_explicitly_set) {
            config.save_output = 0;
        }
    }
    
    return config;
}
// Load mask from .png/.jpg (grayscale) or .bin (uint8) with known H, W
cv::Mat load_mask_file(const std::string& path, int width, int height) {
    if (path.empty()) return cv::Mat();
    std::string ext = get_file_extension(path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
        cv::Mat m = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (m.empty()) {
            std::cerr << "Error: cannot read mask image: " << path << std::endl;
            return cv::Mat();
        }
        if (m.cols != width || m.rows != height) {
            cv::resize(m, m, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        }
        return m;
    } else if (ext == ".bin") {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            std::cerr << "Error: cannot open mask bin: " << path << std::endl;
            return cv::Mat();
        }
        std::vector<uint8_t> buf(static_cast<size_t>(width) * height);
        ifs.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(uint8_t));
        if (ifs.gcount() != static_cast<std::streamsize>(buf.size())) {
            std::cerr << "Error: mask bin size mismatch. expected " << buf.size() << " bytes\n";
            return cv::Mat();
        }
        cv::Mat m(height, width, CV_8U);
        std::memcpy(m.data, buf.data(), buf.size());
        return m;
    } else if (ext == ".npy") {
        std::cerr << "Mask .npy not supported in this build. Convert to .png or .bin.\n"
                  << "Python example: np.load('mask.npy').astype(np.uint8).tofile('mask.bin')\n";
        return cv::Mat();
    }
    std::cerr << "Unsupported mask extension: " << ext << std::endl;
    return cv::Mat();
}

// Load ground-truth disparity from .bin (float32) with known H, W
cv::Mat load_gt_disp_bin(const std::string& path, int width, int height) {
    if (path.empty()) return cv::Mat();
    std::string ext = get_file_extension(path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".bin") {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            std::cerr << "Error: cannot open gt bin: " << path << std::endl;
            return cv::Mat();
        }
        std::vector<float> buf(static_cast<size_t>(width) * height);
        ifs.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
        if (ifs.gcount() != static_cast<std::streamsize>(buf.size() * sizeof(float))) {
            std::cerr << "Error: gt bin size mismatch. expected " << buf.size()*sizeof(float) << " bytes\n";
            return cv::Mat();
        }
        cv::Mat gt(height, width, CV_32F);
        std::memcpy(gt.data, buf.data(), buf.size() * sizeof(float));
        return gt;
    } else if (ext == ".npy") {
        std::cerr << "GT .npy not supported in this build. Convert to .bin.\n"
                  << "Python example: np.load('disp_gt.npy').astype(np.float32).tofile('disp_gt.bin')\n";
        return cv::Mat();
    }
    std::cerr << "Unsupported gt extension: " << ext << std::endl;
    return cv::Mat();
}

// Compute EPE: mean absolute error over valid mask pixels
double compute_epe(const cv::Mat& disp_pred, const cv::Mat& disp_gt, const cv::Mat& mask) {
    CV_Assert(disp_pred.type() == CV_32F && disp_gt.type() == CV_32F);
    CV_Assert(disp_pred.rows == disp_gt.rows && disp_pred.cols == disp_gt.cols);
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.rows == disp_pred.rows && mask.cols == disp_pred.cols));

    double sum_err = 0.0;
    size_t count = 0;
    for (int y = 0; y < disp_pred.rows; ++y) {
        const float* p_pred = disp_pred.ptr<float>(y);
        const float* p_gt   = disp_gt.ptr<float>(y);
        const uint8_t* p_m  = mask.empty() ? nullptr : mask.ptr<uint8_t>(y);
        for (int x = 0; x < disp_pred.cols; ++x) {
            if (!p_m || p_m[x]) {
                sum_err += std::abs(p_pred[x] - p_gt[x]);
                ++count;
            }
        }
    }
    return count > 0 ? (sum_err / static_cast<double>(count)) : 0.0;
}

// Apply mask to colormap: invalid pixels set to white (255,255,255)
void apply_mask_to_colormap(cv::Mat& colormap, const cv::Mat& mask) {
    if (mask.empty()) return;
    CV_Assert(colormap.type() == CV_8UC3 && mask.type() == CV_8U);
    CV_Assert(colormap.rows == mask.rows && colormap.cols == mask.cols);
    for (int y = 0; y < colormap.rows; ++y) {
        cv::Vec3b* p = colormap.ptr<cv::Vec3b>(y);
        const uint8_t* pm = mask.ptr<uint8_t>(y);
        for (int x = 0; x < colormap.cols; ++x) {
            if (!pm[x]) p[x] = cv::Vec3b(255,255,255);
        }
    }
}

// Function to check if file is an image
bool is_image_file(const std::string& filename) {
    std::string ext = get_file_extension(filename);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || 
           ext == ".tiff" || ext == ".tif" || ext == ".webp";
}

// Function to check if file is a video
bool is_video_file(const std::string& filename) {
    std::string ext = get_file_extension(filename);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || 
           ext == ".wmv" || ext == ".flv" || ext == ".webm";
}

// Function to get output filename for disparity
std::string get_output_filename(const std::string& input_filename, const std::string& output_dir) {
    const std::string stem = get_filename_without_extension(input_filename);
    return output_dir + "/" + stem + "_disparity.jpg";
}

// Function to process a stereo image pair
bsnn_ret_t process_stereo_pair(const cv::Mat& left_img, const cv::Mat& right_img, 
                        BSNN_MODEL& bsnn_model, cv::Mat& disparity_result, Timer& time) {
    if (left_img.empty() || right_img.empty()) {
        return BSNN_ERR_INPUT_INVALID;
    }
    
    const auto& meta = bsnn_model.GetModelMeta();
    if (bsnn_model.GetInputAttrs().size() < 2) {
        std::cerr << "Error: Model requires 2 input tensors" << std::endl;
        return BSNN_ERR_MODEL_INVALID;
    }

    const auto& input_tensor = meta.input_tensors[0];
    const auto& in0 = bsnn_model.GetInputAttrs()[0];
    const auto& in1 = bsnn_model.GetInputAttrs()[1];
    size_t input_img_len = input_tensor.width * input_tensor.height * input_tensor.channels;
    size_t left_len = in0.size > 0 ? static_cast<size_t>(in0.size) : input_img_len;
    size_t right_len = in1.size > 0 ? static_cast<size_t>(in1.size) : input_img_len;

    // Step 1: Preprocessing
    time.reset();
    uint8_t* left_buffer = new (std::nothrow) uint8_t[left_len];
    uint8_t* right_buffer = new (std::nothrow) uint8_t[right_len];

    if (!left_buffer || !right_buffer) {
        std::cerr << "Error: Failed to allocate memory for input buffers" << std::endl;
        if (left_buffer) delete[] left_buffer;
        if (right_buffer) delete[] right_buffer;
        return BSNN_ERR_MALLOC_FAIL;
    }
    std::memset(left_buffer, 0, left_len);
    std::memset(right_buffer, 0, right_len);

    lightstereo_preprocess(left_img, right_img, meta, left_buffer, right_buffer);
    cout << "-> preprocess time : " << time.elapsed() << " ms" << endl;

    // Step 2: Model inference
    time.reset();
    std::vector<uint8_t*> inputs = {left_buffer, right_buffer};
    std::vector<size_t> sizes = {left_len, right_len};
    bool success = bsnn_model.Run(inputs, sizes);

    if (!success) {
        delete[] left_buffer;
        delete[] right_buffer;
        return BSNN_ERR_FAIL;
    }

    bst_benchmark::print_engine_inference_stats(bsnn_model.engine_label(), bsnn_model.LastInferUs());

    // Step 3: Post-processing
    time.reset();

    lightstereo_process_output(bsnn_model, disparity_result);
    cout << "-> post process time : " << time.elapsed() << " ms" << endl;
    
    bsnn_model.ReleaseOutputBuffer();
    delete[] left_buffer;
    delete[] right_buffer;
    return BSNN_SUCC;
}

// Function to process stereo image pairs from directories
bsnn_ret_t process_directory(const Config& config, BSNN_MODEL& bsnn_model) {
    if (!file_exists(config.left_path) || !is_directory(config.left_path)) {
        cerr << "Error: Left image directory does not exist: " << config.left_path << endl;
        return BSNN_ERR_PARAM_INVALID;
    }
    
    if (!file_exists(config.right_path) || !is_directory(config.right_path)) {
        cerr << "Error: Right image directory does not exist: " << config.right_path << endl;
        return BSNN_ERR_PARAM_INVALID;
    }
    
    // Create output directory if saving is enabled
    if (config.save_output && !create_directory_if_not_exists(config.output_path)) {
        cerr << "Error: Cannot create output directory: " << config.output_path << endl;
        return BSNN_ERR_FAIL;
    }
    
    // Get all image files from left directory
    std::vector<std::string> left_image_files = list_directory_images(config.left_path);
    
    if (left_image_files.empty()) {
        cout << "No image files found in left directory: " << config.left_path << endl;
        return BSNN_SUCC;
    }
    
    cout << "Found " << left_image_files.size() << " left image files to process." << endl;
    
    // Preload optional mask/gt if provided
    cv::Mat mask, gt;
    const auto& meta = bsnn_model.GetModelMeta();
    int width = meta.output_tensors[0].width;
    int height = meta.output_tensors[0].height;
    if (!config.mask_path.empty()) mask = load_mask_file(config.mask_path, width, height);
    if (!config.gt_path.empty()) gt = load_gt_disp_bin(config.gt_path, width, height);

    Timer time;
    bsnn_ret_t ret = BSNN_SUCC;
    for (const auto& left_image_file : left_image_files) {
        std::string left_filename = get_filename(left_image_file);
        std::string right_image_file = config.right_path + "/" + left_filename;
        
        if (!file_exists(right_image_file)) {
            cerr << "Warning: Corresponding right image not found: " << right_image_file << endl;
            continue;
        }
        
        cout << "\nProcessing stereo pair: " << left_filename << endl;
        
        cv::Mat left_img = cv::imread(left_image_file);
        cv::Mat right_img = cv::imread(right_image_file);
        cv::Mat disparity_result;
        
        if ((ret = process_stereo_pair(left_img, right_img, bsnn_model, disparity_result, time)) == BSNN_SUCC) {
            // Create disparity colormap
            cv::Mat disparity_colormap = create_disparity_colormap(disparity_result, 192.0f);
            // Apply mask to colormap (optional)
            if (!mask.empty()) apply_mask_to_colormap(disparity_colormap, mask);
            // Compute and print EPE (optional)
            if (!gt.empty()) {
                double epe = compute_epe(disparity_result, gt, mask);
                cout << "-> EPE: " << epe << endl;
            }
            
            // Save output if enabled
            if (config.save_output) {
                std::string output_file = get_output_filename(left_filename, config.output_path);
                cv::imwrite(output_file, disparity_colormap);
                cout << "-> Output saved to: " << output_file << endl;
            }
        } else {
            cerr << "Failed to process stereo pair: " << left_filename << endl;
        }
    }
    return ret;
}

// Function to process single stereo image file pair
bsnn_ret_t process_image_file(const Config& config, BSNN_MODEL& bsnn_model) {
    if (!file_exists(config.left_path)) {
        cerr << "Error: Left image file does not exist: " << config.left_path << endl;
        return BSNN_ERR_PARAM_INVALID;
    }
    
    if (!file_exists(config.right_path)) {
        cerr << "Error: Right image file does not exist: " << config.right_path << endl;
        return BSNN_ERR_PARAM_INVALID;
    }
    
    // Create output directory if saving is enabled
    if (config.save_output && !create_directory_if_not_exists(config.output_path)) {
        cerr << "Error: Cannot create output directory: " << config.output_path << endl;
        return BSNN_ERR_FAIL;
    }
    
    cv::Mat left_img = cv::imread(config.left_path);
    cv::Mat right_img = cv::imread(config.right_path);
    cv::Mat disparity_result;
    Timer time;
    
    bsnn_ret_t ret = process_stereo_pair(left_img, right_img, bsnn_model, disparity_result, time);
    if (ret == BSNN_SUCC) {
        // Create disparity colormap
        cv::Mat disparity_colormap = create_disparity_colormap(disparity_result, 192.0f);
        // Optional: load and apply mask / compute EPE
        const auto& meta = bsnn_model.GetModelMeta();
        int width = meta.output_tensors[0].width;
        int height = meta.output_tensors[0].height;
        cv::Mat mask = config.mask_path.empty() ? cv::Mat() : load_mask_file(config.mask_path, width, height);
        cv::Mat gt = config.gt_path.empty() ? cv::Mat() : load_gt_disp_bin(config.gt_path, width, height);
        if (!mask.empty()) apply_mask_to_colormap(disparity_colormap, mask);
        if (!gt.empty()) {
            double epe = compute_epe(disparity_result, gt, mask);
            cout << "-> EPE: " << epe << endl;
        }
        
        // Save output if enabled
        if (config.save_output) {
            std::string output_file = get_output_filename(
                get_filename(config.left_path), config.output_path);
            cv::imwrite(output_file, disparity_colormap);
            cout << "-> Output saved to: " << output_file << endl;
        }
    } else {
        cerr << "Failed to process stereo pair" << endl;
    }
    return ret;
}


int main(int argc, char* argv[]) {
    // Parse command line arguments
    Config config = parse_arguments(argc, argv);
    
    // Show help if requested
    if (config.show_help) {
        print_help();
        return 0;
    }

    // Initialize model
    const auto target_engine = bst_bsnn::resolve_target_engine(
        config.engine_cli.empty() ? nullptr : config.engine_cli.c_str());
    BSNN_MODEL bsnn_model(config.model_path, 1, target_engine);
    if (!bsnn_model.model_init_succ_) {
        cerr << "Error: Model initialization failed" << endl;
        return -1;
    }
    
    cout << "Configuration:" << endl;
    cout << "  Model path: " << config.model_path << endl;
    
    bsnn_ret_t ret = BSNN_SUCC;

    if (config.use_camera) {
        cout << "  Error: Camera mode not implemented for LightStereo" << endl;
        return 1;
    } else {
        cout << "  Left path: " << config.left_path << endl;
        cout << "  Right path: " << config.right_path << endl;
        cout << "  Save output: " << (config.save_output ? "Yes" : "No") << endl;
        if (config.save_output) {
            cout << "  Output path: " << config.output_path << endl;
        }
        
        // Check if inputs are directories or files
        bool left_is_dir = is_directory(config.left_path);
        bool right_is_dir = is_directory(config.right_path);
        bool left_is_file = is_regular_file(config.left_path);
        bool right_is_file = is_regular_file(config.right_path);
        
        if (left_is_dir && right_is_dir) {
            cout << "  Input type: Directory pair" << endl;
            ret = process_directory(config, bsnn_model);
        } else if (left_is_file && right_is_file) {
            if (is_image_file(config.left_path) && is_image_file(config.right_path)) {
                cout << "  Input type: Image file pair" << endl;
                ret = process_image_file(config, bsnn_model);
            } else {
                cerr << "Error: Both files must be image files" << endl;
                return 1;
            }
        } else {
            cerr << "Error: Left and right paths must both be directories or both be files" << endl;
            return 1;
        }
    }
    
    if (ret != BSNN_SUCC) {
        cerr << "Error: Program execution failed with code " << ret << endl;
        return -1;
    }

    printf("SUCCESS\n");
    return 0;
}
