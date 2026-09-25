#include "image_process.h"
#include <algorithm>
#include <cstdio>
using namespace std;

cv::Mat crop_and_resize(const cv::Mat& src) {
    int x1 = 0, y1 = 160, x2 = 1280, y2 = 720;

    if (src.cols < x2 || src.rows < y2) {
        std::cerr << "Error: Image size (" << src.cols << "x" << src.rows 
                  << ") is smaller than crop region (" << x2 << "x" << y2 << ")" << std::endl;
        return cv::Mat();
    }

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat cropped = src(roi);

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(800, 320), 0.0, 0.0, cv::INTER_LINEAR);

    return resized;
}

// 处理 std::vector<T>
template<typename T>
void save_to_bin(const std::vector<T>& data, 
                 const std::string& filename,
                 const std::vector<int>& shape ,
                 bool save_shape  ) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    if (save_shape) {
        // 写入形状信息
        int shape_size = static_cast<int>(shape.size());
        file.write(reinterpret_cast<const char*>(&shape_size), sizeof(int));
        
        for (int dim_size : shape) {
            file.write(reinterpret_cast<const char*>(&dim_size), sizeof(int));
        }
    }
    
    // 写入数据
    if (!data.empty()) {
        file.write(reinterpret_cast<const char*>(data.data()), 
                   data.size() * sizeof(T));
    }
    
    file.close();
}

// 处理 cv::Mat
void save_to_bin(const cv::Mat& mat, 
                 const std::string& filename,
                 const std::vector<int>& shape ,
                 bool save_shape ) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    if (save_shape) {
        // 写入形状信息
        int shape_size = static_cast<int>(shape.size());
        file.write(reinterpret_cast<const char*>(&shape_size), sizeof(int));
        
        for (int dim_size : shape) {
            file.write(reinterpret_cast<const char*>(&dim_size), sizeof(int));
        }
    }
    
    // 写入数据
    if (!mat.empty()) {
        file.write(reinterpret_cast<const char*>(mat.data), 
                   mat.total() * mat.elemSize());
    }
    
    file.close();
}

// 处理原始指针和计数
template<typename T>
void save_to_bin(T* data_ptr, 
                 size_t count,
                 const std::string& filename,
                 const std::vector<int>& shape ,
                 bool save_shape ) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    if (save_shape) {
        // 写入形状信息
        int shape_size = static_cast<int>(shape.size());
        file.write(reinterpret_cast<const char*>(&shape_size), sizeof(int));
        
        for (int dim_size : shape) {
            file.write(reinterpret_cast<const char*>(&dim_size), sizeof(int));
        }
    }
    
    // 写入数据
    if (count > 0 && data_ptr != nullptr) {
        file.write(reinterpret_cast<const char*>(data_ptr), 
                   count * sizeof(T));
    }
    
    file.close();
}

cv::Mat loadTensorFromBin(const std::string& filename, 
                         int height, 
                         int width, 
                         int channels = 3) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    size_t expected_size = height * width * channels;
    
    // Read the image data
    std::vector<int8_t> buffer(expected_size);
    file.read(reinterpret_cast<char*>(buffer.data()), expected_size * sizeof(int8_t));
    
    if (file.gcount() != static_cast<std::streamsize>(expected_size * sizeof(int8_t))) {
        throw std::runtime_error("Error reading file or file size doesn't match specified dimensions");
    }

    file.close();

    // Create Mat with specified dimensions
    cv::Mat img(height, width, CV_8UC3); // CV_8UC3 for 3-channel uint8 image
    
    // Copy data from buffer to Mat, converting from int8 to uint8
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                int idx = i * width * channels + j * channels + c;
                img.at<cv::Vec3b>(i, j)[c] = static_cast<uchar>(static_cast<uint8_t>(buffer[idx]));
            }
        }
    }

    return img;
}

template <typename Stype, typename Dtype>
inline void convert_hwc2chw(size_t ch, size_t w, size_t h, const Stype* src, Dtype* dst) {
  size_t index = 0UL;
  const size_t hw_stride = w * h;
  for (size_t s = 0UL; s < hw_stride; ++s) {
    size_t stride_index = s;
    for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
      dst[stride_index] = static_cast<Dtype>(src[index++]);
    }
  }
}

// Helper function to compute softmax
cv::Mat softmax(const cv::Mat& input) {
    cv::Mat exp_input, result;
    cv::exp(input - cv::mean(input)[0], exp_input);
    cv::reduce(exp_input, result, 1, cv::REDUCE_SUM, -1);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            exp_input.at<float>(i, j) /= result.at<float>(i, 0);
        }
    }
    return exp_input;
}

CondLanePostProcessor::CondLanePostProcessor(float hm_thr, int min_points, float nms_thr, 
                                           bool use_offset, float hm_downscale, float mask_downscale)
    : hm_thr_(hm_thr), min_points_(min_points), nms_thr_(nms_thr), 
      use_offset_(use_offset), hm_downscale_(hm_downscale), mask_downscale_(mask_downscale) {}

float CondLanePostProcessor::cal_dis(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<Seed> CondLanePostProcessor::nms_seeds_tiny(const std::vector<Seed>& seeds) {
    if (seeds.empty()) return seeds;
    
    auto cal_dis = [](const cv::Point2f& p1, const cv::Point2f& p2) {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        return std::sqrt(dx * dx + dy * dy);
    };

    auto search_groups = [&](const cv::Point2f& coord, const std::vector<std::vector<Seed>>& groups) {
        for (size_t idx_group = 0; idx_group < groups.size(); ++idx_group) {
            for (const auto& group_point : groups[idx_group]) {
                cv::Point2f group_point_coord(group_point.coord[0], group_point.coord[1]);
                if (cal_dis(coord, group_point_coord) <= nms_thr_) {
                    return static_cast<int>(idx_group);
                }
            }
        }
        return -1;
    };

    auto choose_highest_score = [](const std::vector<Seed>& group) {
        int highest_idx = -1;
        float highest_score = -1;
        for (size_t i = 0; i < group.size(); ++i) {
            if (group[i].score > highest_score) {
                highest_score = group[i].score;
                highest_idx = i;
            }
        }
        return highest_idx;
    };

    std::vector<std::vector<Seed>> groups;
    std::vector<Seed> update_seeds;

    for (const auto& seed : seeds) {
        cv::Point2f coord(seed.coord[0], seed.coord[1]);
        int idx_group = search_groups(coord, groups);
        if (idx_group < 0) {
            groups.push_back({seed});
        } else {
            groups[idx_group].push_back(seed);
        }
    }

    for (auto& group : groups) {
        int choose_idx = choose_highest_score(group);
        if (choose_idx >= 0) {
            update_seeds.push_back(group[choose_idx]);
        }
    }

    return update_seeds;
}

std::vector<std::vector<int>> CondLanePostProcessor::get_range(const std::vector<cv::Mat>& ranges) {
    std::vector<std::vector<int>> lane_ends;
    
    for (size_t idx = 0; idx < ranges.size(); ++idx) {
        const auto& range_mat = ranges[idx];
        
        std::cout << "Range matrix " << idx << " shape: " << range_mat.rows << "x" << range_mat.cols << std::endl;
        
        // range_mat 应该是 [1, 40] 形状，值为0或1（argmax结果）
        if (range_mat.rows == 1 && range_mat.cols == 40) {
            std::vector<int> lane_range;
            for (int j = 0; j < 40; ++j) {
                lane_range.push_back(static_cast<int>(range_mat.at<float>(0, j)));
            }
            
            int max_rows = 40;
            int min_idx = -1, max_idx = -1;
            
            // 找到第一个有效的行
            for (int row_idx = 0; row_idx < max_rows; ++row_idx) {
                if (lane_range[row_idx]) {  // if valid
                    min_idx = row_idx  ;  // Python: row_idx - 1
                    break;
                }
            }
            
            // 找到最后一个有效的行（反向遍历原数组）
            for (int row_idx = max_rows - 1; row_idx >= 0; --row_idx) {
                if (lane_range[row_idx]) {  // 检查原数组的倒数第几个元素
                    max_idx = row_idx;  // 不是 max_rows - row_idx
                    break;
                }
            }
            
            if (max_idx != -1) {
                max_idx = std::min(max_rows - 1, max_idx);
            }
            if (min_idx != -1) {
                min_idx = std::max(0, min_idx);
            }
            
            // 修正：如果 min_idx 为 -1，说明没有找到有效的开始位置
            if (min_idx == -1) min_idx = 0;
            if (max_idx == -1) max_idx = max_rows - 1;
            
            std::vector<int> lane_end = {min_idx, max_idx};
            lane_ends.push_back(lane_end);
            
            std::cout << "Lane ends for range " << idx << ": [" << lane_end[0] << ", " << lane_end[1] << "]" << std::endl;
        } else {
            std::vector<int> default_end = {0, 39};
            lane_ends.push_back(default_end);
            std::cout << "Using default lane ends [0, 39]" << std::endl;
        }
    }
    
    return lane_ends;
}

std::vector<Lane> CondLanePostProcessor::process_lanes(const std::vector<Seed>& seeds, float downscale) {
    std::vector<Lane> lanes;
    
    std::cout << "Processing " << seeds.size() << " seeds in process_lanes" << std::endl;
    
    for (size_t seed_idx = 0; seed_idx < seeds.size(); ++seed_idx) {
        const auto& seed = seeds[seed_idx];
        
        std::cout << "\n=== Processing seed " << seed_idx << " ===" << std::endl;
        std::cout << "Reg shape: " << seed.reg.rows << "x" << seed.reg.cols << std::endl;
        std::cout << "Mask shape: " << seed.mask.rows << "x" << seed.mask.cols << std::endl;
        std::cout << "Range shape: " << seed.range.rows << "x" << seed.range.cols << std::endl;
        
        if (seed.mask.empty() || seed.reg.empty()) {
            std::cout << "Seed " << seed_idx << " has empty mask or reg" << std::endl;
            continue;
        }
        
        if (seed.mask.rows != 40 || seed.mask.cols != 100) {
            std::cout << "Warning: Mask shape is not [40, 100], actual: [" 
                      << seed.mask.rows << ", " << seed.mask.cols << "]" << std::endl;
            continue;
        }
        
        // 计算 softmax
        cv::Mat mask_softmax = softmax(seed.mask);
        std::cout << "Mask softmax computed" << std::endl;
        
        // 计算每行的期望 x 坐标
        cv::Mat row_pos = cv::Mat::zeros(40, 1, CV_32F);
        for (int y = 0; y < 40; ++y) {
            float sum = 0.0f, weighted_sum = 0.0f;
            for (int x = 0; x < 100; ++x) {
                float weight = mask_softmax.at<float>(y, x);
                sum += weight;
                weighted_sum += weight * x;
            }
            row_pos.at<float>(y, 0) = sum > 0 ? static_cast<float>(weighted_sum / sum) : 0;
        }
        
        // 获取车道线范围
        std::vector<cv::Mat> single_range = {seed.range};
        auto lane_ends = get_range(single_range);
        
        if (lane_ends.empty()) {
            std::cout << "No lane ends found for seed " << seed_idx << std::endl;
            continue;
        }
        
        int start_row = lane_ends[0][0];
        int end_row = lane_ends[0][1];
        
        std::cout << "Lane range: [" << start_row << ", " << end_row << "]" << std::endl;
        
        Lane lane;
        lane.id_class = seed.id_class;
        lane.score = seed.score;
        lane.seed = seed;
        
        for (int y = start_row; y <= end_row; ++y) {
            if (y >= 40) {
                std::cout << "Row " << y << " out of bounds for row_pos with 40 rows" << std::endl;
                break;
            }
            
            int x = static_cast<int>(row_pos.at<float>(y, 0));
            if (x >= 100) {
                std::cout << "X coordinate " << x << " out of bounds for reg with 100 cols" << std::endl;
                x = 99;
            }
            
            float reg_value = use_offset_ ? seed.reg.at<float>(y, x) : 0.5f;
            float adjusted_x = x + reg_value;
            
            lane.points.emplace_back(adjusted_x * downscale, y * downscale);
        }
        
        std::cout << "Lane " << seed_idx << " has " << lane.points.size() << " points" << std::endl;
        
        if (lane.points.size() > 1) {
            lanes.push_back(lane);
            std::cout << "Added lane " << seed_idx << " with " << lane.points.size() << " points" << std::endl;
        } else {
            std::cout << "Lane " << seed_idx << " has only " << lane.points.size() 
                      << " points, not added" << std::endl;
        }
    }
    
    std::cout << "Total lanes created: " << lanes.size() << std::endl;
    return lanes;
}

std::vector<Lane> CondLanePostProcessor::process(const std::vector<Seed>& output, float downscale) {
    if (output.empty()) return {};
    
    auto seeds = nms_seeds_tiny(output);
    if (seeds.empty()) return {};
    
    return process_lanes(seeds, downscale);
}

// Main postprocessor implementation
LaneDetectionPostProcessor::LaneDetectionPostProcessor(const std::vector<int>& crop_bbox)
    : crop_bbox_(crop_bbox) {}


bool LaneDetectionPostProcessor::load_outputs(const BSNN_MODEL& model,
                         uint32_t tensors_per_batch,
                         bool a1000_layout) {
    (void)a1000_layout;
    bool success = true;

    cv::Mat* targets[] = {&hm_, &hm_maxpool_, &regs_, &masks_, &feat_range_};
    const char* names[] = {"hm", "hm_maxpool", "regs", "masks", "feat_range"};
    const uint32_t n = std::min(tensors_per_batch, model.GetOutputTensorCount());
    if (n < 5) {
        std::cerr << "ERROR: expected 5 output tensors, got " << n << std::endl;
        return false;
    }

    for (uint32_t index = 0; index < n && index < 5; index++) {
        const bsnn_tensor_attr_t& attr = model.GetOutputAttrs()[index];
        uint8_t* raw = model.OutputVa(index);
        if (!raw) {
            std::cerr << "ERROR: output va is NULL for " << names[index] << std::endl;
            success = false;
            continue;
        }

        const float dequant_scale = BSNN_MODEL::QuantScale(attr, 0);
        const bool is_uint8 = (attr.dtype == BSNN_TENSOR_UINT8);
        const int elem = 1;
        int out_c = 0, out_w = 0, out_h = 0, pitch0 = 0;
        bool dwh = (attr.dorder == BSNN_TENSOR_DWH);

        if (dwh) {
            out_c = static_cast<int>(attr.dims[0]);
            out_w = static_cast<int>(attr.dims[1]);
            out_h = static_cast<int>(attr.dims[2]);
            pitch0 = static_cast<int>(attr.pitches[0]);
            if (pitch0 < out_c) pitch0 = out_c;
        } else {
            out_w = static_cast<int>(attr.dims[0]);
            out_h = static_cast<int>(attr.dims[1]);
            out_c = static_cast<int>(attr.dims[2]);
            pitch0 = static_cast<int>(attr.pitches[0]);
            if (pitch0 < out_w * elem) pitch0 = out_w * elem;
        }

        const int HxW = out_h * out_w;
        std::vector<float> chw(static_cast<size_t>(HxW) * static_cast<size_t>(out_c));
        printf("-> load %s: C=%d W=%d H=%d pitch=%d dorder=%d dtype=%d scale=%.6f\n",
               names[index], out_c, out_w, out_h, pitch0, attr.dorder, attr.dtype, dequant_scale);

        if (dwh) {
            // DWH / NHWC: each spatial pixel has padded C channels.
            std::vector<uint8_t> hwc(static_cast<size_t>(HxW) * static_cast<size_t>(out_c));
            for (int i = 0; i < HxW; ++i) {
                for (int j = 0; j < out_c; ++j) {
                    hwc[static_cast<size_t>(i) * out_c + j] = raw[static_cast<size_t>(i) * pitch0 + j];
                }
            }
            std::vector<uint8_t> tmp(hwc.size());
            convert_hwc2chw(out_c, out_w, out_h, hwc.data(), tmp.data());
            for (size_t i = 0; i < tmp.size(); ++i) {
                if (is_uint8) {
                    chw[i] = static_cast<float>(tmp[i]) * dequant_scale;
                } else {
                    chw[i] = static_cast<float>(static_cast<int8_t>(tmp[i])) * dequant_scale;
                }
            }
        } else {
            // WHD / NCHW: channel-planar, W padded.
            for (int c = 0; c < out_c; ++c) {
                for (int h = 0; h < out_h; ++h) {
                    const uint8_t* src_row =
                        raw + (static_cast<size_t>(c) * out_h + static_cast<size_t>(h)) * pitch0;
                    float* dst_row = chw.data() + (static_cast<size_t>(c) * out_h + static_cast<size_t>(h)) * out_w;
                    for (int w = 0; w < out_w; ++w) {
                        if (is_uint8) {
                            dst_row[w] = static_cast<float>(src_row[w]) * dequant_scale;
                        } else {
                            dst_row[w] = static_cast<float>(static_cast<int8_t>(src_row[w])) * dequant_scale;
                        }
                    }
                }
            }
        }

        if (out_c == 1) {
            *targets[index] = cv::Mat(out_h, out_w, CV_32F, chw.data()).clone();
        } else {
            *targets[index] = cv::Mat(out_c, HxW, CV_32F, chw.data()).clone();
        }
    }

    if (!success) {
        std::cerr << "Failed to load one or more output tensors." << std::endl;
        return false;
    }
    std::cout << "Successfully loaded all output tensors." << std::endl;
    return true;
}

bool LaneDetectionPostProcessor::load_tensor_from_bin(const std::string& filename, cv::Mat& tensor, std::vector<int>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ Cannot open file: " << filename << std::endl;
        return false;
    }

    // Read number of dimensions (assuming max 4D)
    int32_t dims[4];
    file.read(reinterpret_cast<char*>(dims), sizeof(int32_t) * 4);

    // Determine actual shape
    shape.clear();
    for (int i = 0; i < 4; ++i) {
        if (dims[i] > 0) {
            shape.push_back(dims[i]);
        } else {
            break;
        }
    }

    if (shape.empty()) {
        std::cerr << "Invalid shape in " << filename << std::endl;
        return false;
    }

    // Calculate total size
    size_t total_size = 1;
    for (int s : shape) total_size *= s;

    std::vector<float> data(total_size);
    file.read(reinterpret_cast<char*>(data.data()), sizeof(float) * total_size);
    file.close();

    // Reshape based on dimensions
    if (shape.size() == 4) {
        int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
        if (N == 1) {
            if (C == 1) {
                tensor = cv::Mat(H, W, CV_32F, data.data()).clone();
            } else {
                tensor = cv::Mat(C, H * W, CV_32F, data.data()).clone();
            }
        }
    } else if (shape.size() == 3) {
        int C = shape[0], H = shape[1], W = shape[2];
        tensor = cv::Mat(C, H * W, CV_32F, data.data()).clone();
    } else if (shape.size() == 2) {
        tensor = cv::Mat(shape[0], shape[1], CV_32F, data.data()).clone();
    } else {
        std::cerr << "Unsupported tensor dimension: " << shape.size() << std::endl;
        return false;
    }

    return true;
}

bool LaneDetectionPostProcessor::load_reference_outputs() {
    std::vector<int> shape;
    bool success = true;
    success &= load_tensor_from_bin("hm.bin", hm_, shape);
    std::cout << "HM shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3] << std::endl;
    
    success &= load_tensor_from_bin("hm_maxpool.bin", hm_maxpool_, shape);
    std::cout << "HM_MaxPool shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3] << std::endl;
    
    success &= load_tensor_from_bin("regs.bin", regs_, shape);
    std::cout << "Regs shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3] << std::endl;
    
    success &= load_tensor_from_bin("masks.bin", masks_, shape);
    std::cout << "Masks shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3] << std::endl;
    
    success &= load_tensor_from_bin("feat_range.bin", feat_range_, shape);
    std::cout << "Feat_Range shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3] << std::endl;

    if (!success) {
        std::cerr << "❌ Failed to load one or more ONNX output files." << std::endl;
        return false;
    }

    std::cout << "✅ Successfully loaded all ONNX output tensors." << std::endl;
    return true;
}


std::vector<Seed> LaneDetectionPostProcessor::ctdet_decode() {
    std::vector<Seed> seeds;
    
    // Step 1: Find local maxima (hm == hm_maxpool)
    cv::Mat heat_nms;

    double min_val, max_val;
    cv::minMaxLoc(hm_, &min_val, &max_val);
    std::cout << "Heatmap min: " << min_val << ", max: " << max_val << std::endl;
    if (max_val < 0.1f) {
        std::cerr << "Warning: Heatmap values are very low!" << std::endl;
    }

    cv::compare(hm_, hm_maxpool_, heat_nms, cv::CMP_EQ);
    heat_nms.convertTo(heat_nms, CV_32F);
    
    // Step 2: Apply threshold to heatmap values (not to the comparison result)
    // Only keep points that are local maxima AND have score > threshold
    cv::Mat final_mask = cv::Mat::zeros(hm_.size(), CV_8UC1);
    for (int y = 0; y < hm_.rows; ++y) {
        for (int x = 0; x < hm_.cols; ++x) {
            if (heat_nms.at<float>(y, x) > 0.5f && hm_.at<float>(y, x) > 0.5f) {
                final_mask.at<uchar>(y, x) = 255;
            }
        }
    }
    
    // Find non-zero points
    std::vector<cv::Point> points;
    cv::findNonZero(final_mask, points);
    
    std::cout << "Found " << points.size() << " seed points after NMS and threshold" << std::endl;
    
    for (const auto& pt : points) {
        Seed seed;
        seed.coord = {pt.x, pt.y};
        seed.id_class = 1;  // Assuming single class
        seed.score = hm_.at<float>(pt.y, pt.x);
        std::cout << "seed score " << seed.score << std::endl;
        seeds.push_back(seed);
    }
    
    return seeds;
}

std::vector<int> LaneDetectionPostProcessor::parse_pos(const std::vector<Seed>& seeds, int h, int w) {
    std::vector<int> positions;
    for (const auto& seed : seeds) {
        int label = seed.id_class - 1;  // 通常为0
        int x = seed.coord[0];  // 在20x50网格中的x坐标
        int y = seed.coord[1];  // 在20x50网格中的y坐标
        
        // 正确的位置计算：label * (h*w) + y * w + x
        int pos = label * h * w + y * w + x;
        positions.push_back(pos);
        
        std::cout << "Seed coord: [" << x << ", " << y << "], label: " << label 
                  << ", pos: " << pos << std::endl;
    }
    return positions;
}

std::vector<std::vector<cv::Point2f>> LaneDetectionPostProcessor::adjust_result(const std::vector<Lane>& lanes) {
    float h_img = 320.0f, w_img = 800.0f;
    float ratio_x = (crop_bbox_[2] - crop_bbox_[0]) / w_img;
    float ratio_y = (crop_bbox_[3] - crop_bbox_[1]) / h_img;
    float offset_x = crop_bbox_[0];
    float offset_y = crop_bbox_[1];
    
    std::vector<std::vector<cv::Point2f>> results;
    for (const auto& lane : lanes) {
        std::vector<cv::Point2f> pts;
        for (const auto& pt : lane.points) {
            pts.emplace_back(pt.x * ratio_x + offset_x, pt.y * ratio_y + offset_y);
        }
        if (pts.size() > 1) {
            results.push_back(pts);
        }
    }
    
    return results;
}
std::vector<std::vector<cv::Point2f>> LaneDetectionPostProcessor::postprocess() {
    auto seeds = ctdet_decode();
    
    if (seeds.empty()) {
        return {};
    }
    
    int h_hm = hm_.rows;  // 20
    int w_hm = hm_.cols;  // 50
    auto positions = parse_pos(seeds, h_hm, w_hm);
    
    for (size_t i = 0; i < seeds.size() && i < positions.size(); ++i) {
        int pos = positions[i];

        if (pos < regs_.rows) {
            // 重塑 reg 和 mask 为 [40, 100]
            cv::Mat reg_slice = regs_.row(pos);
            cv::Mat mask_slice = masks_.row(pos);
            
            if (reg_slice.cols == 40 * 100) {
                seeds[i].reg = reg_slice.reshape(1, 40);  // 40x100
                seeds[i].mask = mask_slice.reshape(1, 40);  // 40x100
            }
        }
        
        if (pos < feat_range_.rows) {
            // feat_range_ 是 [1000, 2*40] = [1000, 80]
            cv::Mat range_slice = feat_range_.row(pos);  // [1, 80]
            
            if (range_slice.cols == 2 * 40) {
                // Reshape to [2, 40]
                cv::Mat range_2d = range_slice.reshape(1, 2);  // 2x40
                
                // 执行 argmax 操作：沿着行（dim 0）找到最大值的索引
                // 结果应该是 [1, 40]，每个位置是 0 或 1
                cv::Mat argmax_result = cv::Mat::zeros(1, 40, CV_32F);
                
                for (int col = 0; col < 40; ++col) {
                    float val0 = range_2d.at<float>(0, col);
                    float val1 = range_2d.at<float>(1, col);
                    
                    // 找到最大值的索引 (0 or 1)
                    int max_idx = (val0 > val1) ? 0 : 1;
                    argmax_result.at<float>(0, col) = static_cast<float>(max_idx);
                }
                
                seeds[i].range = argmax_result;  // [1, 40]
            }
        }
    }
    
    CondLanePostProcessor processor(0.5f, 5, 3.0f, true, 16.0f, 8.0f);
    auto lanes = processor.process(seeds, 8.0f);
    
    return adjust_result(lanes);
}

void LaneDetectionPostProcessor::visualize(const std::string& original_image_path, 
                                          const std::string& output_image_path,
                                          const std::vector<std::vector<cv::Point2f>>& lanes) {
    cv::Mat img = cv::imread(original_image_path);
    if (img.empty()) {
        std::cerr << "Cannot load image: " << original_image_path << std::endl;
        return;
    }
    
    // Define colors for different lanes
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(255, 255, 0),  // Cyan
        cv::Scalar(255, 0, 255),  // Magenta
        cv::Scalar(0, 255, 255),  // Yellow
        cv::Scalar(128, 0, 128),  // Purple
    };
    
    for (size_t i = 0; i < lanes.size(); ++i) {
        const auto& lane = lanes[i];
        if (lane.size() < 2) continue;
        
        cv::Scalar color = colors[i % colors.size()];
        
        // Draw lane as connected lines
        for (size_t j = 1; j < lane.size(); ++j) {
            cv::line(img, lane[j-1], lane[j], color, 2, cv::LINE_AA);
        }

    }
    
    cv::imwrite(output_image_path, img);
    std::cout << "✅ Visualization saved to " << output_image_path << std::endl;
}