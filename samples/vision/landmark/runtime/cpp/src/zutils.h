#ifndef BSNN_PROJECTS_UTILS_H
#define BSNN_PROJECTS_UTILS_H

#include <iostream>
#include <cmath>
#include <tuple>
#include <chrono>
#include <queue>
#include <algorithm>
#include <utility>
#include <numeric>
#include <opencv2/opencv.hpp>
// #include <arm_neon.h>

#define EPSILON 1e-5
bool equal(float a, float b);

struct ObjInfo {
    int classId;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
    float area;
    std::vector<float> mask_coeff; // optional mask coefficients (seg)
    std::vector<float> keypoints;  // optional keypoints (pose)
};

int64_t GetIndex(int64_t c, int64_t y, int64_t x, int64_t width, int64_t height, int64_t num_channels);

std::tuple<int64_t, int64_t , int64_t> GetCYX(int64_t index, int64_t width, int64_t height, int64_t num_channels);


void max_pool_2d(float* input_ts, 
                const int width, const int height,  const int channels, 
                float* output_ts, int kernel_size, int padding, int stride);

void test_maxpool();

template<typename T>
inline float sigmoid(T x) { 
    return (1.0 / (1 + exp(-static_cast<float>(x)))); 
}

// do element-wise sigmoid on tensor
template<typename T>
void sigmoid_on_tensor(T* input, float* output, int64_t tensor_size) {
    for(int64_t i = 0; i < tensor_size; i++) {
        output[i] = sigmoid<T>(input[i]);
    }
}

// multiply a scale on every elements in tensor
template<typename T1, typename T2>
void elementwise_mul(T1* inputs, 
                    T2 value, 
                    int64_t tensor_size,
                    float* outputs) {
    for(int64_t i = 0; i < tensor_size; i++) {
        outputs[i] = static_cast<float>(inputs[i]) * static_cast<float>(value);
    }
}


class Timer {
public:
    Timer():_begin(std::chrono::high_resolution_clock::now()) {

    }
    void reset() {
        _begin = std::chrono::high_resolution_clock::now();
    }

    // time elapsed, default milliseconds
    template<typename Duration=std::chrono::milliseconds>
    int64_t elapsed() const {
        return std::chrono::duration_cast<Duration>(std::chrono::high_resolution_clock::now() - _begin).count();
    }

    int64_t elapsed_micro() const {
        return elapsed<std::chrono::microseconds>();
    }

    int64_t elapsed_nano() const {
        return elapsed<std::chrono::nanoseconds>();
    }

    int64_t elapsed_seconds() const {
        return elapsed<std::chrono::seconds>();
    }


    int64_t elapsed_minutes() const {
        return elapsed<std::chrono::minutes>();
    }

    int64_t elapsed_hours() const {
        return elapsed<std::chrono::hours>();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _begin;
};


template<typename T>
struct ValueComp {
    bool operator() (const std::pair<T, int64_t>& lhs,
                    const std::pair<T, int64_t>& rhs) {
        
        return (lhs.first > rhs.first) ||
                (lhs.first == rhs.first && lhs.second < rhs.second);
    }
};

// get top k elements and  coorespoding indices from an array
template<typename T>
void GetTopK(const T* inputs, 
            const int64_t n, 
            const int64_t k,
            const int64_t stride,
            const int64_t src_offset,
            const int64_t dst_offset,
            T* outputs,
            int64_t* indices) {
    const T* src_ptr = inputs + src_offset;
    std::vector<std::pair<T, int64_t>> heap_data;
    heap_data.reserve(k);
    
    for(int64_t i = 0; i < k && i < n; i++) {
        heap_data.emplace_back(*src_ptr, i);
        src_ptr += stride;
    }

    std::priority_queue<
        std::pair<T, int64_t>,
        std::vector<std::pair<T, int64_t>>,
        ValueComp<T>> pq(ValueComp<T>(), std::move(heap_data));

    for(int64_t i = k; i < n; i++) {
        if(pq.top().first < *src_ptr) {
            pq.pop();
            pq.emplace(*src_ptr, i);
        }

        src_ptr += stride;
    }
    
    int64_t cnt = 0;
    while(!pq.empty()) {
        const std::pair<T, int64_t>& ele = pq.top();
        
        outputs[cnt + dst_offset * k] = ele.first;
        indices[cnt + dst_offset * k] = ele.second * stride + src_offset;
        cnt++;
        pq.pop();
    }

}


inline u_char saturate_uchar(int v) {
    u_char uchar_max = std::numeric_limits<u_char>::max();
    return static_cast<u_char>(static_cast<unsigned int>(v) <= uchar_max? v: v > 0? uchar_max : 0);
}

inline u_char saturate_uchar(float v) {
    int iv = round(v);
    return saturate_uchar(iv);
}


void RGB2NV12(const cv::Mat& src, unsigned char * yuvbuff);
void softmax(float* input_p, int C, int H, int W);
void argmax(float* input_p, uint8_t* output_p, int C, int H, int W);

void soft_nms(std::vector<ObjInfo>& bboxes_in, std::vector<ObjInfo>& bboxes_out, 
            const int method, const float& sigma, const float& iou_thresh, const float& threshold);

#endif

