#ifndef BST_BENCHMARK_TIMING_H
#define BST_BENCHMARK_TIMING_H

#include <cinttypes>
#include <cstdio>

namespace bst_benchmark {

/** 统一用微秒统计 NPU 推理段，输出 ms（三位小数）与 FPS（三位小数）。 */
inline void print_engine_inference_stats(const char* engine_label, int64_t elapsed_us) {
    const double ms = static_cast<double>(elapsed_us) / 1000.0;
    const double fps = ms > 0.0 ? 1000.0 / ms : 0.0;
    std::printf("-> %s engine inference FPS: %.3f\n", engine_label, fps);
    std::printf("-> bsnn model inference time : %.3f ms\n", ms);
}

}  // namespace bst_benchmark

#endif
