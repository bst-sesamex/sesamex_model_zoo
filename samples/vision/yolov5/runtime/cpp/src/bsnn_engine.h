#ifndef BST_BSNN_ENGINE_H
#define BST_BSNN_ENGINE_H

#include "bsnn.h"

#include <cstdlib>
#include <cstring>

// Zoo layout aliases: WHD == NCHW, DWH == NHWC (same numeric values as the old SDK).
#ifndef BSNN_TENSOR_WHD
#define BSNN_TENSOR_WHD BSNN_TENSOR_NCHW
#define BSNN_TENSOR_DWH BSNN_TENSOR_NHWC
#endif

#ifndef BSNN_SUCC
#define BSNN_SUCC BSNN_SUCCESS
#define BSNN_ERR_FAIL BSNN_NOK
#define BSNN_ERR_INPUT_INVALID BSNN_ERR_INVALID
#endif

using bsnn_tensor_attr_t = bsnn_tensor_attr;
using bsnn_ret_t = BSNN_RET;

namespace bst_bsnn {

inline const char* engine_label(int core_mask) {
    if (core_mask & BSNN_CORE_MASK_ENGINE_TOP) {
        return "top";
    }
    if (core_mask & BSNN_CORE_MASK_ENGINE_LITE) {
        return "lite";
    }
    return "auto";
}

inline int engine_from_name(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return BSNN_CORE_MASK_AUTO;
    }
    if (std::strcmp(name, "top") == 0 || std::strcmp(name, "TOP") == 0) {
        return BSNN_CORE_MASK_ENGINE_TOP;
    }
    if (std::strcmp(name, "lite") == 0 || std::strcmp(name, "LITE") == 0) {
        return BSNN_CORE_MASK_ENGINE_LITE;
    }
    if (std::strcmp(name, "auto") == 0 || std::strcmp(name, "AUTO") == 0) {
        return BSNN_CORE_MASK_AUTO;
    }
    return BSNN_CORE_MASK_AUTO;
}

inline int resolve_target_engine(const char* cli_engine = nullptr) {
    if (cli_engine != nullptr && cli_engine[0] != '\0') {
        return engine_from_name(cli_engine);
    }
    const char* env = std::getenv("BST_BSNN_TARGET_ENGINE");
    if (env != nullptr && env[0] != '\0') {
        return engine_from_name(env);
    }
    return BSNN_CORE_MASK_AUTO;
}

inline bool parse_engine_cli(int argc, char** argv, int& index, const char** out_cli) {
    if (index + 1 >= argc) {
        return false;
    }
    if (std::strcmp(argv[index], "-e") != 0 && std::strcmp(argv[index], "--engine") != 0) {
        return false;
    }
    *out_cli = argv[index + 1];
    index += 1;
    return true;
}

}  // namespace bst_bsnn

#endif
