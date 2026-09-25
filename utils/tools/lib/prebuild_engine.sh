#!/bin/sh
# 供 models/*/Deploy/C++/prebuild/*.sh source，解析 Kalos TOP/LITE engine。
# 用法:
#   ENGINE="${BST_BSNN_TARGET_ENGINE:-lite}"
#   . "$REPO_ROOT/commons/tools/benchmark/prebuild_engine.sh"
#   在参数循环中: benchmark_engine_consume_option "$1" "$2" && shift 2 && continue
#   循环结束后: benchmark_engine_finalize

benchmark_engine_consume_option() {
    case "$1" in
        --engine)
            if [ -z "${2:-}" ]; then
                echo "[Error] --engine requires top, lite, or auto" >&2
                exit 1
            fi
            ENGINE="$2"
            return 0
            ;;
    esac
    return 1
}

benchmark_engine_finalize() {
    case "$ENGINE" in
        top|lite|auto) ;;
        *)
            echo "[Error] BSNN engine must be top, lite, or auto (got: $ENGINE)" >&2
            exit 1
            ;;
    esac
    export BST_BSNN_TARGET_ENGINE="$ENGINE"
    echo "[Info] BSNN engine: $ENGINE"
}
