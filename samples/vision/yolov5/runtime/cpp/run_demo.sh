#!/bin/sh
# Runner 需要在 -m 目录下找到 .meta；权重在 yolov5_*/model/，不能只用 yolov5_* 顶层
set -e

# yolov5 后处理递归较深，A1000 板端默认栈（8MB）不足会 SIGSEGV
ulimit -s unlimited 2>/dev/null || ulimit -S -s 65536 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
ENGINE="${BST_BSNN_TARGET_ENGINE:-auto}"
# shellcheck source=/dev/null
. "$REPO_ROOT/utils/tools/lib/prebuild_engine.sh"

MODEL_DIR=""
DEVICE=""

append_file_content() {
    file_path="$1"
    if [ -r "$file_path" ]; then
        file_content=$(tr '\000' '\n' < "$file_path" 2>/dev/null)
        if [ -n "$file_content" ]; then
            SYSTEM_INFO="$SYSTEM_INFO
$file_content"
        fi
    fi
}

detect_device_from_system() {
    SYSTEM_INFO=""

    append_file_content "/proc/device-tree/compatible"
    append_file_content "/proc/device-tree/model"
    append_file_content "/sys/firmware/devicetree/base/compatible"
    append_file_content "/sys/firmware/devicetree/base/model"
    append_file_content "/etc/os-release"

    uname_info=$(uname -a 2>/dev/null)
    if [ -n "$uname_info" ]; then
        SYSTEM_INFO="$SYSTEM_INFO
$uname_info"
    fi

    system_info_lc=$(printf "%s" "$SYSTEM_INFO" | tr '[:upper:]' '[:lower:]')

    case "$system_info_lc" in
        *a2000*)
            echo "A2000"
            return 0
            ;;
        *a1000*)
            echo "A1000"
            return 0
            ;;
        *c1200*)
            echo "C1200"
            return 0
            ;;
    esac

    return 1
}

set_model_for_device() {
    case "$1" in
        A1000)
            if [ -f ../../model/a1000/yolov5m-a1000/yolov5m.meta ]; then
                MODEL_DIR="../../model/a1000/yolov5m-a1000"
                return 0
            fi
            ;;
        C1200)
            if [ -f "../../model/c1200/yolov5m-c1200/yolov5m.meta" ]; then
                MODEL_DIR="../../model/c1200/yolov5m-c1200"
                return 0
            fi
            ;;
    esac

    return 1
}

fallback_model_dir() {
    if [ -f "../../model/c1200/yolov5m-c1200/yolov5m.meta" ]; then
        MODEL_DIR="../../model/c1200/yolov5m-c1200"
    elif [ -f "../../model/a1000/yolov5m-a1000/yolov5m.meta" ]; then
        MODEL_DIR="../../model/a1000/yolov5m-a1000"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -A1000)
            DEVICE="A1000"
            set_model_for_device "$DEVICE" || true
            shift
            ;;
        -C1200)
            DEVICE="C1200"
            set_model_for_device "$DEVICE" || true
            shift
            ;;
        -A2000)
            DEVICE="A2000"
            set_model_for_device "$DEVICE" || true
            shift
            ;;
        -m)
            MODEL_DIR="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-A1000 | -C1200 | -A2000] [--engine top|lite] [-m <model_dir>]"
            exit 0
            ;;
        *)
            shift 1
            ;;
    esac
done

if [ -z "$DEVICE" ]; then
    DEVICE=$(detect_device_from_system || true)
    if [ -n "$DEVICE" ]; then
        set_model_for_device "$DEVICE" || true
    fi
fi

if [ -z "$MODEL_DIR" ]; then
    fallback_model_dir
fi

if [ -z "$MODEL_DIR" ]; then
    echo "[Error] No yolov5 model directory found"
    exit 1
fi

if [ -z "$DEVICE" ]; then
    echo "[Warning] Unable to detect device, fallback to available model directory: $MODEL_DIR"
else
    echo "[Info] Device: $DEVICE"
    echo "[Info] Model directory: $MODEL_DIR"
fi

mkdir -p ./output
benchmark_engine_finalize
./prebuild/app/yolov5 -i ../../test_data/139.jpg -m "$MODEL_DIR" -o ./output/inferresult.jpg -e "$ENGINE"