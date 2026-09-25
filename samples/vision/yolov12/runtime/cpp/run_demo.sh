#!/bin/sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
ENGINE="${BST_BSNN_TARGET_ENGINE:-auto}"
# shellcheck source=/dev/null
. "$REPO_ROOT/utils/tools/lib/prebuild_engine.sh"

DEVICE=""
YOLOV10_MODEL="../../model/c1200/yolov10n-det-c1200/"
YOLOV11_MODEL="../../model/c1200/yolov11n-det-c1200/"
YOLOV12_MODEL="../../model/c1200/yolov12n-det-c1200/"

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

set_models_for_device() {
    case "$1" in
        A1000)
            [ -d "../../model/a1000/yolov10n-det-a1000" ] && YOLOV10_MODEL="../../model/a1000/yolov10n-det-a1000/"
            [ -d "../../model/a1000/yolov11n-det-a1000" ] && YOLOV11_MODEL="../../model/a1000/yolov11n-det-a1000/"
            [ -d "../../model/a1000/yolov12n-det-a1000" ] && YOLOV12_MODEL="../../model/a1000/yolov12n-det-a1000/"
            ;;
        C1200)
            [ -d "../../model/c1200/yolov10n-det-c1200" ] && YOLOV10_MODEL="../../model/c1200/yolov10n-det-c1200/"
            [ -d "../../model/c1200/yolov11n-det-c1200" ] && YOLOV11_MODEL="../../model/c1200/yolov11n-det-c1200/"
            [ -d "../../model/c1200/yolov12n-det-c1200" ] && YOLOV12_MODEL="../../model/c1200/yolov12n-det-c1200/"
            ;;
        A2000)
            [ -d "../../model/a2000/yolov10n-det-a2000" ] && YOLOV10_MODEL="../../model/a2000/yolov10n-det-a2000/"
            [ -d "../../model/a2000/yolov11n-det-a2000" ] && YOLOV11_MODEL="../../model/a2000/yolov11n-det-a2000/"
            [ -d "../../model/a2000/yolov12n-det-a2000" ] && YOLOV12_MODEL="../../model/a2000/yolov12n-det-a2000/"
            ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -A1000)
            DEVICE="A1000"
            shift
            ;;
        -C1200)
            DEVICE="C1200"
            shift
            ;;
        -A2000)
            DEVICE="A2000"
            shift
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-A1000 | -C1200 | -A2000]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DEVICE" ]; then
    DEVICE=$(detect_device_from_system || true)
fi

if [ -z "$DEVICE" ]; then
    DEVICE="C1200"
    echo "[Warning] Unable to detect device, fallback to C1200 models when available"
fi

set_models_for_device "$DEVICE"

mkdir -p ./output
benchmark_engine_finalize
# 使用 yolo-det：与系统 OpenCV 4.5 + libstdc++ 一致。
# yolo-det-c1200 依赖 OpenCV 4.9（*.so.409）与 libc++，常见 rootfs 上不可用。
APP="./prebuild/app/yolo-det"
if [ ! -x "$APP" ]; then
    APP="./prebuild/app/yolo-det-c1200"
fi

echo "[Info] Device: $DEVICE"
echo "============== Running YOLOv10n ($DEVICE) =============="
"$APP" -i ../../test_data/000000000139.jpg -m "$YOLOV10_MODEL" -o ./output/yolov10_inferresult.jpg -e "$ENGINE"

echo "============== Running YOLOv11n ($DEVICE) =============="
"$APP" -i ../../test_data/000000000139.jpg -m "$YOLOV11_MODEL" -o ./output/yolov11_inferresult.jpg -e "$ENGINE"

echo "============== Running YOLOv12n ($DEVICE) =============="
"$APP" -i ../../test_data/000000000139.jpg -m "$YOLOV12_MODEL" -o ./output/yolov12_inferresult.jpg -e "$ENGINE"