#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
ENGINE="${BST_BSNN_TARGET_ENGINE:-auto}"
# shellcheck source=/dev/null
. "$REPO_ROOT/utils/tools/lib/prebuild_engine.sh"

DEVICE=""

append_file_content() {
    local file_path="$1"
    if [[ -r "$file_path" ]]; then
        local file_content
        file_content=$(tr '\000' '\n' < "$file_path" 2>/dev/null)
        if [[ -n "$file_content" ]]; then
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

    local uname_info
    uname_info=$(uname -a 2>/dev/null)
    if [[ -n "$uname_info" ]]; then
        SYSTEM_INFO="$SYSTEM_INFO
$uname_info"
    fi

    local system_info_lc
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

while [[ $# -gt 0 ]]; do
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

if [[ -z "$DEVICE" ]]; then
    DEVICE=$(detect_device_from_system || true)
fi

if [[ -z "$DEVICE" ]]; then
    DEVICE="C1200"
    echo "[Warning] Unable to detect device, fallback to bundled C1200 YOLOv8 models"
elif [[ "$DEVICE" != "C1200" ]]; then
    echo "[Warning] Detected device $DEVICE, but this prebuild currently only bundles C1200 YOLOv8 models"
fi

mkdir -p ./output
benchmark_engine_finalize

echo "[Info] Device: $DEVICE"
DET_MODEL="../../model/c1200/yolov8n-det-c1200/"
SEG_MODEL="../../model/c1200/yolov8n-seg-c1200/"
POSE_MODEL="../../model/c1200/yolov8n-pose-c1200/"
if [ "$DEVICE" = "A1000" ]; then
    [ -d "../../model/a1000/yolov8n-det-a1000" ] && DET_MODEL="../../model/a1000/yolov8n-det-a1000/"
    [ -d "../../model/a1000/yolov8n-seg-a1000" ] && SEG_MODEL="../../model/a1000/yolov8n-seg-a1000/"
    [ -d "../../model/a1000/yolov8n-pose-a1000" ] && POSE_MODEL="../../model/a1000/yolov8n-pose-a1000/"
fi
echo "============== Running YOLOv8-Det =============="
./prebuild/app/yolov8 -i ../../test_data/bus.jpg -m "$DET_MODEL" -o output/bus_det.jpg -t det -e "$ENGINE"

echo "============== Running YOLOv8-Seg =============="
./prebuild/app/yolov8 -i ../../test_data/bus.jpg -m "$SEG_MODEL" -o output/bus_seg.jpg -t seg -e "$ENGINE"

echo "============== Running YOLOv8-Pose =============="
./prebuild/app/yolov8 -i ../../test_data/bus.jpg -m "$POSE_MODEL" -o output/bus_pose.jpg -t pose -e "$ENGINE"
