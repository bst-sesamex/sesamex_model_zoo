#!/bin/sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
ENGINE="${BST_BSNN_TARGET_ENGINE:-auto}"
# shellcheck source=/dev/null
. "$REPO_ROOT/utils/tools/lib/prebuild_engine.sh"

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
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-A1000 | -C1200] [--engine top|lite]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$DEVICE" ]; then
    DEVICE=$(detect_device_from_system || true)
fi

if [ -z "$DEVICE" ]; then
    DEVICE="C1200"
    echo "[Warning] Unable to detect device, fallback to C1200 models"
fi

SIZE56_DIR="../../model/c1200/size56-c1200"
SIZE224_DIR="../../model/c1200/size224-c1200"
case "$DEVICE" in
    A1000)
        [ -d "../../model/a1000/size56-a1000" ] && SIZE56_DIR="../../model/a1000/size56-a1000"
        [ -d "../../model/a1000/size224-a1000" ] && SIZE224_DIR="../../model/a1000/size224-a1000"
        ;;
esac

mkdir -p ./output
benchmark_engine_finalize

echo "[Info] Device: $DEVICE"
echo "============== Running Model of Input size 56 =============="
./prebuild/app/landmark-infer -i ../../test_data/crop_bgr_56.png -m "$SIZE56_DIR" -o ./output/inference_56.jpg -e "$ENGINE"

echo "============== Running Model of Input size 224 =============="
./prebuild/app/landmark-infer -i ../../test_data/crop_bgr_224.png -m "$SIZE224_DIR" -o ./output/inference_224.jpg -e "$ENGINE"
