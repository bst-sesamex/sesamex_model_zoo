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

UNET_512_DIR="../../model/c1200/unet_512x512-c1200"
UNET_1024_DIR="../../model/c1200/unet_512x1024-c1200"
if [ "$DEVICE" = "A1000" ]; then
    [ -d "../../model/a1000/unet_512x512-a1000" ] && UNET_512_DIR="../../model/a1000/unet_512x512-a1000"
    [ -d "../../model/a1000/unet_512x1024-a1000" ] && UNET_1024_DIR="../../model/a1000/unet_512x1024-a1000"
fi

mkdir -p ./output
benchmark_engine_finalize

echo "[Info] Device: $DEVICE"
echo "============== Running Model of Input size 512x512 =============="
./prebuild/app/unet-seg -i ../../test_data/test.png -m "$UNET_512_DIR" -o ./output/inference_512x512.jpg -e "$ENGINE"

echo "============== Running Model of Input size 512x1024 =============="
./prebuild/app/unet-seg -i ../../test_data/test.png -m "$UNET_1024_DIR" -o ./output/inference_512x1024.jpg -e "$ENGINE"
