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

MOBILE_DIR="../../model/c1200/mobilenetv2-c1200"
OSNET_DIR="../../model/c1200/osnet-c1200"
RESNET_DIR="../../model/c1200/resnet50-c1200"
if [ "$DEVICE" = "A1000" ]; then
    [ -d "../../model/a1000/mobilenetv2-a1000" ] && MOBILE_DIR="../../model/a1000/mobilenetv2-a1000"
    [ -d "../../model/a1000/osnet-a1000" ] && OSNET_DIR="../../model/a1000/osnet-a1000"
    [ -d "../../model/a1000/resnet50-a1000" ] && RESNET_DIR="../../model/a1000/resnet50-a1000"
fi

mkdir -p ./output
benchmark_engine_finalize

echo "[Info] Device: $DEVICE"
echo "============== Running Mobilenetv2 True =============="
./prebuild/app/reid-pair-match -i1 ../../test_data/-1_c1s1_001351_04.jpg -i2 ../../test_data/-1_c1s1_001376_05.jpg \
    -m "$MOBILE_DIR" --metric cosine --threshold 0.3193 -e "$ENGINE"

echo "============== Running Mobilenetv2 False =============="
./prebuild/app/reid-pair-match -i1 ../../test_data/-1_c1s1_001351_04.jpg -i2 ../../test_data/-1_c1s1_000401_03.jpg \
    -m "$MOBILE_DIR" --metric cosine --threshold 0.3193 -e "$ENGINE"

echo "============== Running Osnet True =============="
./prebuild/app/reid-pair-match -i1 ../../test_data/-1_c1s1_001351_04.jpg -i2 ../../test_data/-1_c1s1_001376_05.jpg \
    -m "$OSNET_DIR" --metric cosine --threshold 0.3677 -e "$ENGINE"

echo "============== Running Osnet False =============="
./prebuild/app/reid-pair-match -i1 ../../test_data/-1_c1s1_001351_04.jpg -i2 ../../test_data/-1_c1s1_000401_03.jpg \
    -m "$OSNET_DIR" --metric cosine --threshold 0.3677 -e "$ENGINE"

echo "============== Running Resnet50 True =============="
./prebuild/app/reid-pair-match -i1 ../../test_data/-1_c1s1_001351_04.jpg -i2 ../../test_data/-1_c1s1_001376_05.jpg \
    -m "$RESNET_DIR" --metric cosine --threshold 0.2576 -e "$ENGINE"

echo "============== Running Resnet False =============="
./prebuild/app/reid-pair-match -i1 ../../test_data/-1_c1s1_001351_04.jpg -i2 ../../test_data/-1_c1s1_000401_03.jpg \
    -m "$RESNET_DIR" --metric cosine --threshold 0.2576 -e "$ENGINE"
