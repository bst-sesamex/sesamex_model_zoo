#!/bin/sh
set -e

# condlanenet 后处理栈消耗大，A1000 板端默认栈不足会 SIGSEGV
ulimit -s unlimited 2>/dev/null || ulimit -S -s 65536 2>/dev/null || true

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
    echo "[Warning] Unable to detect device, fallback to C1200 models"
fi

echo "[Info] Device: $DEVICE"
MODEL_DIR="../../model/c1200/condlanenet-c1200"
case "$DEVICE" in
    A1000)
        if [ -f ../../model/a1000/condlanenet-a1000/condlanenet.meta ]; then
            MODEL_DIR="../../model/a1000/condlanenet-a1000"
        fi
        ;;
esac
echo "============== Running CondLaneNet =============="
echo "[Info] Model directory: $MODEL_DIR"
benchmark_engine_finalize
./prebuild/app/condlanenet ../../test_data/0352_19.jpg "$MODEL_DIR" -e "$ENGINE"
