#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [-A1000 | -C1200 | -A2000] [--clean] [-DDEVICE_RUNTIME_DIR=...] [-D <path>]

Options:
  -A1000            Build for DEVICE=A1000B (maps to A1000B_runtime)
  -C1200            Build for DEVICE=C1200   (maps to C1200_runtime)
  -A2000            Build for DEVICE=A2000   (maps to A2000_runtime)
  --clean           Remove build directory before configuring (默认已启用)
  -DDEVICE_RUNTIME_DIR=PATH  Explicit runtime directory override（此时无需 DEVICE 映射）
  -D PATH         Same as -DDEVICE_RUNTIME_DIR=PATH，便于简写

Notes:
  - 默认 DEVICE=C1200；如需切换可显式传入设备参数
  - 本脚本在 code 目录下使用 ./build；默认在配置前清空该目录
  - MODEL_ZOO_HOST_NATIVE=1 时清空 ./build、DEVICE 置空，用本机工具链编译
  - *_runtime 目录默认位于仓库 libs/bsnn 下，可通过环境变量 BSNN_RUNTIME_ROOT 覆盖
  - 未指定设备时自动识别板卡平台，并选用 libs/bsnn 下已有的 *_runtime
EOF
}

# Resolve script directory and move there to make relative paths stable
DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${DEFAULT_SCRIPT_DIR}"
BSNN_ENV_SH="${DEFAULT_SCRIPT_DIR}/bsnn_env.sh"
if [[ -f "${BSNN_ENV_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${BSNN_ENV_SH}"
fi

# Handle --src argument from wrapper
if [[ "${1:-}" == "--src" ]]; then
  SCRIPT_DIR="$(cd "$2" && pwd)"
  shift 2
fi

cd "$SCRIPT_DIR"

if [[ "${MODEL_ZOO_HOST_NATIVE:-}" == "1" ]]; then
  echo "[Info] MODEL_ZOO_HOST_NATIVE=1: DEVICE cleared, using ./build (cleaned before configure)"
fi

# Default to no DEVICE so wrapper can call without parameters.
DEVICE=""
# Always clean by default; --clean is accepted but redundant
CLEAN=1
EXTRA_CMAKE_ARGS=()
CUSTOM_DEVICE_RUNTIME_DIR=""

# 计算 runtime 根目录（未设置时默认 <repo>/libs/bsnn）
RUNTIME_ROOT=""
if declare -F bsnn_resolve_runtime_root >/dev/null 2>&1; then
  RUNTIME_ROOT="$(bsnn_resolve_runtime_root "${SCRIPT_DIR}")"
fi
if [[ -z "${RUNTIME_ROOT}" ]]; then
  if [[ -n "${BSNN_RUNTIME_ROOT:-}" ]]; then
    RUNTIME_ROOT="${BSNN_RUNTIME_ROOT}"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -A1000) DEVICE="A1000B" ;;
    -C1200) DEVICE="C1200" ;;
    -A2000) DEVICE="A2000" ;;
    --clean) CLEAN=1 ;;
    -DDEVICE_RUNTIME_DIR=*)
      CUSTOM_DEVICE_RUNTIME_DIR="${1#-DDEVICE_RUNTIME_DIR=}"
      EXTRA_CMAKE_ARGS+=("-DDEVICE_RUNTIME_DIR=${CUSTOM_DEVICE_RUNTIME_DIR}")
      ;;
    -D)
      if [[ $# -lt 2 ]]; then
        echo "[Error] -D 需要紧跟一个路径参数" >&2
        exit 1
      fi
      shift
      CUSTOM_DEVICE_RUNTIME_DIR="$1"
      EXTRA_CMAKE_ARGS+=("-DDEVICE_RUNTIME_DIR=${CUSTOM_DEVICE_RUNTIME_DIR}")
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[Error] Unknown option: $1"; usage; exit 1 ;;
  esac
    shift
done

if [[ "${MODEL_ZOO_HOST_NATIVE:-}" == "1" ]]; then
  DEVICE=""
fi

# 未显式指定设备时，按板卡 / libs/bsnn 下已有 runtime 自动选择
if [[ -z "${DEVICE}" && "${MODEL_ZOO_HOST_NATIVE:-}" != "1" && -n "${RUNTIME_ROOT}" ]]; then
  if declare -F bsnn_pick_device_name >/dev/null 2>&1; then
    AUTO_DEVICE="$(bsnn_pick_device_name "${RUNTIME_ROOT}")"
    if [[ -n "${AUTO_DEVICE}" ]]; then
      DEVICE="${AUTO_DEVICE}"
      echo "[Info] Auto-selected DEVICE=${DEVICE} (runtime: ${RUNTIME_ROOT}/${AUTO_DEVICE}_runtime)"
    fi
  fi
fi

echo "[Info] DEVICE=${DEVICE}"
echo "[Info] BSNN_RUNTIME_ROOT=${RUNTIME_ROOT}"

# Resolve DEVICE_RUNTIME_DIR:
# 1) If user provided CUSTOM_DEVICE_RUNTIME_DIR via -D, use it.
# 2) If DEVICE is empty, fall back to the source/script directory.
# 3) Otherwise map DEVICE -> runtime name under RUNTIME_ROOT.
if [[ -n "${CUSTOM_DEVICE_RUNTIME_DIR}" ]]; then
  DEVICE_RUNTIME_DIR="${CUSTOM_DEVICE_RUNTIME_DIR}"
else
  if [[ -z "${DEVICE}" ]]; then
    if [[ -n "${RUNTIME_ROOT}" && -d "${RUNTIME_ROOT}/C1200_runtime" ]]; then
      echo "[Info] No DEVICE specified but RUNTIME_ROOT/C1200_runtime found; using it as default for aarch64 native build"
      DEVICE_RUNTIME_DIR="${RUNTIME_ROOT}/C1200_runtime"
      EXTRA_CMAKE_ARGS+=("-DDEVICE_RUNTIME_DIR=${DEVICE_RUNTIME_DIR}")
    else
      echo "[Info] No DEVICE specified; using source directory as DEVICE_RUNTIME_DIR"
      DEVICE_RUNTIME_DIR="${SCRIPT_DIR}"
    fi
  else
    case "${DEVICE}" in
      A1000B) DEVICE_RUNTIME_NAME="A1000B_runtime" ;;
      C1200)  DEVICE_RUNTIME_NAME="C1200_runtime" ;;
      A2000)  DEVICE_RUNTIME_NAME="A2000_runtime" ;;
      *) echo "[Error] Unsupported DEVICE=${DEVICE}"; exit 1 ;;
    esac
    if [[ -n "${RUNTIME_ROOT}" ]]; then
      DEVICE_RUNTIME_DIR="${RUNTIME_ROOT}/${DEVICE_RUNTIME_NAME}"
    else
      # No runtime root configured; fallback to source directory to allow
      # building without external runtimes.
      echo "[Warning] BSNN_RUNTIME_ROOT not set; falling back to source directory for runtime files"
      DEVICE_RUNTIME_DIR="${SCRIPT_DIR}"
    fi
  fi
fi

# If the resolved runtime dir doesn't exist, warn and fall back to source dir
if [[ ! -d "${DEVICE_RUNTIME_DIR}" ]]; then
  echo "[Warning] Runtime directory not found: ${DEVICE_RUNTIME_DIR}" >&2
  echo "         Falling back to source directory: ${SCRIPT_DIR}" >&2
  DEVICE_RUNTIME_DIR="${SCRIPT_DIR}"
fi

echo "[Info] Runtime directory: ${DEVICE_RUNTIME_DIR}"

BUILD_DIR="${SCRIPT_DIR}/build"
if [[ ${CLEAN} -eq 1 && -d "${BUILD_DIR}" ]]; then
  echo "[Info] Removing build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

echo "[Step] Configuring with CMake"
CMAKE_CMD=(
  cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}"
  -DDEVICE="${DEVICE}"
  -DCMAKE_BUILD_TYPE=Release
)
CMAKE_CMD+=("${EXTRA_CMAKE_ARGS[@]}")
if [[ -n "${RUNTIME_ROOT}" ]]; then
  CMAKE_CMD+=( -DBSNN_RUNTIME_ROOT="${RUNTIME_ROOT}" )
fi
"${CMAKE_CMD[@]}"

echo "[Step] Building"
JOBS=4
if command -v nproc >/dev/null 2>&1; then JOBS=$(nproc); fi
if [[ "$OSTYPE" == darwin* ]] && command -v sysctl >/dev/null 2>&1; then JOBS=$(sysctl -n hw.ncpu); fi
cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo "[Step] Installing built binary to: ${SCRIPT_DIR}/prebuild/app"
APP_DIR="${SCRIPT_DIR}/prebuild/app"   # <sample>/runtime/cpp/prebuild/app
mkdir -p "${APP_DIR}"
APP_DIR="$(cd "${APP_DIR}" && pwd)"
if [[ "${MODEL_ZOO_HOST_NATIVE:-}" == "1" ]]; then
  echo "[Info] host-native build (MODEL_ZOO_HOST_NATIVE=1)"
fi
BIN=""
if command -v find >/dev/null 2>&1; then
  BIN=$(find "${BUILD_DIR}" -maxdepth 1 -type f -executable \
      ! -name "CMake*" \
      ! -name "Makefile" \
      ! -name "*.cmake" \
      ! -name "*.txt" \
      ! -name "*.sh" \
      ! -name "*.bin" \
      -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi
if [[ -z "${BIN}" || ! -f "${BIN}" ]]; then
  echo "[Error] Failed to locate built executable under: ${BUILD_DIR}" >&2
  exit 2
fi
cp -f "${BIN}" "${APP_DIR}/"
chmod +x "${APP_DIR}/$(basename "${BIN}")" || true
echo "[Done] Installed: ${APP_DIR}/$(basename "${BIN}")"

echo "[Done] Build finished: ${BUILD_DIR}"
