#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
ENGINE="${BST_BSNN_TARGET_ENGINE:-auto}"
# shellcheck source=/dev/null
. "$REPO_ROOT/utils/tools/lib/prebuild_engine.sh"

# LightStereo convenience run script
# Auto-detect executable, prepare inputs, run inference, save output & EPE.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR=""
INPUT_DIR="$SCRIPT_DIR/../../test_data"
OUTPUT_DIR="$SCRIPT_DIR/output"
DEVICE=""

LEFT_IMG="$INPUT_DIR/0326_left.png"
RIGHT_IMG="$INPUT_DIR/0326_right.png"
MASK_BIN="$INPUT_DIR/mask.bin"
GT_BIN="$INPUT_DIR/disp_gt.bin"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
Options:
  -A1000           Use A1000 model bundle
  -C1200           Use C1200 model bundle
  -A2000           Use A2000 model bundle
  -m <model_dir>    Model directory (default: $MODEL_DIR)
  -i <input_dir>    Inputs directory (default: $INPUT_DIR)
  -o <output_dir>   Output directory (default: $OUTPUT_DIR)
  -no-epe           Skip EPE computation even if GT present
  -no-mask          Ignore mask even if present
  --engine top|lite|auto BSNN engine (default: auto)
  -h                Show help

Input directory expectations:
  0326_left.png, 0326_right.png, mask.bin (uint8), disp_gt.bin (float32)
Bin files are assumed to be pre-converted; .npy not supported here.
EOF
}

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

set_model_for_device() {
  case "$1" in
    A1000)
      if [[ -d "$SCRIPT_DIR/../../model/a1000/lightstereo-a1000" ]]; then
        MODEL_DIR="$SCRIPT_DIR/../../model/a1000/lightstereo-a1000"
        return 0
      fi
      ;;
    C1200)
      if [[ -d "$SCRIPT_DIR/../../model/c1200/lightstereo-c1200" ]]; then
        MODEL_DIR="$SCRIPT_DIR/../../model/c1200/lightstereo-c1200"
        return 0
      fi
      ;;
    A2000)
      if [[ -d "$SCRIPT_DIR/main_a2000_OpenStereo-2/model" ]]; then
        MODEL_DIR="$SCRIPT_DIR/main_a2000_OpenStereo-2/model"
        return 0
      elif [[ -d "$SCRIPT_DIR/main_a2000_openstereo/model" ]]; then
        MODEL_DIR="$SCRIPT_DIR/main_a2000_openstereo/model"
        return 0
      fi
      ;;
  esac

  return 1
}

fallback_model_dir() {
  local candidate
  for candidate in \
    "$SCRIPT_DIR/../../model/c1200/lightstereo-c1200" \
    "$SCRIPT_DIR/../../model/a1000/lightstereo-a1000" \
    "$SCRIPT_DIR/main_a2000_OpenStereo-2/model" \
    "$SCRIPT_DIR/main_a2000_openstereo/model"; do
    if [[ -d "$candidate" ]]; then
      MODEL_DIR="$candidate"
      return 0
    fi
  done

  return 1
}

NO_EPE=0
NO_MASK=0
while [[ $# -gt 0 ]]; do
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
    -m) MODEL_DIR="$2"; shift 2;;
    -i) INPUT_DIR="$2"; shift 2;;
    -o) OUTPUT_DIR="$2"; shift 2;;
    -no-epe) NO_EPE=1; shift;;
    -no-mask) NO_MASK=1; shift;;
    --engine) ENGINE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$DEVICE" ]]; then
  DEVICE=$(detect_device_from_system || true)
  if [[ -n "$DEVICE" ]]; then
    set_model_for_device "$DEVICE" || true
  fi
fi

if [[ -z "$MODEL_DIR" ]]; then
  fallback_model_dir || true
fi

[[ -n "$MODEL_DIR" ]] || err "OpenStereo model directory not found"

mkdir -p "$OUTPUT_DIR"
benchmark_engine_finalize

# Resolve absolute paths after overrides
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

LEFT_IMG="$INPUT_DIR/0326_left.png"
RIGHT_IMG="$INPUT_DIR/0326_right.png"
MASK_BIN="$INPUT_DIR/mask.bin"
GT_BIN="$INPUT_DIR/disp_gt.bin"

err() { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN ] $*" >&2; }
info() { echo "[INFO ] $*"; }

[[ -f "$LEFT_IMG" ]] || err "Left image not found: $LEFT_IMG"
[[ -f "$RIGHT_IMG" ]] || err "Right image not found: $RIGHT_IMG"
[[ -d "$MODEL_DIR" ]] || err "Model directory not found: $MODEL_DIR"

# Validate presence of bin files if EPE/mask requested
if [[ $NO_MASK -eq 0 && ! -f "$MASK_BIN" ]]; then
  warn "Mask bin not found: $MASK_BIN (will skip mask)"
fi
if [[ $NO_EPE -eq 0 && ! -f "$GT_BIN" ]]; then
  warn "GT bin not found: $GT_BIN (will skip EPE)"
fi

# Locate executable (installed path preferred -> prebuild/app)
EXE=""
if [[ -x "$SCRIPT_DIR/prebuild/app/lightstereo" ]]; then
  EXE="$SCRIPT_DIR/prebuild/app/lightstereo"
else
  # fallback: search under build tree
  EXE_CANDIDATES=$(find "$SCRIPT_DIR/build" -maxdepth 2 -type f -name lightstereo 2>/dev/null || true)
  for c in $EXE_CANDIDATES; do
    if [[ -x "$c" ]]; then EXE="$c"; break; fi
  done
fi
[[ -n "$EXE" ]] || err "lightstereo executable not found. Build and install first."

info "Using executable: $EXE"
info "Detected device: ${DEVICE:-unknown}"
info "Model dir      : $MODEL_DIR"
info "Inputs dir     : $INPUT_DIR"
info "Output dir     : $OUTPUT_DIR"

RUN_MASK_ARG=""
if [[ $NO_MASK -eq 0 && -f "$MASK_BIN" ]]; then
  RUN_MASK_ARG="-mask $MASK_BIN"
elif [[ $NO_MASK -eq 0 ]]; then
  warn "Mask file not found; skipping mask application"
fi

RUN_GT_ARG=""
if [[ $NO_EPE -eq 0 && -f "$GT_BIN" ]]; then
  RUN_GT_ARG="-gt $GT_BIN"
elif [[ $NO_EPE -eq 0 ]]; then
  warn "GT file not found; skipping EPE"
fi

CMD=("$EXE" -left "$LEFT_IMG" -right "$RIGHT_IMG" -m "$MODEL_DIR" -o "$OUTPUT_DIR" -save 1 -e "$ENGINE")
[[ -n "$RUN_MASK_ARG" ]] && CMD+=($RUN_MASK_ARG)
[[ -n "$RUN_GT_ARG" ]] && CMD+=($RUN_GT_ARG)

info "Running: ${CMD[*]}"
"${CMD[@]}"

info "Done. Disparity results saved in $OUTPUT_DIR"
