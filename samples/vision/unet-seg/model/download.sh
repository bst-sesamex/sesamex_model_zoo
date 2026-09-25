#!/usr/bin/env bash
# Verify the in-repo quantized unet-seg model package(s).
# Since v1.0.1 the model files are COMMITTED to this repository under
# model/<platform>/<variant>-<platform>/ — no download is required.
# This script only checks completeness and reports what is missing.
#
# Usage:
#   ./download.sh                 # check all variants for c1200
#   ./download.sh unet_512x512          # check one variant
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_one() {
  local platform="$1" variant="$2"; shift 2
  local dir="${SCRIPT_DIR}/${platform}/${variant}-${platform}"
  local missing=0 f
  if [ ! -d "${dir}" ]; then
    echo "[MISS] ${variant}-${platform}: directory not present"
    return 1
  fi
  for f in "$@"; do
    if [ -s "${dir}/${f}" ]; then
      echo "[ok  ] ${variant}-${platform}/${f}"
    else
      echo "[MISS] ${variant}-${platform}/${f}"
      missing=1
    fi
  done
  return $missing
}

FAIL=0
WANT_VARIANTS=("$@")
KNOWN_VARIANTS=()
check_variant() {
  local variant="$1"; shift
  KNOWN_VARIANTS+=("$variant")
  if [ "${#WANT_VARIANTS[@]}" -gt 0 ]; then
    local keep=0 w
    for w in "${WANT_VARIANTS[@]}"; do [ "$w" = "$variant" ] && keep=1; done
    [ "$keep" -eq 1 ] || return 0
  fi
  check_one c1200 "$variant" "$@" || FAIL=1
}

check_variant unet_512x512 "weights.bin" "reid.Enrsd.bin" "reid.meta" "reid_stripped.lib"

check_variant unet_512x1024 "weights.bin" "reid.Enrsd.bin" "reid.meta" "reid_stripped.lib"

for wanted in "${WANT_VARIANTS[@]}"; do
  found=0
  for known in "${KNOWN_VARIANTS[@]}"; do
    if [ "$wanted" = "$known" ]; then found=1; break; fi
  done
  if [ "$found" -eq 0 ]; then
    echo "[ERROR] Unknown variant: $wanted" >&2
    echo "Available: ${KNOWN_VARIANTS[*]}" >&2
    exit 2
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "[WARN] Some files are missing (partial package; see model/README.md)."
  exit 1
fi
echo ""
echo "All checked model files are present under $SCRIPT_DIR/c1200/."
