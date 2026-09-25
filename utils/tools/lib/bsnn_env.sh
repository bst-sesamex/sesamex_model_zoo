#!/usr/bin/env bash
# Shared helpers: repo root, default BSNN runtime path, platform auto-detection.

bsnn_find_repo_root() {
  local dir="$1"
  while [[ -n "${dir}" && "${dir}" != "/" ]]; do
    if [[ -f "${dir}/VERSION" && -d "${dir}/samples" ]]; then
      echo "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  return 1
}

bsnn_default_runtime_root() {
  local start="${1:-$(pwd)}"
  local root=""
  if root="$(bsnn_find_repo_root "${start}")"; then
    echo "${root}/libs/bsnn"
  fi
}

bsnn_resolve_runtime_root() {
  if [[ -n "${BSNN_RUNTIME_ROOT:-}" ]]; then
    echo "${BSNN_RUNTIME_ROOT}"
    return 0
  fi
  bsnn_default_runtime_root "${1:-$(pwd)}"
}

bsnn_detect_device_name() {
  local chunks="" path uname_out
  for path in \
    /proc/device-tree/compatible \
    /proc/device-tree/model \
    /sys/firmware/devicetree/base/compatible \
    /sys/firmware/devicetree/base/model \
    /etc/os-release; do
    if [[ -r "${path}" ]]; then
      chunks+="$(tr '\0' ' ' < "${path}" 2>/dev/null || cat "${path}" 2>/dev/null)"$'\n'
    fi
  done
  if command -v uname >/dev/null 2>&1; then
    uname_out="$(uname -a 2>/dev/null || true)"
    [[ -n "${uname_out}" ]] && chunks+="${uname_out}"$'\n'
  fi
  chunks="$(echo "${chunks}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${chunks}" == *"a2000"* ]]; then
    echo "A2000"
  elif [[ "${chunks}" == *"a1000"* ]]; then
    echo "A1000B"
  elif [[ "${chunks}" == *"c1200"* ]]; then
    echo "C1200"
  fi
}

bsnn_pick_device_name() {
  local runtime_root="$1"
  local detected=""
  detected="$(bsnn_detect_device_name || true)"
  if [[ -n "${detected}" && -d "${runtime_root}/${detected}_runtime/usr" ]]; then
    echo "${detected}"
    return 0
  fi
  for candidate in A1000B C1200 A2000; do
    if [[ -d "${runtime_root}/${candidate}_runtime/usr" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo ""
}

bsnn_device_to_flag() {
  case "$1" in
    A1000B) echo "-A1000" ;;
    C1200)  echo "-C1200" ;;
    A2000)  echo "-A2000" ;;
    *) echo "" ;;
  esac
}
