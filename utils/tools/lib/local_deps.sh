#!/usr/bin/env bash
# PyBSNN wheel installer for init_env.sh.
# The wheel IS committed to this repository under pybsnn/ (in-repo weights
# policy, v1.0.1). The former model-server download channel is retired.

PYBSNN_WHEEL_C1200="aura/pybsnn-0.2.0+0.13.0-cp310-cp310-linux_aarch64.whl"
PYBSNN_WHEEL_A1000="${PYBSNN_WHEEL_A1000:-}"  # set when an a1000 wheel is published

local_deps_pybsnn_platform() {
    local info="" f
    for f in /proc/device-tree/compatible /proc/device-tree/model /etc/os-release; do
        if [ -r "$f" ]; then
            info+="$(tr '\000' '\n' <"$f" 2>/dev/null)"$'\n'
        fi
    done
    info=$(printf '%s' "$info" | tr '[:upper:]' '[:lower:]')
    case "$info" in
        *a1000*) echo "a1000" ;;
        *) echo "c1200" ;;
    esac
}

local_deps_pick_pybsnn_wheel_name() {
    local platform="$1"
    case "$platform" in
        a1000) echo "${PYBSNN_WHEEL_A1000}" ;;
        *)     echo "${PYBSNN_WHEEL_C1200}" ;;
    esac
}

local_deps_install_pybsnn() {
    local platform wheel_name wheel_path
    local root="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

    if python3 -m pip show pybsnn >/dev/null 2>&1; then
        echo "[Info] PyBSNN already installed, skipping."
        return 0
    fi

    platform="$(local_deps_pybsnn_platform)"
    wheel_name="$(local_deps_pick_pybsnn_wheel_name "$platform")"
    wheel_path="${root}/pybsnn/${wheel_name}"
    if [ -z "$wheel_name" ] || [ ! -f "$wheel_path" ]; then
        echo "[ERROR] No PyBSNN wheel available for platform '$platform'." >&2
        echo "        Expected: ${wheel_path}" >&2
        echo "        This repository ships a c1200 (aarch64) wheel; an a1000 wheel" >&2
        echo "        is not published yet (place it at pybsnn/ and set PYBSNN_WHEEL_A1000)." >&2
        return 1
    fi
    echo "[Info] Installing PyBSNN (${platform}) from repo: ${wheel_path}"
    pip3 install --no-deps --force-reinstall "$wheel_path"
}

local_deps_export_env() {
    # Board images ship libbsnnrt under /usr; allow override for local SDK layouts.
    export BSNN_RUNTIME_ROOT="${BSNN_RUNTIME_ROOT:-/usr}"
}
