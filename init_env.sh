#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/utils/tools/lib/local_deps.sh"

echo "--- Initializing Model Zoo environment ---"

local_deps_export_env "$ROOT_DIR"

ARCH=$(uname -m)
echo "Host arch: ${ARCH}"
if [ "$ARCH" = "aarch64" ]; then
    local_deps_install_pybsnn "$ROOT_DIR"
else
    echo "Skip PyBSNN install (aarch64 only). Run init_env.sh on the target board."
fi

echo "--- Environment ready ---"
echo "BSNN SDK runtime: ${BSNN_RUNTIME_ROOT:-/usr}"
echo "Next step: cd samples/vision/<model>/runtime/python && python3 pybsnn_demo.py"
