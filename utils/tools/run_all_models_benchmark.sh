#!/usr/bin/env bash
# 仓库根目录下执行: ./scripts/run_all_models_benchmark.sh
# 参数原样传给 run_all_models_benchmark.py，例如:
#   ./scripts/run_all_models_benchmark.sh                 # 默认自动检测平台
#   ./scripts/run_all_models_benchmark.sh --device C1200  # 手动指定平台
#   ./scripts/run_all_models_benchmark.sh --include-extras

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT/utils/tools/run_all_models_benchmark.py" "$@"
