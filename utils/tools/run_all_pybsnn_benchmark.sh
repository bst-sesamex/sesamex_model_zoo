#!/usr/bin/env bash
# 仓库根目录下执行: ./scripts/run_all_pybsnn_benchmark.sh
# 参数原样传给 run_all_pybsnn_benchmark.py，例如:
#   ./scripts/run_all_pybsnn_benchmark.sh                 # 默认自动检测平台
#   ./scripts/run_all_pybsnn_benchmark.sh --quiet        # 只打汇总表
#   ./scripts/run_all_pybsnn_benchmark.sh --dry-run      # 只列出将执行的命令

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT/utils/tools/run_all_pybsnn_benchmark.py" "$@"
