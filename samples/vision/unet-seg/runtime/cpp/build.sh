#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for unified build logic.
# - Prefer repo-local common script under utils/tools/lib
# - Fall back to per-sample tools/ copy for standalone archives

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the repository root. Version-control metadata is NOT required:
# a directory is treated as the repo root when it carries the unified build
# script itself, the repo markers (VERSION + samples/), or a .git directory.
# This keeps zip/tar checkouts (no .git) working. The marker set mirrors
# bsnn_find_repo_root() in utils/tools/lib/bsnn_env.sh.
find_repo_root() {
	local dir="$1"
	while [[ -n "${dir}" && "${dir}" != "/" ]]; do
		if [[ -f "${dir}/utils/tools/lib/common_build.sh" ]] \
			|| [[ -f "${dir}/VERSION" && -d "${dir}/samples" ]] \
			|| [[ -d "${dir}/.git" ]]; then
			echo "${dir}"
			return 0
		fi
		dir="$(dirname "${dir}")"
	done
	return 1
}

REPO_ROOT="$(find_repo_root "${SCRIPT_DIR}" || true)"

COMMON_BUILD_SH=""
if [[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/utils/tools/lib/common_build.sh" ]]; then
	COMMON_BUILD_SH="${REPO_ROOT}/utils/tools/lib/common_build.sh"
elif [[ -f "${SCRIPT_DIR}/tools/build/common_build.sh" ]]; then
	COMMON_BUILD_SH="${SCRIPT_DIR}/tools/build/common_build.sh"
else
	echo "[Error] common_build.sh not found." >&2
	echo "        Searched:" >&2
	echo "          - ${REPO_ROOT:-<no-repo-root>}/utils/tools/lib/common_build.sh" >&2
	echo "          - ${SCRIPT_DIR}/tools/build/common_build.sh" >&2
	exit 2
fi

echo "[Info] Using common build script: ${COMMON_BUILD_SH}"

exec bash "${COMMON_BUILD_SH}" --src "${SCRIPT_DIR}" "$@"
