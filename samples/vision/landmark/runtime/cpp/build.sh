#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
	local dir="$1"
	while [[ -n "${dir}" && "${dir}" != "/" ]]; do
		if [[ -d "${dir}/.git" ]]; then
			echo "${dir}"
			return 0
		fi
		dir="$(dirname "${dir}")"
	done
	return 1
}

REPO_ROOT=""
if REPO_ROOT="$(find_repo_root "${SCRIPT_DIR}")"; then
	:
else
	REPO_ROOT=""
fi

COMMON_BUILD_SH=""
if [[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/utils/tools/lib/common_build.sh" ]]; then
	COMMON_BUILD_SH="${REPO_ROOT}/utils/tools/lib/common_build.sh"
elif [[ -f "${SCRIPT_DIR}/tools/build/common_build.sh" ]]; then
	COMMON_BUILD_SH="${SCRIPT_DIR}/tools/build/common_build.sh"
else
	echo "[Error] common_build.sh not found." >&2
	exit 2
fi

echo "[Info] Using common build script: ${COMMON_BUILD_SH}"
exec "${COMMON_BUILD_SH}" --src "${SCRIPT_DIR}" "$@"


