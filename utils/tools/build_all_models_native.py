#!/usr/bin/env python3
"""
遍历 samples/vision/ 下各 Deploy C++ 工程，在本机默认工具链下完成一次 Release 编译。

设计要点：
  - 与各工程 Deploy/code 一致，使用 build/；配置前会先清空 build/。
  - MODEL_ZOO_HOST_NATIVE=1 时，构建结束后会把可执行文件拷入对应 Deploy/prebuild/app，便于接着执行 run_demo / run_all_models_benchmark。
  - 子进程会移除常见交叉编译相关环境变量，降低误用 CMAKE_TOOLCHAIN_FILE / CC / CXX 的概率。

未纳入默认列表的工程：
  - yolov5/Deploy/board_deploy_with_opencv/deploy_code：随仓库自带的 libbsnn 依赖 liblwnn.so.0，与常见本机 liblwnn.so.2 不匹配，链接阶段会失败；请在提供匹配 LWNN 的运行环境或板端 SDK 下单独 cmake 编译。
  - models/Qwen2.5-1.5B-Instruct：无统一 C++ CMake 部署工程；train/deploy/cpp 等需 CUDA/TensorRT，不纳入本脚本。

用法:
  python3 scripts/build_all_models_native.py
  python3 scripts/build_all_models_native.py --dry-run
  python3 scripts/build_all_models_native.py -q
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = REPO_ROOT / "samples" / "vision"

# 与交叉/工具链相关的变量：从子环境剔除，避免“本地编译”误用交叉设置。
_ENV_KEYS_STRIP = (
    "CMAKE_TOOLCHAIN_FILE",
    "CROSS_COMPILE",
    "CC",
    "CXX",
    "CPP",
    "AR",
    "AS",
    "LD",
    "NM",
    "RANLIB",
    "STRIP",
    "OBJCOPY",
    "OBJDUMP",
    "PKG_CONFIG_SYSROOT_DIR",
    "PKG_CONFIG_LIBDIR",
    "CONFIG_SITE",
    "QMAKE_CC",
    "QMAKE_CXX",
)


def host_native_subprocess_env() -> Dict[str, str]:
    e = dict(os.environ)
    for k in _ENV_KEYS_STRIP:
        e.pop(k, None)
    # bsnnrt 0.13.0 is board-installed; do not point CMake at a missing libs/bsnn tree.
    e["BSNN_RUNTIME_ROOT"] = "/usr"
    e.pop("DEVICE", None)
    e["MODEL_ZOO_HOST_NATIVE"] = "1"
    return e


@dataclass
class BuildJob:
    label: str
    kind: str  # "build_sh" | "wrapper_build_sh"
    path: Path


def _run_build_sh(job: BuildJob, env: Dict[str, str], quiet: bool) -> None:
    sh = job.path / "build.sh"
    if not sh.is_file():
        raise FileNotFoundError(f"Missing build.sh: {sh}")
    
    cmd = ["bash", str(sh)]
            
    _run(cmd, cwd=job.path, env=env, quiet=quiet)


def _run(
    cmd: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    quiet: bool,
) -> None:
    if not quiet:
        print(f"[cmd] cwd={cwd} {' '.join(cmd)}", flush=True)
    subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        check=True,
    )


def discover_jobs() -> List[BuildJob]:
    jobs: List[BuildJob] = [
        BuildJob("yolov5", "build_sh", MODELS / "yolov5" / "runtime" / "cpp"),
        BuildJob("openstereo2", "build_sh", MODELS / "openstereo2" / "runtime" / "cpp"),
        BuildJob("condlanenet", "build_sh", MODELS / "condlanenet" / "runtime" / "cpp"),
        BuildJob("classification", "wrapper_build_sh", MODELS / "classification" / "runtime" / "cpp"),
        BuildJob("fcos", "wrapper_build_sh", MODELS / "fcos" / "runtime" / "cpp"),
        BuildJob("yolov8", "wrapper_build_sh", MODELS / "yolov8" / "runtime" / "cpp"),
        BuildJob("yolov12", "wrapper_build_sh", MODELS / "yolov12" / "runtime" / "cpp"),
        BuildJob("landmark", "build_sh", MODELS / "landmark" / "runtime" / "cpp"),
        BuildJob("reid", "build_sh", MODELS / "reid" / "runtime" / "cpp"),
        BuildJob("unet-seg", "build_sh", MODELS / "unet-seg" / "runtime" / "cpp"),
    ]
    return jobs


_RUNNERS: Dict[str, Callable[[BuildJob, Dict[str, str], bool], None]] = {
    "build_sh": _run_build_sh,
    "wrapper_build_sh": _run_build_sh,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="本机编译 models 下各 Deploy C++ 工程（清空并使用各 code/build）")
    p.add_argument("--dry-run", action="store_true", help="只列出将执行的工程，不编译")
    p.add_argument("-q", "--quiet", action="store_true", help="少打印 cmake 命令行")
    args = p.parse_args(list(argv) if argv is not None else None)

    jobs = discover_jobs()
    env_plain = host_native_subprocess_env()

    failed: List[str] = []
    for job in jobs:
        if not job.path.is_dir():
            failed.append(f"{job.label} (missing dir: {job.path})")
            continue
        runner = _RUNNERS.get(job.kind)
        if runner is None:
            failed.append(f"{job.label} (unknown kind {job.kind})")
            continue
        print(f"\n========== {job.label} ==========", flush=True)
        if args.dry_run:
            print(f"  kind={job.kind} path={job.path}", flush=True)
            continue
        try:
            runner(job, env_plain, args.quiet)
            print(f"[ok] {job.label}", flush=True)
        except subprocess.CalledProcessError as e:
            failed.append(f"{job.label} (exit {e.returncode})")
        except Exception as e:
            failed.append(f"{job.label} ({e})")

    print("\n========== summary ==========", flush=True)
    if failed:
        for line in failed:
            print(f"  FAIL: {line}", flush=True)
        return 1
    print("  All builds succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
