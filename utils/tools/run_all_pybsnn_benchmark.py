#!/usr/bin/env python3
"""
一键遍历 models/ 下各工程的 Deploy/pybsnn*.py 示例脚本，
运行推理并汇总 bsnn model inference time (ms) 与 FPS。

脚本会解析 demo 输出中的:
  -> bsnn model inference time : <float> ms

用法:
  python3 scripts/run_all_pybsnn_benchmark.py
  python3 scripts/run_all_pybsnn_benchmark.py --device C1200
  python3 scripts/run_all_pybsnn_benchmark.py --quiet
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "utils" / "py_utils" / "deploy"))
from pybsnn_paths import env_with_bsnn_runtime  # noqa: E402

# Regex to parse the inference time we just added to the demos
RE_BSNN_INFER_MS = re.compile(
    r"-> bsnn model inference time\s*:\s*([\d.]+)\s*ms", re.IGNORECASE
)

def _append_file_content(chunks: List[str], path: Path) -> None:
    try:
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "\n"))
    except OSError:
        pass

def detect_device_from_system() -> Optional[str]:
    """尝试检测当前运行平台 (A1000/C1200/A2000)。"""
    chunks: List[str] = []
    for path_str in (
        "/proc/device-tree/compatible",
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/compatible",
        "/sys/firmware/devicetree/base/model",
        "/etc/os-release",
    ):
        _append_file_content(chunks, Path(path_str))

    try:
        uname = subprocess.run(
            ["uname", "-a"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if uname:
            chunks.append(uname)
    except OSError:
        pass

    system_info = "\n".join(chunks).lower()
    if "a2000" in system_info: return "A2000"
    if "a1000" in system_info: return "A1000"
    if "c1200" in system_info: return "C1200"
    return None

def discover_pybsnn_scripts(models_root: Path) -> List[Path]:
    """查找所有 Deploy/ 目录下的 pybsnn*.py 脚本 (支持 python 子目录)。"""
    candidates = list(models_root.glob("**/runtime/python/pybsnn*.py"))
    # 同时兼容可能直接在 Deploy 目录下的脚本
    candidates.extend(list(models_root.glob("**/Deploy/pybsnn*.py")))
    # 过滤掉一些可能存在的备份或无关文件
    def is_valid(p: Path):
        return p.is_file() and not p.name.startswith(".")
    
    return sorted([p for p in candidates if is_valid(p)], key=lambda x: str(x))

def run_script(
    cwd: Path,
    argv: List[str],
    env: dict,
    stream_output: bool = True,
) -> Tuple[int, str, float]:
    """执行子进程并捕获输出。"""
    t0 = time.perf_counter()
    if stream_output:
        p = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        chunks: List[str] = []
        assert p.stdout is not None
        for line in p.stdout:
            chunks.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        p.wait()
        elapsed = time.perf_counter() - t0
        return p.returncode, "".join(chunks), elapsed

    p = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - t0
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out, elapsed

def main():
    parser = argparse.ArgumentParser(description="Run all pybsnn demo scripts and benchmark inference time.")
    parser.add_argument("--models-root", type=Path, help="Path to samples/vision directory.")
    parser.add_argument("--device", choices=("auto", "none", "A1000", "C1200", "A2000"), default="auto")
    parser.add_argument("--quiet", action="store_true", help="Only show the summary table.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned commands.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    models_root = args.models_root or (repo_root / "samples" / "vision")
    if not models_root.is_dir():
        print(f"[ERROR] models root not found: {models_root}", file=sys.stderr)
        return 1

    selected_device = detect_device_from_system() if args.device == "auto" else (None if args.device == "none" else args.device)

    scripts = discover_pybsnn_scripts(models_root)
    if not scripts:
        print(f"[WARN] No pybsnn*.py scripts found under {models_root}", file=sys.stderr)
        return 0

    print(f"[INFO] Found {len(scripts)} pybsnn demo scripts.")
    if selected_device:
        print(f"[INFO] Target device: {selected_device}")

    if args.dry_run:
        print("\n=== Dry run: planned commands ===\n")
        for script_path in scripts:
            rel = script_path.relative_to(models_root)
            print(f"[DRY-RUN] cd {script_path.parent} && {sys.executable} {script_path.name}")
        print(f"\n{len(scripts)} command(s) planned.")
        return 0

    summary = []
    env = env_with_bsnn_runtime(repo_root, os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    if selected_device:
        env["BST_BENCHMARK_DEVICE"] = selected_device

    for script_path in scripts:
        rel = script_path.relative_to(models_root)
        cwd = script_path.parent
        # BSNN Python demos usually don't take -device flags directly but we could inject if needed.
        # For now, just run standard python.
        argv = [sys.executable, script_path.name]
        
        if not args.quiet:
            print(f"\n{'='*80}\nRUNNING: {rel}\n{'='*80}\n", flush=True)
        
        code, text, wall = run_script(cwd, argv, env, stream_output=not args.quiet)
        
        # Parse inference time
        matches = RE_BSNN_INFER_MS.findall(text)
        infer_ms = float(matches[-1]) if matches else None
        fps = 1000.0 / infer_ms if (infer_ms and infer_ms > 0) else None
        
        status = "OK" if code == 0 else f"FAIL({code})"
        if code == 0 and not matches:
            status = "FAIL(no timing)"
        if status.startswith("FAIL") and args.quiet:
            tail = "\n".join(text.strip().splitlines()[-12:])
            if tail:
                print(f"--- log tail: {rel} ---\n{tail}\n", file=sys.stderr)

        summary.append({
            "script": str(rel),
            "infer_ms": infer_ms,
            "fps": fps,
            "wall_s": wall,
            "status": status
        })

    # Print summary table
    print("\n" + "="*110)
    print(f"{'PyBSNN Demo Script':<60} | {'Status':<10} | {'Infer (ms)':>10} | {'FPS':>8} | {'Wall (s)':>8}")
    print("-"*110)
    for row in summary:
        ms_str = f"{row['infer_ms']:.2f}" if row['infer_ms'] is not None else "N/A"
        fps_str = f"{row['fps']:.2f}" if row['fps'] is not None else "N/A"
        print(f"{row['script']:<60} | {row['status']:<10} | {ms_str:>10} | {fps_str:>8} | {row['wall_s']:>8.2f}")
    print("="*110)

    fail_count = sum(1 for row in summary if str(row["status"]).startswith("FAIL"))
    if fail_count:
        print(f"Finished with {fail_count} failed job(s).", file=sys.stderr)
        return 1
    print("All discovered pybsnn demos completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
