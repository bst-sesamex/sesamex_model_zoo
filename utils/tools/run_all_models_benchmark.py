#!/usr/bin/env python3
"""
一键遍历各 sample 的 `runtime/cpp/run_demo.sh`，运行推理并汇总帧率与耗时。

BST 端侧 demo 可执行文件通常会打印:
  -> auto/top/lite engine inference FPS: <float>
  -> bsnn model inference time : <float> ms (板端单调时钟，预热后同步调用)

用法:
  python3 scripts/run_all_models_benchmark.py
  python3 scripts/run_all_models_benchmark.py --device C1200
  python3 scripts/run_all_models_benchmark.py --dry-run
  python3 scripts/run_all_models_benchmark.py --quiet   # 只打汇总表，不转发子脚本输出
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# (script_or_id, label, fps, infer_ms, wall_s, note) — 与 print_summary_row 参数一致
SummaryRow = Tuple[str, str, Optional[float], Optional[float], Optional[float], str]


RE_BANNER = re.compile(r"Running\s+(.+?)\s*=+\s*$")
# YOLO/OpenStereo/FCOS：{auto,top,lite} engine inference FPS；分类 cls-infer 也可能打印 Inference FPS
RE_FPS = re.compile(
    r"(?:->\s*)?(?:(?:auto|top|lite)\s+engine\s+inference\s+FPS|Inference FPS)\s*:\s*([\d.]+)",
    re.IGNORECASE,
)
RE_BSNN_INFER_MS = re.compile(
    r"bsnn\s+model\s+inference\s+time\s*:\s*([\d.]+)", re.IGNORECASE
)


def _append_file_content(chunks: List[str], path: Path) -> None:
    try:
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "\n"))
    except OSError:
        pass


def detect_device_from_system() -> Optional[str]:
    """Best-effort detect A1000/C1200/A2000 from common system descriptors."""
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
    if "a2000" in system_info:
        return "A2000"
    if "a1000" in system_info:
        return "A1000"
    if "c1200" in system_info:
        return "C1200"
    return None


@dataclass
class ParsedRun:
    label: str
    fps: Optional[float] = None
    infer_ms: Optional[float] = None

    def effective_infer_ms(self) -> Optional[float]:
        return self.infer_ms

    def effective_fps(self) -> Optional[float]:
        if self.infer_ms is not None and self.infer_ms > 0:
            return 1000.0 / self.infer_ms
        return None


def parse_demo_output(text: str, default_label: str) -> List[ParsedRun]:
    """根据 demo 标准输出解析多段推理的 FPS / 推理耗时。"""
    runs: List[ParsedRun] = []
    current = default_label
    pending_fps_idx: Optional[int] = None

    for line in text.splitlines():
        bm = RE_BANNER.search(line.strip())
        if bm:
            current = bm.group(1).strip()
            pending_fps_idx = None
            continue

        fm = RE_FPS.search(line)
        if fm:
            fps = float(fm.group(1))
            runs.append(ParsedRun(label=current, fps=fps))
            pending_fps_idx = len(runs) - 1
            continue

        im = RE_BSNN_INFER_MS.search(line)
        if im:
            ms = float(im.group(1))
            if pending_fps_idx is not None and 0 <= pending_fps_idx < len(runs):
                if runs[pending_fps_idx].infer_ms is None:
                    runs[pending_fps_idx].infer_ms = ms
                    continue
            runs.append(ParsedRun(label=current, fps=None, infer_ms=ms))
            pending_fps_idx = None

    return runs


def discover_run_scripts(models_root: Path) -> List[Path]:
    """
    各工程约定的预编译一键脚本（名称不统一）：
    - run_demo.sh：全部 sample 统一入口
    """
    patterns = (
        "**/runtime/cpp/run_demo.sh",
        "**/runtime/cpp/prebuild/run_demo.sh",
        "**/Deploy/prebuild/run_demo.sh",
        "**/Deploy/C++/prebuild/run_demo.sh",
    )
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(models_root.glob(pattern))
    seen: set[Path] = set()
    ordered: List[Path] = []
    for p in sorted(candidates, key=lambda x: str(x.resolve())):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            ordered.append(p)
    return ordered


def ensure_prebuild_app_executables(prebuild_dir: Path) -> None:
    """部分预置包中 app 下二进制未带 +x，会导致 126 Permission denied。"""
    app_dir = prebuild_dir / "app"
    if not app_dir.is_dir():
        return
    for p in app_dir.iterdir():
        if not p.is_file():
            continue
        try:
            mode = p.stat().st_mode
            if mode & 0o111:
                continue
            p.chmod(mode | 0o111)
        except OSError:
            pass


def env_with_deploy_libs(cwd: Path, base_env: dict) -> dict:
    """Prefer board-installed bsnnrt 0.13.0 (/usr/lib), then optional bundled runtimes."""
    env = dict(base_env)
    search_paths = ["/usr/lib", "/usr/lib64", "/lib"]

    lib64_local = cwd.parent / "code" / "usr" / "lib64"
    if lib64_local.is_dir():
        search_paths.append(str(lib64_local.resolve()))

    repo_root = Path(__file__).resolve().parent.parent
    for runtime in ["C1200_runtime", "A1000B_runtime", "A2000_runtime"]:
        for lib_name in ("lib", "lib64"):
            lib_dir = repo_root / "libs" / "bsnn" / runtime / "usr" / lib_name
            if lib_dir.is_dir():
                search_paths.append(str(lib_dir.resolve()))

    prev = env.get("LD_LIBRARY_PATH", "").strip()
    prefix = ":".join(dict.fromkeys(search_paths))
    env["LD_LIBRARY_PATH"] = f"{prefix}:{prev}" if prev else prefix
    env.setdefault("BSNN_RUNTIME_ROOT", "/usr")
    return env


def run_shell(
    cwd: Path,
    argv: Sequence[str],
    env: dict,
    *,
    stream_output: bool = True,
) -> Tuple[int, str, float]:
    """
    执行子进程。默认把合并后的 stdout+stderr 实时打印到当前终端，并保留全文供解析。
    stream_output=False 时仅捕获（配合 --quiet）。
    """
    t0 = time.perf_counter()
    if stream_output:
        p = subprocess.Popen(
            list(argv),
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
        list(argv),
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - t0
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out, elapsed


def optional_extra_jobs(models_root: Path) -> List[dict[str, Any]]:
    """
    无 Deploy/prebuild 一键脚本的工程：在依赖齐全时用 Python 等追加任务。
    分类板端推理已由 run_demo.sh 纳入默认扫描；此处仅保留 ONNX 全量精度验证（与 --include-extras 配合）。
    """
    jobs: List[dict[str, Any]] = []

    cls_train = models_root / "classification" / "train"
    for name, script in (
        ("resnet18", "resnet18.py"),
        ("resnet34", "resnet34.py"),
        ("mobilenetv2", "mobilenetv2.py"),
    ):
        onnx = cls_train / "compiler" / f"{name}_cifar10.onnx"
        if onnx.is_file():
            jobs.append(
                {
                    "id": f"classification-{name}-onnx-acc",
                    "cwd": cls_train,
                    "argv": [
                        sys.executable,
                        script,
                        "--mode",
                        "test_onnx",
                        "--onnx",
                        str(onnx.relative_to(cls_train)),
                        "--data_root",
                        "./data",
                        "--batch_size",
                        "128",
                    ],
                    "default_label": f"classification {name} test_onnx (full val set)",
                    "wall_only": True,
                }
            )

    return jobs


def print_summary_row(
    script_or_id: str,
    label: str,
    fps: Optional[float],
    infer_ms: Optional[float],
    wall_s: Optional[float],
    note: str = "",
) -> None:
    fps_s = f"{fps:.2f}" if fps is not None else "—"
    ms_s = f"{infer_ms:.3f}" if infer_ms is not None else "—"
    wall_s_str = f"{wall_s:.3f}" if wall_s is not None else "—"
    print(
        f"{script_or_id:48} | {label:42} | FPS: {fps_s:>8} | infer_ms: {ms_s:>10} | wall_s: {wall_s_str:>8} | {note}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all model demos under models/ and print FPS / timing.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Path to samples/vision (default: <repo>/samples/vision)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "none", "A1000", "C1200", "A2000"),
        default="auto",
        help="Device passed to prebuild scripts. Default auto-detects from system info; use none to disable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned commands.")
    parser.add_argument(
        "--no-continue-on-error",
        action="store_true",
        help="Stop after the first failed demo.",
    )
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="追加 classification test_onnx（全量验证）等无 prebuild 脚本的条目（需文件存在）。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印子脚本实时输出，仅打印末尾汇总表（仍完整捕获用于解析）。",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    models_root = args.models_root or (repo_root / "samples" / "vision")
    if not models_root.is_dir():
        print(f"[ERROR] models root not found: {models_root}", file=sys.stderr)
        return 2

    selected_device: Optional[str] = None
    if args.device == "auto":
        selected_device = detect_device_from_system()
    elif args.device != "none":
        selected_device = args.device

    device_flag: List[str] = []
    if selected_device:
        device_flag = [f"-{selected_device}"]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    scripts = discover_run_scripts(models_root)
    if not scripts:
        print(
            f"[WARN] No run_demo.sh found under {models_root}",
            file=sys.stderr,
        )

    if args.device == "auto":
        if selected_device:
            print(f"[INFO] Auto-detected device: {selected_device}")
        else:
            print("[WARN] Unable to auto-detect device, benchmark will rely on each prebuild script fallback.")
    elif selected_device:
        print(f"[INFO] Using user-specified device: {selected_device}")
    else:
        print("[INFO] Device override disabled; benchmark will not pass platform flags.")

    stream = not args.quiet
    summary_rows: List[SummaryRow] = []
    dry_run_lines: List[str] = []

    fail_count = 0
    continue_on_error = not args.no_continue_on_error

    for script_path in scripts:
        rel = script_path.relative_to(models_root)
        cwd = script_path.parent
        argv = ["bash", str(script_path.resolve())] + device_flag
        if args.dry_run:
            dry_run_lines.append(f"[DRY-RUN] cd {cwd} && {' '.join(argv)}")
            continue

        ensure_prebuild_app_executables(cwd)
        run_env = env_with_deploy_libs(cwd, env)
        if stream:
            print(f"\n{'─' * 80}\n▶ {rel}\n  cwd: {cwd}\n  cmd: {' '.join(argv)}\n{'─' * 80}\n", flush=True)
        code, text, wall = run_shell(cwd, argv, run_env, stream_output=stream)
        default_label = rel.parts[0] if rel.parts else str(rel)
        runs = parse_demo_output(text, default_label=default_label)
        hard_fail = code != 0

        if hard_fail:
            fail_count += 1
            summary_rows.append(
                (str(rel), default_label, None, None, wall, f"EXIT {code}")
            )
            if not stream:
                tail = "\n".join(text.strip().splitlines()[-12:])
                if tail:
                    print(f"--- log tail: {rel} ---\n{tail}\n", file=sys.stderr)
            if not continue_on_error:
                break
            continue

        exit_note = ""

        if not runs:
            fail_count += 1
            summary_rows.append(
                (
                    str(rel),
                    default_label,
                    None,
                    None,
                    wall,
                    "no measured inference time in output",
                )
            )
        else:
            for i, r in enumerate(runs):
                if r.infer_ms is None:
                    fail_count += 1
                label = r.label if i > 0 or r.label != default_label else default_label
                note = "no measured inference time" if r.infer_ms is None else ""
                if i == len(runs) - 1:
                    note = exit_note
                summary_rows.append(
                    (
                        str(rel) if i == 0 else "",
                        label,
                        r.effective_fps(),
                        r.effective_infer_ms(),
                        wall,
                        note,
                    )
                )

    if args.include_extras:
        for job in optional_extra_jobs(models_root):
            jid = job["id"]
            cwd: Path = job["cwd"]
            argv = job["argv"]
            default_label = job["default_label"]
            wall_only = bool(job.get("wall_only"))

            if args.dry_run:
                dry_run_lines.append(f"[DRY-RUN] cd {cwd} && {' '.join(str(x) for x in argv)}")
                continue

            if stream:
                print(
                    f"\n{'─' * 80}\n▶ extra: {jid}\n  cwd: {cwd}\n  cmd: {' '.join(str(x) for x in argv)}\n{'─' * 80}\n",
                    flush=True,
                )
            code, text, wall = run_shell(cwd, argv, env, stream_output=stream)
            if code != 0:
                fail_count += 1
                summary_rows.append((jid, default_label, None, None, wall, f"EXIT {code}"))
                if not continue_on_error:
                    break
                continue

            if wall_only:
                summary_rows.append(
                    (
                        jid,
                        default_label,
                        None,
                        None,
                        wall,
                        "full run wall (not per-frame; see demo)",
                    )
                )
            else:
                runs = parse_demo_output(text, default_label=default_label)
                if runs:
                    for i, r in enumerate(runs):
                        summary_rows.append(
                            (
                                jid if i == 0 else "",
                                r.label,
                                r.effective_fps(),
                                r.effective_infer_ms(),
                                wall,
                                "",
                            )
                        )
                else:
                    summary_rows.append(
                        (
                            jid,
                            default_label,
                            None,
                            None,
                            wall,
                            "no engine FPS in stdout; full-process wall",
                        )
                    )

    if args.dry_run:
        print("\n=== Dry run: planned commands ===\n")
        for line in dry_run_lines:
            print(line)
        print(f"\n{len(dry_run_lines)} command(s) planned.")
        return 0

    print("\n" + "=" * 120)
    print(
        f"{'script / job':48} | {'run label':42} | {'FPS':>8} | {'infer_ms':>10} | {'wall_s':>8} | note"
    )
    print("=" * 120)
    for row in summary_rows:
        print_summary_row(*row)
    print("=" * 120)

    if fail_count:
        print(f"Finished with {fail_count} failed job(s).", file=sys.stderr)
        return 1
    print("All discovered demos completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
