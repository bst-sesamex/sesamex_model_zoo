"""Platform-aware prebuild paths for PyBSNN demo scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

DEVICE_TO_PLATFORM = {
    "A1000": "a1000",
    "C1200": "c1200",
    "A2000": "a2000",
}


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


def get_platform(device: Optional[str] = None) -> str:
    if device is None:
        device = os.environ.get("BST_BENCHMARK_DEVICE")
    if device is None:
        device = detect_device_from_system()
    if device:
        return DEVICE_TO_PLATFORM.get(device.upper(), device.lower())
    return "c1200"


def sample_root(script_dir: str) -> Path:
    """<sample>/runtime/python -> <sample>"""
    p = Path(script_dir).resolve()
    if p.name in ("python", "cpp"):
        return p.parent.parent
    return p


def model_dir_root(script_dir: str) -> Path:
    """Preferred model storage: <sample>/model (in-repo weights; download.sh verifies)."""
    return sample_root(script_dir) / "model"


def prebuild_dir(script_dir: str) -> Path:
    """Legacy C++ prebuild layout (kept for compatibility with older samples)."""
    return (sample_root(script_dir) / "runtime" / "cpp" / "prebuild").resolve()


def _has_bsnn_artifacts(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(path.glob("*.meta")):
        return True
    if any(path.glob("*.Enrsd.bin")) or any(path.glob("*_stripped.lib")):
        return True
    inner = path / "model"
    if inner.is_dir():
        if any(inner.glob("*.meta")):
            return True
        if any(inner.glob("*.Enrsd.bin")) or any(inner.glob("*_stripped.lib")):
            return True
    return False


def resolve_model_dir(
    script_dir: str,
    variant: str,
    *,
    inner_model: bool = False,
    device: Optional[str] = None,
    legacy_candidates: Sequence[str] = (),
) -> str:
    """
    Resolve prebuild/model/<platform>/<variant>-<platform>[/model].

    ``variant`` examples: yolov8n-det, fcos, resnet34, lightstereo, yolov5m.
    """
    pb = prebuild_dir(script_dir)
    mroot = model_dir_root(script_dir)
    platform = get_platform(device)

    def _variant_dirs(base: Path) -> List[Path]:
        return [
            base / platform / f"{variant}-{platform}",
            base / f"{variant}-{platform}",
        ]

    candidates: List[Path] = []
    # New layout: <sample>/model/... (in-repo weights)
    for d in _variant_dirs(mroot):
        candidates.append(d / "model" if (d / "model").is_dir() else d)
    # Legacy layout: runtime/cpp/prebuild/model/...
    for d in _variant_dirs(pb / "model"):
        candidates.append(d / "model" if (d / "model").is_dir() else d)
    for rel in legacy_candidates:
        candidates.append((pb / rel).resolve())

    for path in candidates:
        if _has_bsnn_artifacts(path):
            if inner_model and (path / "model").is_dir() and any((path / "model").glob("*.meta")):
                return str((path / "model").resolve())
            return str(path.resolve())

    return str(candidates[0])


def resolve_prebuild_asset(script_dir: str, *parts: str) -> str:
    """Resolve a sample asset (test image etc.).

    Search order:
      1. <sample>/test_data/<parts>      (new layout)
      2. <sample>/runtime/cpp/prebuild/<parts>  (legacy layout)
      3. <sample>/runtime/python/<parts> (demo-local outputs)
    """
    root = sample_root(script_dir)
    search = [
        root / "test_data",
        root / "runtime" / "cpp" / "prebuild",
        root / "runtime" / "python",
    ]
    for base in search:
        p = base.joinpath(*parts)
        if p.exists():
            return str(p.resolve())
    return str((root / "test_data").joinpath(*parts).resolve())


def as_chw_uint8(img_hwc: np.ndarray) -> np.ndarray:
    """Convert HWC BGR/RGB uint8 to C-contiguous CHW for bsnnrt 0.13.0 WHD inputs."""
    if img_hwc.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {img_hwc.shape}")
    return np.ascontiguousarray(img_hwc.transpose(2, 0, 1), dtype=np.uint8)


def env_with_bsnn_runtime(
    repo_root: Path,
    base_env: Optional[dict] = None,
) -> dict:
    """Set BSNN_RUNTIME_ROOT and LD_LIBRARY_PATH for PyBSNN / libbsnn on device."""
    env = dict(base_env if base_env is not None else os.environ)
    env["BSNN_RUNTIME_ROOT"] = "/usr"
    prev = env.get("LD_LIBRARY_PATH", "").strip()
    env["LD_LIBRARY_PATH"] = f"/usr/lib:/lib:{prev}" if prev else "/usr/lib:/lib"
    return env
