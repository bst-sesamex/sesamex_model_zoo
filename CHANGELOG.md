# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Restored quantized model package names and C++ runtime sources from the source project.
- Unified warmup and board-side monotonic timing in C++ and Python examples.
- Aligned C1200 inference latency and derived FPS across the index, sample pages, and benchmark manifest.
- Recorded SHA-256 values for model assets; kept device-specific validation reports outside the algorithm repository.

## [1.0.1] - 2026-09-24

### Changed

- **In-repo weights**: quantized model packages (80 files, ~245 MB, C1200 chip)
  are now committed under each sample's `model/c1200/` directory - all 10
  vision samples ship complete packages. The former model-server download
  channel (`pkgs.bstai.top`, not publicly reachable) is retired;
  `model/download.sh` now verifies file presence instead of downloading.
  A1000 (Kalos) packages are not included in this release.
- **PyBSNN wheel in-repo**: `pybsnn/aura/pybsnn-0.2.0+0.13.0-cp310-cp310-linux_aarch64.whl`
  (C1200, aarch64) is committed; `init_env.sh` installs it from the repository
  instead of downloading.

### Known limitations

- Accuracy baselines (PyTorch float references) are not measured for this
  release; see each sample's `evaluator/README` for what is recorded.
- No md5/sha256 manifest for model packages yet.
- No A1000 (Kalos) model packages or PyBSNN wheel yet.

## [1.0.0] - 2026-09-23

### Added

- Initial open-source release of the model zoo.
- Vision samples: `openstereo2`, `yolov8`, `yolov5`, `yolov12`, `fcos`, `classification`, `condlanenet`,
  `landmark`, `reid`, `unet-seg`.
- Each sample follows the standard layout: `conversion/`, `model/`, `evaluator/`, `runtime/python/`, `runtime/cpp/`, `test_data/`.
- Shared libraries under `utils/py_utils/` and batch tools under `utils/tools/`.
- Dataset preparation scripts under `datasets/` (COCO download script, ImageNet
  layout, dataset index).
- Release manifests: `docs/manifests/models.yaml`, `docs/manifests/benchmarks.yaml`
  (20 model variants inventoried; checksums not yet recorded).
- CI workflow for lint and script sanity checks (`.github/workflows/ci.yml`).

### Known limitations

- Accuracy baselines (PyTorch float references) are not measured for this
  release; see each sample's `evaluator/README` for what is recorded.
- No md5/sha256 manifest for model packages yet.
