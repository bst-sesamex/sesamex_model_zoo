# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.1.0] - 2026-10-21

- **openstereo2**: rebuilt both C1200 and Kalos (A1000) `lightstereo` packages
  from the repository's own float ONNX with BSTNNX 8.3.1. Both packages share
  the same single-output graph (`disp_pred`, consensus fused in-network); the
  C++/Python demos are adapted accordingly. Resolves the A1000
  first-inference-timeout issue of the superseded package; the C1200 package
  is ~22% faster (87.0 ms vs 111.8 ms, py).
- **Kalos (A1000) packages for all 10 vision samples** now ship with the
  repository (openstereo2 rebuilt as above; the other 19 variants use the
  previously validated packages).
- Unified warmup and board-side monotonic timing in C++ and Python examples.
- Recorded SHA-256 values for C1200 model assets; device-specific validation
  reports are kept outside the algorithm repository.

## [1.0.1] - 2026-09-24

### Changed

- **In-repo weights**: quantized model packages are committed under each
  sample's `model/<platform>/` directory - all 10 vision samples ship
  complete C1200 packages; A1000 (Kalos) packages ship as well.
- **PyBSNN wheel in-repo**: `pybsnn/aura/pybsnn-0.2.0+0.13.0-cp310-cp310-linux_aarch64.whl`
  (C1200, aarch64) is committed; `init_env.sh` installs it from the repository.

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

