# Contributing to sesamex-model-zoo

Thanks for your interest in contributing! This document describes how to add a new
model sample to the zoo.

## Repository layout

```
sesamex-model-zoo/
├── samples/<category>/<model>/   # one directory per model family
├── utils/py_utils/               # shared pre/post-processing, visualization libs
├── utils/tools/                  # batch perf / eval / mapping tools
├── datasets/                     # dataset preparation scripts
└── docs/                         # release process, catalog generator
```

Categories: `vision/` (classification, detection, segmentation, OCR, depth),
`robotics/` (gait, policy, navigation perception), `embodied/` (LLM, multimodal,
speech).

## Adding a new model sample

A sample directory MUST follow the "7-piece" layout:

```
samples/vision/<model>/
├── README.md / README_cn.md      # 1. overview + variant index table
├── conversion/                   # 2. export ONNX -> calibration data -> quantized compile
│   ├── README.md / README_cn.md
│   └── <model>_<task>_<chip>_<WxH>_<inputfmt>.yaml
├── evaluator/                    # 3. accuracy report + scripts
│   ├── README.md / README_cn.md  #    8-section report, see below
│   ├── eval_<model>_<task>_<dataset>.py
│   └── eval_batch.py
├── model/                        # 4. model files
│   ├── README.md / README_cn.md  #    file list + versions + md5
│   └── download.sh
├── runtime/                      # 5/6. runtimes
│   ├── python/
│   │   ├── main.py
│   │   ├── <model>_<task>.py
│   │   ├── run.sh
│   │   └── README.md / README_cn.md
│   └── cpp/
│       ├── CMakeLists.txt
│       ├── main.cc
│       └── README.md
└── test_data/                    # 7. input images + on-device output images
```

### Naming convention

Artifacts are named `{model}_{task}_{chip}_{WxH}_{inputfmt}.{ext}`, e.g.
`yolov8n_det_c1200_640x640_nv12.bin`.

### One directory, multiple variants

When a directory hosts a model family (e.g. `yolov12/` containing v10n/v11n/v12n),
the directory-level README carries a family overview + variant index table;
`conversion/` and `model/` hold one yaml and one download entry per variant;
routines in `runtime/` are split by task.

### Licensing rules

- The repository is Apache-2.0. Training code of third-party frameworks is NOT
  committed here; `conversion/README.md` links to the upstream repository instead.
- Model weights ARE committed in-repo since v1.0.1 (`model/c1200/...`); do not
  commit SDK binaries or unreviewed large assets. Curated ONNX and training
  checkpoints belong under `conversion/assets/` via Git LFS with source and
  SHA-256 entries in `docs/manifests/conversion_assets.yaml`; `model/download.sh`
  only verifies file presence (no downloads).
- Do not include internal hostnames, IP addresses, credentials or personal
  information in any file.

## Evaluator report (8 mandatory sections)

`evaluator/README_cn.md` must contain, in order:

1. Prerequisites (Python env / dependencies / device preparation)
2. Dataset description (e.g. COCO2017 val 5000 images, ImageNet val, SceneFlow,
   TuSimple) and directory layout
3. Usage (one copy-paste command per task)
4. Performance benchmark table (single-thread latency ms / multi-thread FPS with
   thread count; note NV12 input, on-device measurement, saturation point)
5. Accuracy benchmark table (PyTorch baseline -> quantized -> retention %, split
   by task)
6. Performance test notes (command examples + high-performance mode settings)
7. Accuracy test notes (metric definition / thresholds / calibration image count /
   differences vs. official tools)
8. Verification summary (N/N models passed on-device verification)

## Pull requests

1. Fork, create a branch, make your change.
2. Run `bash -n` on changed shell scripts and `python -m py_compile` on changed
   Python files (CI does the same).
3. Update `CHANGELOG.md` under an *Unreleased* heading.
4. Open a PR describing what changed and how you verified it.
