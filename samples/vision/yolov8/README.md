English | [简体中文](./README_cn.md)

# YOLOv8

Deployment sample of YOLOv8 for Black Sesame Aura / Kalos platforms:
model conversion notes, model download, Python & C++ runtimes, and
evaluation references.

## Overview

- **Task**: 目标检测 / 实例分割 / 姿态估计
- **Paper**: https://arxiv.org/abs/2305.02233
- **Upstream implementation**: ultralytics/ultralytics (AGPL-3.0) -- https://github.com/ultralytics/ultralytics

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Task | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `yolov8n-det` | 640x640 | 目标检测 | 392.46 | 2.548 ms |
| `yolov8n-seg` | 640x640 | 实例分割 | 370.23 | 2.701 ms |
| `yolov8n-pose` | 640x640 | 姿态估计 | 399.36 | 2.504 ms |

Aura C1200 timing is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.

## Layout

```text
.
├── conversion/                   # conversion notes (reference type)
├── evaluator/                    # accuracy / performance reference
├── model/                        # model download script & file list
├── runtime/python/               # PyBSNN Python runtime
├── runtime/cpp/                  # C++ runtime (source + build.sh)
└── test_data/                    # input image + on-device output
```

## Quick start

```bash
# 1. model files are committed in-repo (optional completeness check)
cd model && ./download.sh   # completeness check && cd ..

# 2. Python runtime (on board; run repo-root init_env.sh first)
cd runtime/python
python3 pybsnn_demo.py

# 3. C++ runtime (native build on board)
cd runtime/cpp && bash build.sh
```

See [evaluator/README.md](./evaluator/README.md) for benchmark & accuracy notes.
