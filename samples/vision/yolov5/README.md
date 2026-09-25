English | [简体中文](./README_cn.md)

# YOLOv5

Deployment sample of YOLOv5 for Black Sesame Aura / Kalos platforms:
model conversion notes, model download, Python & C++ runtimes, and
evaluation references.

## Overview

- **Task**: 目标检测
- **Paper**: https://github.com/ultralytics/yolov5
- **Upstream implementation**: ultralytics/yolov5 (GPL-3.0) -- https://github.com/ultralytics/yolov5

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Task | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `yolov5m` | 640x640 | 目标检测 | 145.84 | 6.857 ms |

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
