English | [简体中文](./README_cn.md)

# Face Landmark Detection (68-point)

Deployment sample of face landmark detection for Black Sesame Aura / Kalos platforms:
model conversion notes, model packages, Python & C++ runtimes, and evaluation references.

## Overview

- **Task**: face landmark detection (68 points)
- **Upstream implementation**: https://github.com/cunjian/pytorch_face_landmark

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Task | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `size56` | 56x56 | face landmark (68 pts) | 2403.85 | 0.416 ms |
| `size224` | 224x224 | face landmark (68 pts) | 1506.02 | 0.664 ms |

Aura C1200 timing is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.

## Layout

```text
.
├── conversion/                   # conversion notes (reference type)
├── evaluator/                    # accuracy / performance reference
├── model/                        # model packages & file list
├── runtime/python/               # PyBSNN Python runtime
├── runtime/cpp/                  # C++ runtime (source + build.sh)
└── test_data/                    # input crops + on-device output
```

## Quick start

```bash
# 1. Python runtime (on board; run repo-root init_env.sh first)
cd runtime/python
python3 pybsnn_demo.py

# 2. C++ runtime (native build on board)
cd runtime/cpp && bash build.sh && sh run_demo.sh
```

See [evaluator/README.md](./evaluator/README.md) for benchmark & accuracy notes.
