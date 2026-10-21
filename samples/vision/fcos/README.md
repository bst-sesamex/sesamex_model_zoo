English | [简体中文](./README_cn.md)

# FCOS

Deployment sample of FCOS for Black Sesame Aura / Kalos platforms:
model conversion notes, model packages, Python & C++ runtimes, and
evaluation references.

## Overview

- **Task**: 目标检测（anchor-free）
- **Paper**: https://arxiv.org/pdf/1904.01355.pdf
- **Upstream implementation**: tianzhi0549/FCOS (non-commercial) -- https://github.com/tianzhi0549/FCOS

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Task | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `fcos` | 896x896 | 目标检测 | 50.79 | 19.689 ms |

Aura C1200 timing is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.

## Layout

```text
.
├── conversion/                   # conversion notes (reference type)
├── evaluator/                    # accuracy / performance reference
├── model/                        # model packages & file list
├── runtime/python/               # PyBSNN Python runtime
├── runtime/cpp/                  # C++ runtime (source + build.sh)
└── test_data/                    # input image + on-device output
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
