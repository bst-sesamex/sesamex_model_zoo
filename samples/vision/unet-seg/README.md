English | [简体中文](./README_cn.md)

# Semantic Segmentation (UNet, ESPNet-v2 backbone)

Deployment sample of real-time semantic segmentation for Black Sesame Aura / Kalos
platforms: model conversion notes, model packages, Python & C++ runtimes, and
evaluation references.

## Overview

- **Task**: semantic segmentation (19 classes, CamVid protocol)
- **Upstream implementation**: https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks (ESPNet-v2)

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Output | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `unet_512x512` | 512x512 | 512x512x19 | 35.51 | 28.158 ms |
| `unet_512x1024` | 512x1024 | 512x1024x19 | 18.15 | 55.111 ms |

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

# 2. C++ runtime (native build on board); runs both input shapes
cd runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

See [evaluator/README.md](./evaluator/README.md) for benchmark & accuracy notes.
