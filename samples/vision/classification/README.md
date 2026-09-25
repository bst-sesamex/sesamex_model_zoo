English | [简体中文](./README_cn.md)

# Image Classification (ResNet-18/34, MobileNetV2)

Deployment sample of Image Classification (ResNet-18/34, MobileNetV2) for Black Sesame Aura / Kalos platforms:
model conversion notes, model download, Python & C++ runtimes, and
evaluation references.

## Overview

- **Task**: 图像分类
- **Paper**: https://arxiv.org/abs/1512.03385 (ResNet) / https://arxiv.org/abs/1801.04381 (MobileNetV2)
- **Upstream implementation**: 自研模型定义（随 conversion/ 提供） -- ./conversion/

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Task | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `resnet18` | 224x224 | 图像分类 | 763.36 | 1.310 ms |
| `resnet34` | 224x224 | 图像分类 | 490.44 | 2.039 ms |
| `mobilenetv2` | 224x224 | 图像分类 | 1464.13 | 0.683 ms |

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
