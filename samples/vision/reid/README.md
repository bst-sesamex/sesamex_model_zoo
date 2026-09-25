English | [简体中文](./README_cn.md)

# Person Re-Identification (MobileNetV2 / OSNet / ResNet-50)

Deployment sample of person re-identification (feature embedding + pair matching)
for Black Sesame Aura / Kalos platforms: model conversion notes, model download,
Python & C++ runtimes, and evaluation references.

## Overview

- **Task**: person re-identification (pair-wise cosine matching)
- **Upstream implementation**: https://github.com/KaiyangZhou/deep-person-reid (Torchreid, MIT)

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Embedding | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `mobilenetv2` | 128x256 | 1280-d | 1785.71 | 0.560 ms |
| `osnet` | 128x256 | 1280-d | 203.52 | 4.9135 ms |
| `resnet50` | 128x256 | 2048-d | 469.04 | 2.132 ms |

Aura C1200 timing is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.

## Layout

```text
.
├── conversion/                   # conversion notes (reference type)
├── evaluator/                    # accuracy / performance reference
├── model/                        # model download script & file list
├── runtime/python/               # PyBSNN Python runtime
├── runtime/cpp/                  # C++ runtime (source + build.sh)
└── test_data/                    # input pedestrian crops
```

## Quick start

```bash
# 1. model files are committed in-repo (optional completeness check)
cd model && ./download.sh   # completeness check && cd ..

# 2. Python runtime (on board; run repo-root init_env.sh first)
cd runtime/python
python3 pybsnn_demo.py

# 3. C++ runtime (native build on board); runs True/False pair tests
cd runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

See [evaluator/README.md](./evaluator/README.md) for benchmark & accuracy notes.
