English | [简体中文](./README_cn.md)

# OpenStereo-2 (LightStereo)

Deployment sample of OpenStereo-2 (LightStereo) for Black Sesame Aura / Kalos platforms:
model conversion notes, model download, Python & C++ runtimes, and
evaluation references.

## Overview

- **Task**: 双目立体匹配 / 深度估计
- **Paper**: LightStereo: Simplifying and Strengthening Stereo Matching
- **Upstream implementation**: OpenStereo（来源与许可需二次审计，本版本不带训练代码） -- https://github.com/XiandaGuo/OpenStereo

Training-side code is NOT part of this repository (see `conversion/README.md`).

## Supported models

| Model | Input | Task | Aura C1200 FPS | Inference latency |
|---|---|---|---|---|
| `lightstereo` | 960x544 | 立体匹配 | 14.64 | 68.291 ms |

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
