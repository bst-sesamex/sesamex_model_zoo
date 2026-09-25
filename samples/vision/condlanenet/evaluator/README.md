English | [简体中文](./README_cn.md)

# CondLaneNet Evaluation


Performance & accuracy reference for CondLaneNet on Aura / Kalos.

## 1. Prerequisites

- Board: Aura (C1200 chip) or Kalos (A1000 chip), with `libbsnnrt.so`
- Python 3 + `pybsnn` (installed by repo-root `init_env.sh` on board)
- For accuracy: Python 3.8+, `pycocotools` (detection family) or the
  dataset-specific evaluation tool
- Models are committed in-repo under `../model/`; `../model/download.sh` verifies completeness.

## 2. Dataset

- Evaluation dataset: TuSimple（车道线）
- Smoke-test image: `../test_data/`
- Full datasets are not stored in this repo; see `datasets/` at repo root

## 3. Usage

Single-image smoke test (on board):

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

Batch performance (on board, all models):

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. Performance Benchmark

Conditions: Aura C1200, default auto engine; median of three post-warmup C++ run+wait calls on a monotonic clock, excluding pre/postprocessing.

| Model | Input | Aura C1200 FPS | Aura latency |
|---|---|---|---|
| `condlanenet` | 800x320 | 148.54 FPS | 6.732 ms |

## 5. Accuracy Benchmark

**TuSimple F1 = 0.953**（draft 记录口径，已测量）。复现：TuSimple test set，800x320 预处理与后处理阈值与 runtime 示例保持一致。

## 6. Performance notes

- Reproduce with: bash utils/tools/run_all_models_benchmark.sh --device C1200
- Set the board to performance mode; avoid concurrent CPU/DSP load
- C++ latency is the median of three post-warmup run+wait calls on a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.

## 7. Accuracy notes

- Metric per task: COCO mAP (det/seg/pose), Top-1/Top-5 (classification),
  F1 (lane), EPE (stereo)
- Fixed-shape quantized models differ from official dynamic-shape numbers by
  design; compare float vs quantized under the same protocol
- Calibration data: 100-500 representative inputs
- Values not listed = not measured for this release (no guessed numbers)

## 8. Verification summary

1/1 模型已通过 Aura（C1200）板端推理验证（test_data/vis_cpp.jpg 结果图）。
