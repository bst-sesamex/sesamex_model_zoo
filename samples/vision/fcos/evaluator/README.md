English | [简体中文](./README_cn.md)

# FCOS Evaluation

Performance & accuracy reference for FCOS on Aura / Kalos.

## 1. Prerequisites

- Board: Aura (C1200 chip) or Kalos (A1000 chip), with `libbsnnrt.so`
- Python 3 + `pybsnn` (installed by repo-root `init_env.sh` on board)
- For accuracy: Python 3.8+, `pycocotools` (detection family) or the
  dataset-specific evaluation tool

## 2. Dataset

- Evaluation dataset: COCO val2017
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
| `fcos` | 896x896 | 50.79 FPS | 19.689 ms |

## 5. Accuracy Benchmark

**本版本未发布 COCO mAP 基线（标注为未测量）。** 如需复现：COCO val2017 + pycocotools，保持 896x896 模型 .meta 中定义的输入张量格式 预处理与 score 0.60、per-class NMS + class-agnostic NMS 阈值设置（板端日志：690 候选 → per-class NMS 143 → agnostic NMS 119 → score 过滤 8 框）。

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

1/1 模型已通过 Aura（C1200）板端推理验证（test_data/ 000000000139.jpg → fcos_inference.jpg，检出 8 目标）。
