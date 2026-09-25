English | [简体中文](./README_cn.md)

# YOLOv8 Evaluation


Performance & accuracy reference for YOLOv8 on Aura / Kalos.

## 1. Prerequisites

- Board: Aura (C1200 chip) or Kalos (A1000 chip), with `libbsnnrt.so`
- Python 3 + `pybsnn` (installed by repo-root `init_env.sh` on board)
- For accuracy: Python 3.8+, `pycocotools` (detection family) or the
  dataset-specific evaluation tool
- Models are committed in-repo under `../model/`; `../model/download.sh` verifies completeness.

## 2. Dataset

- Evaluation dataset: COCO val2017（检测/分割/姿态）
- Smoke-test image: `../test_data/`
- Full datasets are not stored in this repo; see `datasets/` at repo root

## 3. Usage

Single-image smoke test (on board):

```bash
cd ../runtime/python && python3 pybsnn_demo.py（det）等
```

Batch performance (on board, all models):

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. Performance Benchmark

Conditions: Aura C1200, default auto engine; median of three post-warmup C++ run+wait calls on a monotonic clock, excluding pre/postprocessing.

| Model | Input | Aura C1200 FPS | Aura latency |
|---|---|---|---|
| `yolov8n-det` | 640x640 | 392.46 FPS | 2.548 ms |
| `yolov8n-seg` | 640x640 | 370.23 FPS | 2.701 ms |
| `yolov8n-pose` | 640x640 | 399.36 FPS | 2.504 ms |

## 5. Accuracy Benchmark

**本版本未发布 COCO mAP 基线（标注为未测量）。** 当前 sample 以部署链路与板端验证为主。如需复现 mAP：使用 COCO val2017 + pycocotools，保持与 runtime 示例一致的 640x640 模型 .meta 中定义的输入张量格式 预处理、score 0.25 / NMS IoU 0.45 阈值；注意固定shape 量化模型与官方动态 shape 的 mAP 口径差异属预期。

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

3/3 模型（det/seg/pose）已通过 Aura（C1200）板端推理验证（test_data/ 含输入 bus.jpg 与三任务结果图）。
