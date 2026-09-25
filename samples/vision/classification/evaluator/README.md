English | [简体中文](./README_cn.md)

# Image Classification（ResNet-18/34、MobileNetV2） Evaluation


Performance & accuracy reference for Image Classification（ResNet-18/34、MobileNetV2） on Aura / Kalos.

## 1. Prerequisites

- Board: Aura (C1200 chip) or Kalos (A1000 chip), with `libbsnnrt.so`
- Python 3 + `pybsnn` (installed by repo-root `init_env.sh` on board)
- For accuracy: Python 3.8+, `pycocotools` (detection family) or the
  dataset-specific evaluation tool
- Models are committed in-repo under `../model/`; `../model/download.sh` verifies completeness.

## 2. Dataset

- Evaluation dataset: ImageNet val
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
| `resnet18` | 224x224 | 763.36 FPS | 1.310 ms |
| `resnet34` | 224x224 | 490.44 FPS | 2.039 ms |
| `mobilenetv2` | 224x224 | 1464.13 FPS | 0.683 ms |

## 5. Accuracy Benchmark

**本版本未发布 ImageNet Top-1 基线（标注为未测量）。** 复现口径：ImageNet val（50000 张）+ Top-1/Top-5，224x224 中心裁剪 + 模型 .meta 中定义的输入张量格式 转换与 runtime 示例一致。

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

3/3 模型已通过 Aura（C1200）板端推理验证（test_data/bird_303.png 输入）。
