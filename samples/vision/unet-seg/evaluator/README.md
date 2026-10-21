English | [简体中文](./README_cn.md)

# Semantic Segmentation (UNet) Evaluation

Performance & accuracy reference for semantic segmentation on Aura / Kalos.

## 1. Prerequisites

- Board: Aura (C1200 chip) or Kalos (A1000 chip), with `libbsnnrt.so`
- Python 3 + `pybsnn` (installed by repo-root `init_env.sh` on board)
- For accuracy: Python 3.8+, CamVid evaluation tooling (mIoU)

## 2. Dataset

- Evaluation dataset: CamVid (19-class protocol, 2048x1024 source images)
- Smoke-test image: `../test_data/test.png` (2048x1024) + on-device outputs
- Full datasets are not stored in this repo; see `datasets/` at repo root

## 3. Usage

Single-image smoke test (on board):

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

C++ demo (both input shapes, overlay output):

```bash
cd ../runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

Batch performance (on board, all models):

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. Performance Benchmark

Conditions: Aura C1200, default auto engine; median of three post-warmup C++ run+wait calls on a monotonic clock, excluding pre/postprocessing.
Inference runs at the model input resolution; the 2048x1024 image is tiled/resized
before inference in the demo pipeline.

| Model | Input | Aura C1200 FPS | Aura latency |
|---|---|---|---|
| `unet_512x512` | 512x512 | 35.51 FPS | 28.158 ms |
| `unet_512x1024` | 512x1024 | 18.15 FPS | 55.111 ms |

Kalos (A1000 chip) data: not measured for this release.

## 5. Accuracy Benchmark

**本版本未发布 CamVid mIoU 基线（标注为未测量）。** 复现口径：CamVid 19 类协议，
全图/滑窗推理后 argmax，与 float 模型在同一预处理（BGR uint8 无归一化）下对比 mIoU。

## 6. Performance notes

- Reproduce with: bash utils/tools/run_all_models_benchmark.sh --device C1200
- Set the board to performance mode; avoid concurrent CPU/DSP load
- C++ latency is the median of three post-warmup run+wait calls on a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.
  (postprocess argmax + overlay: 20-41 ms at 2048x1024)

## 7. Accuracy notes

- Metric: mIoU (19 classes, CamVid protocol)
- Output is INT8 logits with dequant scale 0.250; argmax is scale-invariant
- Calibration data: 100-500 representative street-scene crops
- Values not listed = not measured for this release (no guessed numbers)

## 8. Verification summary

2/2 变体已通过 Aura（C1200）板端推理验证（benchmark_results 记录：
512x512x19 / 1024x512x19 输出张量，dequant scale 0.250，SUCCESS）。
