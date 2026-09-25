English | [简体中文](./README_cn.md)

# Face Landmark Detection Evaluation


Performance & accuracy reference for face landmark detection on Aura / Kalos.

## 1. Prerequisites

- Board: Aura (C1200 chip) or Kalos (A1000 chip), with `libbsnnrt.so`
- Python 3 + `pybsnn` (installed by repo-root `init_env.sh` on board)
- For accuracy: Python 3.8+, 300W dataset evaluation tooling
- Models are committed in-repo under `../model/`; `../model/download.sh` verifies completeness.

## 2. Dataset

- Evaluation dataset: 300W (68-point protocol)
- Smoke-test images: `../test_data/` (56/224 face crops)
- Full datasets are not stored in this repo; see `datasets/` at repo root

## 3. Usage

Single-image smoke test (on board):

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

C++ demo (both sizes):

```bash
cd ../runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

Batch performance (on board, all models):

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. Performance Benchmark

Conditions: Aura C1200, default auto engine; median of three post-warmup C++ run+wait calls on a monotonic clock, excluding pre/postprocessing.

| Model | Input | Aura C1200 FPS | Aura latency |
|---|---|---|---|
| `size56` | 56x56 | 2403.85 FPS | 0.416 ms |
| `size224` | 224x224 | 1506.02 FPS | 0.664 ms |

Kalos (A1000 chip) data: not measured for this release.

## 5. Accuracy Benchmark

**本版本未发布 300W NME 基线（标注为未测量）。** 复现口径：300W 68 点协议，归一化误差 NME（inter-ocular distance），
预处理（对齐裁剪到 56/224）与 runtime 示例一致。量化模型与 float 模型需在同一协议下对比。

## 6. Performance notes

- Reproduce with: bash utils/tools/run_all_models_benchmark.sh --device C1200
- Set the board to performance mode; avoid concurrent CPU/DSP load
- C++ latency is the median of three post-warmup run+wait calls on a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.
  crop alignment and keypoint decoding

## 7. Accuracy notes

- Metric: NME (normalized mean error) over 68 points
- Fixed-shape quantized models differ from official dynamic-shape numbers by
  design; compare float vs quantized under the same protocol
- Calibration data: 100-500 representative face crops
- Values not listed = not measured for this release (no guessed numbers)

## 8. Verification summary

2/2 变体已通过 Aura（C1200）板端推理验证（benchmark_results 记录：68 点输出，
norm range [0.117, 0.898]，SUCCESS）。
