English | [简体中文](./README_cn.md)

# Person Re-Identification Evaluation


Performance & accuracy reference for person re-identification on Aura / Kalos.

## 1. Prerequisites

- Board: Aura (C1200 chip) or Kalos (A1000 chip), with `libbsnnrt.so`
- Python 3 + `pybsnn` (installed by repo-root `init_env.sh` on board)
- For accuracy: Python 3.8+, Torchreid rank evaluation tooling
- Models are committed in-repo under `../model/`; `../model/download.sh` verifies completeness.

## 2. Dataset

- Evaluation dataset: Market-1501 / DukeMTMC-reID (standard rank-1/mAP protocol)
- Smoke-test images: `../test_data/` (pedestrian crops, pair True/False demos)
- Full datasets are not stored in this repo; see `datasets/` at repo root

## 3. Usage

Single-image smoke test (on board):

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

C++ demo (pair matching, 3 backbones x True/False):

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
| `mobilenetv2` | 128x256 | 1785.71 FPS | 0.560 ms |
| `osnet` | 128x256 | 203.52 FPS | 4.9135 ms |
| `resnet50` | 128x256 | 469.04 FPS | 2.132 ms |

Kalos (A1000 chip) data: not measured for this release.

## 5. Accuracy Benchmark

**本版本未发布 Market-1501 rank-1/mAP 基线（标注为未测量）。** 复现口径：Market-1501 标准协议
（query/gallery 划分、CDF/最大化准则），特征为量化模型 1280/2048 维嵌入 + 余弦距离 + C++ demo 阈值。

## 6. Performance notes

- Reproduce with: bash utils/tools/run_all_models_benchmark.sh --device C1200
- Set the board to performance mode; avoid concurrent CPU/DSP load
- C++ latency is the median of three post-warmup run+wait calls on a board-side monotonic clock; FPS = 1000 / latency. Pre/postprocessing is excluded.
  extra inference for the second image plus cosine distance (< 0.01 ms)

## 7. Accuracy notes

- Metric: rank-1 / mAP (Market-1501 protocol)
- Pair thresholds (cosine distance): mobilenetv2 0.3193, osnet 0.3677,
  resnet50 0.2576 - calibrated on-device, see conversion/README.md
- Calibration data: 100-500 representative pedestrian crops
- Values not listed = not measured for this release (no guessed numbers)

## 8. Verification summary

3/3 变体已通过 Aura（C1200）板端推理验证（benchmark_results 记录：6 对 True/False
匹配结果与板端标定阈值一致，SUCCESS）。
