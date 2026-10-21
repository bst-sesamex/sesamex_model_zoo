English | [简体中文](./README_cn.md)

# Face Landmark Detection Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `size56` | 1x3x56x56 RGB888P (published package) | `.bin` model package |
| `size224` | 1x3x224x224 RGB888P (published package) | `.bin` model package |

## Step 1: Export ONNX

Upstream implementation: [pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)
(68-point models; the deployed variants use coordinate-regression heads).

Check the exported ONNX shape and dtype. The archived compiler config reads BGR calibration images and converts them to RGB using its mean/scale.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` defines the image color conversion, mean and scale used at calibration time.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./landmark_size56_c1200_56x56_rgb.yaml) reads the included ONNX from `assets/size56/`. Put representative images in `calibration_images/size56/` (create the folder) and run from the repository root:

```bash
cd samples/vision/landmark/conversion
bstnnx_run --config landmark_size56_c1200_56x56_rgb.yaml
```

## Variant examples

- `size56`: [landmark_size56_c1200_56x56_rgb.yaml](./landmark_size56_c1200_56x56_rgb.yaml)
- `size224`: [landmark_size224_c1200_224x224_rgb.yaml](./landmark_size224_c1200_224x224_rgb.yaml)

## Output protocol

68 个关键点的归一化坐标，扁平化为 136 维向量 `[x0, y0, x1, y1, ...]`（值域 [0,1]）。
runtime/cpp 的 `Landmark Result Summary` 与 runtime/python 的 `postprocess` 为权威解析实现。

See runtime/python/pybsnn_demo.py for the authoritative parsing.
