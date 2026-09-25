English | [简体中文](./README_cn.md)

# CondLaneNet Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `condlanenet` | 3x320x800 RGB888P (published package) | `.bin` 模型包 |

## Step 1: Export ONNX

Upstream implementation: coremagi/CondLaneNet（实现基于 mmdetection v2.0.0，Apache-2.0） (https://github.com/coremagi/CondLaneNet).

Check the exported ONNX input shape, name, color order and preprocessing against your training code before compiling.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` uses the source project's image color conversion, mean and scale.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./condlanenet_lane_c1200_800x320_bgr.yaml) reads the included ONNX from `assets/condlanenet/`. Put representative images in `calibration_images/condlanenet/` (create the folder) and run from the repository root:

```bash
cd samples/vision/condlanenet/conversion
bstnnx_run --config condlanenet_lane_c1200_800x320_bgr.yaml
```

This is a conversion example, not a reproducible recipe for the committed quantized package. Check the exported ONNX input names, shapes, preprocessing and output protocol before compiling. Published packages may have been built with a different toolchain version. The generated files and layout depend on the compiler version. `../model/download.sh` checks that the committed package files exist; it performs no download.

## Output protocol

车道线实例：行方向分类 + start/end offset（单条车道线输出 stride 分辨率上的 keypoints 集合）。以 runtime/python/pybsnn_demo.py 实际解析为准。

See runtime/python/pybsnn_demo.py for the authoritative parsing.
