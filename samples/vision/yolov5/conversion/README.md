English | [简体中文](./README_cn.md)

# YOLOv5 Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `yolov5m` | 3x640x640 RGB888P (published package) | `.bin` 模型包 |

## Step 1: Export ONNX

Upstream implementation: ultralytics/yolov5（GPL-3.0） (https://github.com/ultralytics/yolov5).

Check the exported ONNX input shape, name, color order and preprocessing against your training code before compiling.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` defines the image color conversion, mean and scale used at calibration time.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./yolov5m_det_c1200_640x640_rgb.yaml) reads the included ONNX from `assets/yolov5m/`. Put representative images in `calibration_images/yolov5m/` (create the folder) and run from the repository root:

```bash
cd samples/vision/yolov5/conversion
bstnnx_run --config yolov5m_det_c1200_640x640_rgb.yaml
```

## Output protocol

v5.0 及以下 head：3 层级 reg/cls 张量对。以 runtime/python/pybsnn_demo.py 实际解析为准。

See runtime/python/pybsnn_demo.py for the authoritative parsing.
