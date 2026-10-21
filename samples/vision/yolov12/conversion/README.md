English | [简体中文](./README_cn.md)

# YOLOv12 / YOLO11 / YOLOv10 Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `yolov10n-det` | 3x640x640 RGB888P (published package) | `.bin` 模型包 |
| `yolov11n-det` | 3x640x640 RGB888P (published package) | `.bin` 模型包 |
| `yolov12n-det` | 3x640x640 RGB888P (published package) | `.bin` 模型包 |

## Step 1: Export ONNX

Upstream implementation: ultralytics/ultralytics（AGPL-3.0） (https://github.com/ultralytics/ultralytics).

Check the exported ONNX input shape, name, color order and preprocessing against your training code before compiling.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` defines the image color conversion, mean and scale used at calibration time.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./yolov12n-det_det_c1200_640x640_rgb.yaml) reads the included ONNX from `assets/yolov12n-det/`. Put representative images in `calibration_images/yolov12n-det/` (create the folder) and run from the repository root:

```bash
cd samples/vision/yolov12/conversion
bstnnx_run --config yolov12n-det_det_c1200_640x640_rgb.yaml
```

## Variant examples

- `yolov10n-det`: [yolov10n-det_det_c1200_640x640_rgb.yaml](./yolov10n-det_det_c1200_640x640_rgb.yaml)
- `yolov11n-det`: [yolov11n-det_det_c1200_640x640_rgb.yaml](./yolov11n-det_det_c1200_640x640_rgb.yaml)
- `yolov12n-det`: [yolov12n-det_det_c1200_640x640_rgb.yaml](./yolov12n-det_det_c1200_640x640_rgb.yaml)

## Output protocol

与 YOLOv8 检测头协议一致（DFL reg + cls，stride 8/16/32）。以 runtime/python/pybsnn_demo.py 实际解析为准。

See runtime/python/pybsnn_demo.py for the authoritative parsing.
