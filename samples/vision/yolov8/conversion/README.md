English | [简体中文](./README_cn.md)

# YOLOv8 Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `yolov8n-det` | 3x640x640 RGB888P (published package) | `.bin` 模型包 |
| `yolov8n-seg` | 3x640x640 RGB888P (published package) | `.bin` 模型包 |
| `yolov8n-pose` | 3x640x640 RGB888P (published package) | `.bin` 模型包 |

## Step 1: Export ONNX

Upstream implementation: ultralytics/ultralytics（AGPL-3.0） (https://github.com/ultralytics/ultralytics).

Check the exported ONNX input shape, name, color order and preprocessing against your training code before compiling.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` uses the source project's image color conversion, mean and scale.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./yolov8n-det_det_c1200_640x640_rgb.yaml) reads the included ONNX from `assets/yolov8n-det/`. Put representative images in `calibration_images/yolov8n-det/` (create the folder) and run from the repository root:

```bash
cd samples/vision/yolov8/conversion
bstnnx_run --config yolov8n-det_det_c1200_640x640_rgb.yaml
```

This is a conversion example, not a reproducible recipe for the committed quantized package. Check the exported ONNX input names, shapes, preprocessing and output protocol before compiling. Published packages may have been built with a different toolchain version. The generated files and layout depend on the compiler version. `../model/download.sh` checks that the committed package files exist; it performs no download.

## Variant examples

- `yolov8n-det`: [yolov8n-det_det_c1200_640x640_rgb.yaml](./yolov8n-det_det_c1200_640x640_rgb.yaml)
- `yolov8n-seg`: [yolov8n-seg_seg_c1200_640x640_rgb.yaml](./yolov8n-seg_seg_c1200_640x640_rgb.yaml)
- `yolov8n-pose`: [yolov8n-pose_pose_c1200_640x640_rgb.yaml](./yolov8n-pose_pose_c1200_640x640_rgb.yaml)

## Output protocol

检测：按 stride 8/16/32 三层（legacy 为 reg/cls 成对 6 张量；fused 为 144 通道）；分割与姿态在检测头之外追加 mask 系数（32 维）与关键点（17 点 x3）张量。以 runtime/python/pybsnn_demo.py 实际解析为准。

See runtime/python/pybsnn_demo.py（det）/ pybsnn_seg_demo.py / pybsnn_pose_demo.py for the authoritative parsing.
