English | [简体中文](./README_cn.md)

# FCOS Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `fcos` | 3x896x896 RGB888P (published package) | `.bin` 模型包 |

## Step 1: Export ONNX

Upstream implementation: tianzhi0549/FCOS（non-commercial） (https://github.com/tianzhi0549/FCOS).

Check the exported ONNX input shape, name, color order and preprocessing against your training code before compiling.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` uses the source project's image color conversion, mean and scale.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./fcos_det_c1200_896x896_bgr.yaml) reads the included ONNX from `assets/fcos/`. Put representative images in `calibration_images/fcos/` (create the folder) and run from the repository root:

```bash
cd samples/vision/fcos/conversion
bstnnx_run --config fcos_det_c1200_896x896_bgr.yaml
```

This is a conversion example, not a reproducible recipe for the committed quantized package. Check the exported ONNX input names, shapes, preprocessing and output protocol before compiling. Published packages may have been built with a different toolchain version. The generated files and layout depend on the compiler version. `../model/download.sh` checks that the committed package files exist; it performs no download.

## Output protocol

15 个输出张量：3 个特征层级（stride 8/16/32）各 cls(80)/box(4)/ctr(1)。板端日志示例：Level0 112x112, cls_pitch=80, box_pitch=16, ctr_pitch=16。以 runtime/python/pybsnn_demo.py 实际解析为准。

See runtime/python/pybsnn_demo.py for the authoritative parsing.
