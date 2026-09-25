English | [简体中文](./README_cn.md)

# OpenStereo-2（LightStereo） Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `lightstereo` | two inputs (left and right), each 3x544x960 RGB888P (published package) | `.bin` 模型包 |

## Step 1: Export ONNX

Upstream implementation: OpenStereo（许可与来源待二次审计；本版本不带训练代码与 .so） (https://github.com/XiandaGuo/OpenStereo).

Check the exported ONNX input shape, name, color order and preprocessing against your training code before compiling.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative, paired left and right images in the two folders named in the YAML. The `Quant.inputs` entries use the source project's RGB preprocessing.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./lightstereo_stereo_c1200_960x544_rgb.yaml) reads the included ONNX from `assets/lightstereo/`. Put representative images in `calibration_images/lightstereo/` (create the folder) and run from the repository root:

```bash
cd samples/vision/openstereo2/conversion
bstnnx_run --config lightstereo_stereo_c1200_960x544_rgb.yaml
```

This is a conversion example, not a reproducible recipe for the committed quantized package. Check the exported ONNX input names, shapes, preprocessing and output protocol before compiling. Published packages may have been built with a different toolchain version. The generated files and layout depend on the compiler version. `../model/download.sh` checks that the committed package files exist; it performs no download.

## Output protocol

视差图 disp（960x544，float），后处理含 disp_gt 对比与异常值剔除。以 runtime/python/pybsnn_demo.py 与 runtime/cpp/src/model_meta.h 实际解析为准。

See runtime/python/pybsnn_demo.py for the authoritative parsing.
