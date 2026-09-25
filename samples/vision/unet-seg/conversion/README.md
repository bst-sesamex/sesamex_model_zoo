English | [简体中文](./README_cn.md)

# Semantic Segmentation (UNet) Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Output | Format |
|---|---|---|---|
| `unet_512x512` | 3x512x512 RGB888P (published package) | 19x512x512 logits | `.bin` model package |
| `unet_512x1024` | 3x512x1024 RGB888P (published package) | 19x512x1024 logits | `.bin` model package |

## Step 1: Export ONNX

Upstream implementation: [Efficient-Segmentation-Networks](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)
(ESPNet-v2 based UNet, MIT license). The repo also ships an export helper
(`export_unet_onnx.py`) in the upstream project; export with fixed input shapes
`3x512x512` / `3x512x1024` (calibration images are BGR and use the mean/scale in the YAML).

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` uses the source project's image color conversion, mean and scale.

The archived segmentation config contains stale `reid` model/path fields; this example reuses only its image preprocessing.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./unet_seg_c1200_512x512_rgb.yaml) reads the included ONNX from `assets/unet_512x512/`. Put representative images in `calibration_images/unet_512x512/` (create the folder) and run from the repository root:

```bash
cd samples/vision/unet-seg/conversion
bstnnx_run --config unet_seg_c1200_512x512_rgb.yaml
```

This is a conversion example, not a reproducible recipe for the committed quantized package. Check the exported ONNX input names, shapes, preprocessing and output protocol before compiling. Published packages may have been built with a different toolchain version. The generated files and layout depend on the compiler version. `../model/download.sh` checks that the committed package files exist; it performs no download.

## Variant examples

- `unet_512x512`: [unet_seg_c1200_512x512_rgb.yaml](./unet_seg_c1200_512x512_rgb.yaml)
- `unet_512x1024`: [unet_seg_c1200_1024x512_rgb.yaml](./unet_seg_c1200_1024x512_rgb.yaml)

## Output protocol

Per-pixel logits over 19 classes (`19xHxW`, INT8 with dequant scale 0.250);
postprocess takes argmax to a class-index map. The C++ demo overlays the mask
on the input image (`output/inference_*.jpg`); the Python demo saves an
argmax grayscale map (`pybsnn_seg_result.png`).

See runtime/python/pybsnn_demo.py for the authoritative parsing.
