English | [简体中文](./README_cn.md)

# Person Re-Identification Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Embedding | Format |
|---|---|---|---|
| `mobilenetv2` | 3x256x128 RGB888P (published package) | 1280-d | `.bin` model package |
| `osnet` | 3x256x128 RGB888P (published package) | 1280-d | `.bin` model package |
| `resnet50` | 3x256x128 RGB888P (published package) | 2048-d | `.bin` model package |

## Step 1: Export ONNX

Upstream implementation: [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
(MIT license). The deployed backbones are `mobilenetv2_x1_0`, `osnet_x1_0` and
`resnet50` in Torchreid's model zoo; export with `torch.onnx.export` at input
shape `3x256x128` and feature pooling enabled.

The ONNX and archived checkpoint are included under `assets/`; see the [asset manifest](../../../../docs/manifests/conversion_assets.yaml) for all variants and checksums.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` defines the image color conversion, mean and scale used at calibration time.

The archived compiler YAML names the model `reid` without identifying the backbone; confirm the ONNX is OSNet before using this variant example.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./reid_osnet_c1200_128x256_rgb.yaml) reads the included ONNX from `assets/osnet/`. Put representative images in `calibration_images/osnet/` (create the folder) and run from the repository root:

```bash
cd samples/vision/reid/conversion
bstnnx_run --config reid_osnet_c1200_128x256_rgb.yaml
```

## Variant examples

- `osnet`: [reid_osnet_c1200_128x256_rgb.yaml](./reid_osnet_c1200_128x256_rgb.yaml)
- `mobilenetv2`: [reid_mobilenetv2_c1200_128x256_rgb.yaml](./reid_mobilenetv2_c1200_128x256_rgb.yaml)
- `resnet50`: [reid_resnet50_c1200_128x256_rgb.yaml](./reid_resnet50_c1200_128x256_rgb.yaml)

## Output protocol

Single feature embedding vector per image (1280-d / 2048-d, post-quantization
dequantized by the SDK). Matching is cosine distance in the consumer
application; thresholds used by the C++ demo: mobilenetv2 0.3193, osnet 0.3677,
resnet50 0.2576 (板端实测标定).

See runtime/python/pybsnn_demo.py for the authoritative parsing.
