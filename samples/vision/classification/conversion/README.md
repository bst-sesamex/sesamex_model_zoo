English | [简体中文](./README_cn.md)

# Image Classification（ResNet-34/34、MobileNetV2） Model Conversion

Three steps: export ONNX -> generate calibration data -> quantized compile with
bstnnx_run. Training-side code is not part of this repository.

## Published models

| Model | Input | Format |
|---|---|---|
| `resnet18` | 3x224x224 RGB888P (published package) | `.bin` 模型包 |
| `resnet34` | 3x224x224 RGB888P (published package) | `.bin` 模型包 |
| `mobilenetv2` | 3x224x224 RGB888P (published package) | `.bin` 模型包 |

## Step 1: Export ONNX

Upstream implementation: 自研模型定义（无外部 license 约束） (./conversion/).

Check the exported ONNX input shape, name, color order and preprocessing against your training code before compiling.

Only ONNX files are available for the three classification variants; the source tree has no matching `.pth` checkpoints.

## Step 2: Calibration data

Place representative calibration images in the folder named in the YAML. `Quant.inputs` uses the source project's image color conversion, mean and scale.

The archived ResNet-34 compiler YAML uses CIFAR-10 calibration and mistakenly sets `model_name: fcos`; this example corrects the name.

## Step 3: Quantized compile (bstnnx_run)

The [BSTNNX 8.3.1 example](./resnet34_cls_c1200_224x224_bgr.yaml) reads the included ONNX from `assets/resnet34/`. Put representative images in `calibration_images/resnet34/` (create the folder) and run from the repository root:

```bash
cd samples/vision/classification/conversion
bstnnx_run --config resnet34_cls_c1200_224x224_bgr.yaml
```

This is a conversion example, not a reproducible recipe for the committed quantized package. Check the exported ONNX input names, shapes, preprocessing and output protocol before compiling. Published packages may have been built with a different toolchain version. The generated files and layout depend on the compiler version. `../model/download.sh` checks that the committed package files exist; it performs no download.

## Variant examples

- `resnet18`: [resnet18_cls_c1200_224x224_bgr.yaml](./resnet18_cls_c1200_224x224_bgr.yaml)
- `resnet34`: [resnet34_cls_c1200_224x224_bgr.yaml](./resnet34_cls_c1200_224x224_bgr.yaml)
- `mobilenetv2`: [mobilenetv2_cls_c1200_224x224_bgr.yaml](./mobilenetv2_cls_c1200_224x224_bgr.yaml)

## Output protocol

CIFAR-10 10 类 logits（后处理 softmax 取 Top-K）；已发布模型包的输出维度为 10。分类模型定义脚本见原工程 `models/classification/train/`。

See runtime/python/pybsnn_demo.py for the authoritative parsing.
