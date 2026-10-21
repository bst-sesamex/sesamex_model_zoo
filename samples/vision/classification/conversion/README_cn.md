[English](./README.md) | 简体中文

# Image Classification（ResNet-18/34、MobileNetV2） 模型转换

本目录记录 Image Classification（ResNet-18/34、MobileNetV2） 在 Black Sesame 平台上的转换侧说明。模型转换分三步：导出 ONNX → 生成校准数据 → bstnnx_run 量化编译。

## 已发布模型

| 模型 | 输入 | 运行格式 |
|---|---|---|
| `resnet18` | 3x224x224 RGB888P（已发布模型包） | `.bin` 模型包 |
| `resnet34` | 3x224x224 RGB888P（已发布模型包） | `.bin` 模型包 |
| `mobilenetv2` | 3x224x224 RGB888P（已发布模型包） | `.bin` 模型包 |

## 三步转换流程

### 第 1 步：导出 ONNX

分类模型结构以本仓库 `conversion/assets/` 下的 ONNX 为准。

导出后请依据训练代码核对 ONNX 的输入形状、名称、颜色顺序和预处理。

三种分类变体仅随仓库提供 ONNX（无配对 `.pth`）。

### 第 2 步：准备校准数据

按照 YAML 中的目录准备有代表性的校准图像。`Quant.inputs` 定义校准时的颜色转换、mean 和 scale。

ResNet-34 使用 CIFAR-10 校准图像，`model_name: resnet34`。

### 第 3 步：bstnnx_run 量化编译

使用 [BSTNNX 8.3.1 示例配置](./resnet34_cls_c1200_224x224_bgr.yaml)。ONNX 已放在 `assets/resnet34/`；请自行创建 `calibration_images/resnet34/` 并放入有代表性的图片，从仓库根目录执行：

```bash
cd samples/vision/classification/conversion
bstnnx_run --config resnet34_cls_c1200_224x224_bgr.yaml
```

## 各变体示例

- `resnet18`: [resnet18_cls_c1200_224x224_bgr.yaml](./resnet18_cls_c1200_224x224_bgr.yaml)
- `resnet34`: [resnet34_cls_c1200_224x224_bgr.yaml](./resnet34_cls_c1200_224x224_bgr.yaml)
- `mobilenetv2`: [mobilenetv2_cls_c1200_224x224_bgr.yaml](./mobilenetv2_cls_c1200_224x224_bgr.yaml)

## 输出协议

CIFAR-10 10 类 logits（后处理 softmax 取 Top-K）；已发布模型包的输出维度为 10。分类模型结构以本仓库 `conversion/assets/` 下的 ONNX 为准。

以 runtime/python/pybsnn_demo.py 的实际解析为准。

## 转换环境

bstnnx_run 编译工具链在 x86 Docker 中运行，不随本仓库发布，请从官方渠道获取对应版本工具链包。
