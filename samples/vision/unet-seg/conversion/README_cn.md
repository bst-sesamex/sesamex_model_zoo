[English](./README.md) | 简体中文

# 语义分割（UNet）模型转换

本目录记录语义分割在 Black Sesame 平台上的转换侧说明。模型转换分三步：导出 ONNX → 生成校准数据 → bstnnx_run 量化编译。

## 已发布模型

| 模型 | 输入 | 输出 | 运行格式 |
|---|---|---|---|
| `unet_512x512` | 3x512x512 RGB888P（已发布模型包） | 19x512x512 logits | `.bin` 模型包 |
| `unet_512x1024` | 3x512x1024 RGB888P（已发布模型包） | 19x512x1024 logits | `.bin` 模型包 |

## 三步转换流程

### 第 1 步：导出 ONNX

上游实现：[Efficient-Segmentation-Networks](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)
（基于 ESPNet-v2 的 UNet，MIT 许可）。上游项目自带导出脚本（`export_unet_onnx.py`）；
以固定输入 shape `3x512x512` / `3x512x1024`（校准图像采用 BGR，mean/scale 见 YAML）导出。


已归档的 ONNX 与归档权重位于 `assets/`；各变体路径和 SHA-256 见[资产清单](../../../../docs/manifests/conversion_assets.yaml)。

### 第 2 步：准备校准数据

按照 YAML 中的目录准备有代表性的校准图像。`Quant.inputs` 中的颜色转换、mean 和 scale 来自旧工程的编译配置。

旧工程分割配置的模型名和路径误写为 `reid`；本示例仅沿用其中的图像预处理参数。

### 第 3 步：bstnnx_run 量化编译

使用 [BSTNNX 8.3.1 示例配置](./unet_seg_c1200_512x512_rgb.yaml)。ONNX 已放在 `assets/unet_512x512/`；请自行创建 `calibration_images/unet_512x512/` 并放入有代表性的图片，从仓库根目录执行：

```bash
cd samples/vision/unet-seg/conversion
bstnnx_run --config unet_seg_c1200_512x512_rgb.yaml
```

此配置是转换示例，不保证复现仓库中的已量化模型包。编译前需核对 ONNX 输入名称、形状、预处理及输出协议；已发布模型包可能由其他工具链版本生成，产物文件和目录结构由实际工具链决定。`../model/download.sh` 只检查仓库内模型文件是否齐全，不执行下载。

## 各变体示例

- `unet_512x512`: [unet_seg_c1200_512x512_rgb.yaml](./unet_seg_c1200_512x512_rgb.yaml)
- `unet_512x1024`: [unet_seg_c1200_1024x512_rgb.yaml](./unet_seg_c1200_1024x512_rgb.yaml)

## 输出协议

19 类逐像素 logits（`19xHxW`，INT8，反量化 scale 0.250）；后处理取 argmax 得到类别索引图。
C++ demo 将分割掩码叠加到输入图（`output/inference_*.jpg`）；Python demo 保存 argmax 灰度图（`pybsnn_seg_result.png`）。
