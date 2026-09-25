[English](./README.md) | 简体中文

# 行人重识别模型转换

本目录记录行人重识别在 Black Sesame 平台上的转换侧说明。模型转换分三步：导出 ONNX → 生成校准数据 → bstnnx_run 量化编译。

## 已发布模型

| 模型 | 输入 | 嵌入维度 | 运行格式 |
|---|---|---|---|
| `mobilenetv2` | 3x256x128 RGB888P（已发布模型包） | 1280 维 | `.bin` 模型包 |
| `osnet` | 3x256x128 RGB888P（已发布模型包） | 1280 维 | `.bin` 模型包 |
| `resnet50` | 3x256x128 RGB888P（已发布模型包） | 2048 维 | `.bin` 模型包 |

## 三步转换流程

### 第 1 步：导出 ONNX

上游实现：[Torchreid](https://github.com/KaiyangZhou/deep-person-reid)（MIT 许可）。
部署骨干为 Torchreid model zoo 的 `mobilenetv2_x1_0`、`osnet_x1_0`、`resnet50`；
以输入 shape `3x256x128`、开启特征池化用 `torch.onnx.export` 导出。


已归档的 ONNX 与归档权重位于 `assets/`；各变体路径和 SHA-256 见[资产清单](../../../../docs/manifests/conversion_assets.yaml)。

### 第 2 步：准备校准数据

按照 YAML 中的目录准备有代表性的校准图像。`Quant.inputs` 中的颜色转换、mean 和 scale 来自旧工程的编译配置。

旧工程编译配置只将模型命名为 `reid`，没有注明骨干网络；用于 OSNet 变体前请核对 ONNX。

### 第 3 步：bstnnx_run 量化编译

使用 [BSTNNX 8.3.1 示例配置](./reid_osnet_c1200_128x256_rgb.yaml)。ONNX 已放在 `assets/osnet/`；请自行创建 `calibration_images/osnet/` 并放入有代表性的图片，从仓库根目录执行：

```bash
cd samples/vision/reid/conversion
bstnnx_run --config reid_osnet_c1200_128x256_rgb.yaml
```

此配置是转换示例，不保证复现仓库中的已量化模型包。编译前需核对 ONNX 输入名称、形状、预处理及输出协议；已发布模型包可能由其他工具链版本生成，产物文件和目录结构由实际工具链决定。`../model/download.sh` 只检查仓库内模型文件是否齐全，不执行下载。

## 各变体示例

- `osnet`: [reid_osnet_c1200_128x256_rgb.yaml](./reid_osnet_c1200_128x256_rgb.yaml)
- `mobilenetv2`: [reid_mobilenetv2_c1200_128x256_rgb.yaml](./reid_mobilenetv2_c1200_128x256_rgb.yaml)
- `resnet50`: [reid_resnet50_c1200_128x256_rgb.yaml](./reid_resnet50_c1200_128x256_rgb.yaml)

## 输出协议

每张图输出一条特征嵌入向量（1280 维 / 2048 维，SDK 反量化后为 float32）。
匹配由应用侧做余弦距离；C++ demo 使用的阈值：mobilenetv2 0.3193、osnet 0.3677、resnet50 0.2576（板端实测标定）。
