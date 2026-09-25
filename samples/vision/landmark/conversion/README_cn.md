[English](./README.md) | 简体中文

# 人脸关键点检测模型转换

本目录记录人脸关键点检测在 Black Sesame 平台上的转换侧说明。模型转换分三步：导出 ONNX → 生成校准数据 → bstnnx_run 量化编译。

## 已发布模型

| 模型 | 输入 | 运行格式 |
|---|---|---|
| `size56` | 1x3x56x56 RGB888P（已发布模型包） | `.bin` 模型包 |
| `size224` | 1x3x224x224 RGB888P（已发布模型包） | `.bin` 模型包 |

## 三步转换流程

### 第 1 步：导出 ONNX

上游实现：[pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)（68 点模型，部署变体使用坐标回归头）。

导出后请依据训练代码核对 ONNX 的输入形状、名称、颜色顺序和预处理。

已归档的 ONNX 与归档权重位于 `assets/`；各变体路径和 SHA-256 见[资产清单](../../../../docs/manifests/conversion_assets.yaml)。

### 第 2 步：准备校准数据

按照 YAML 中的目录准备有代表性的校准图像。`Quant.inputs` 中的颜色转换、mean 和 scale 来自旧工程的编译配置。

### 第 3 步：bstnnx_run 量化编译

使用 [BSTNNX 8.3.1 示例配置](./landmark_size56_c1200_56x56_rgb.yaml)。ONNX 已放在 `assets/size56/`；请自行创建 `calibration_images/size56/` 并放入有代表性的图片，从仓库根目录执行：

```bash
cd samples/vision/landmark/conversion
bstnnx_run --config landmark_size56_c1200_56x56_rgb.yaml
```

此配置是转换示例，不保证复现仓库中的已量化模型包。编译前需核对 ONNX 输入名称、形状、预处理及输出协议；已发布模型包可能由其他工具链版本生成，产物文件和目录结构由实际工具链决定。`../model/download.sh` 只检查仓库内模型文件是否齐全，不执行下载。

## 各变体示例

- `size56`: [landmark_size56_c1200_56x56_rgb.yaml](./landmark_size56_c1200_56x56_rgb.yaml)
- `size224`: [landmark_size224_c1200_224x224_rgb.yaml](./landmark_size224_c1200_224x224_rgb.yaml)

## 输出协议

68 个关键点的归一化坐标，扁平化为 136 维向量 `[x0, y0, x1, y1, ...]`（值域 [0,1]）。
runtime/cpp 的 `Landmark Result Summary` 与 runtime/python 的 `postprocess` 为权威解析实现。
