[English](./README.md) | 简体中文

# CondLaneNet 模型转换

本目录记录 CondLaneNet 在 Black Sesame 平台上的转换侧说明。模型转换分三步：导出 ONNX → 生成校准数据 → bstnnx_run 量化编译。

## 已发布模型

| 模型 | 输入 | 运行格式 |
|---|---|---|
| `condlanenet` | 3x320x800 RGB888P（已发布模型包） | `.bin` 模型包 |

## 三步转换流程

### 第 1 步：导出 ONNX

训练侧代码不随本仓库发布。上游实现：coremagi/CondLaneNet（实现基于 mmdetection v2.0.0，Apache-2.0）（https://github.com/coremagi/CondLaneNet）。

导出后请依据训练代码核对 ONNX 的输入形状、名称、颜色顺序和预处理。

已归档的 ONNX 与归档权重位于 `assets/`；各变体路径和 SHA-256 见[资产清单](../../../../docs/manifests/conversion_assets.yaml)。

### 第 2 步：准备校准数据

按照 YAML 中的目录准备有代表性的校准图像。`Quant.inputs` 中的颜色转换、mean 和 scale 来自旧工程的编译配置。

### 第 3 步：bstnnx_run 量化编译

使用 [BSTNNX 8.3.1 示例配置](./condlanenet_lane_c1200_800x320_bgr.yaml)。ONNX 已放在 `assets/condlanenet/`；请自行创建 `calibration_images/condlanenet/` 并放入有代表性的图片，从仓库根目录执行：

```bash
cd samples/vision/condlanenet/conversion
bstnnx_run --config condlanenet_lane_c1200_800x320_bgr.yaml
```

此配置是转换示例，不保证复现仓库中的已量化模型包。编译前需核对 ONNX 输入名称、形状、预处理及输出协议；已发布模型包可能由其他工具链版本生成，产物文件和目录结构由实际工具链决定。`../model/download.sh` 只检查仓库内模型文件是否齐全，不执行下载。

## 输出协议

车道线实例：行方向分类 + start/end offset（单条车道线输出 stride 分辨率上的 keypoints 集合）。以 runtime/python/pybsnn_demo.py 实际解析为准。

以 runtime/python/pybsnn_demo.py 的实际解析为准。

## 转换环境

bstnnx_run 编译工具链在 x86 Docker 中运行，不随本仓库发布，请从官方渠道获取对应版本工具链包。
