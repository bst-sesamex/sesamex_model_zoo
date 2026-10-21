[English](./README.md) | 简体中文

# FCOS 模型转换

本目录记录 FCOS 在 Black Sesame 平台上的转换侧说明。模型转换分三步：导出 ONNX → 生成校准数据 → bstnnx_run 量化编译。

## 已发布模型

| 模型 | 输入 | 运行格式 |
|---|---|---|
| `fcos` | 3x896x896 RGB888P（已发布模型包） | `.bin` 模型包 |

## 三步转换流程

### 第 1 步：导出 ONNX

训练侧代码不随本仓库发布。上游实现：tianzhi0549/FCOS（non-commercial）（https://github.com/tianzhi0549/FCOS）。

导出后请依据训练代码核对 ONNX 的输入形状、名称、颜色顺序和预处理。

已归档的 ONNX 与归档权重位于 `assets/`；各变体路径和 SHA-256 见[资产清单](../../../../docs/manifests/conversion_assets.yaml)。

### 第 2 步：准备校准数据

按照 YAML 中的目录准备有代表性的校准图像。`Quant.inputs` 定义校准时的颜色转换、mean 和 scale。

### 第 3 步：bstnnx_run 量化编译

使用 [BSTNNX 8.3.1 示例配置](./fcos_det_c1200_896x896_bgr.yaml)。ONNX 已放在 `assets/fcos/`；请自行创建 `calibration_images/fcos/` 并放入有代表性的图片，从仓库根目录执行：

```bash
cd samples/vision/fcos/conversion
bstnnx_run --config fcos_det_c1200_896x896_bgr.yaml
```

## 输出协议

15 个输出张量：3 个特征层级（stride 8/16/32）各 cls(80)/box(4)/ctr(1)。板端日志示例：Level0 112x112, cls_pitch=80, box_pitch=16, ctr_pitch=16。以 runtime/python/pybsnn_demo.py 实际解析为准。

以 runtime/python/pybsnn_demo.py 的实际解析为准。

## 转换环境

bstnnx_run 编译工具链在 x86 Docker 中运行，不随本仓库发布，请从官方渠道获取对应版本工具链包。
