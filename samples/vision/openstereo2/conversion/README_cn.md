[English](./README.md) | 简体中文

# OpenStereo-2（LightStereo） 模型转换

本目录记录 OpenStereo-2（LightStereo） 在 Black Sesame 平台上的转换侧说明。模型转换分三步：导出 ONNX → 生成校准数据 → bstnnx_run 量化编译。

## 已发布模型

| 模型 | 输入 | 运行格式 |
|---|---|---|
| `lightstereo` | 双目两路输入：左目 + 右目，各 3x544x960 RGB888P（已发布模型包） | `.bin` 模型包 |

## 三步转换流程

### 第 1 步：导出 ONNX

训练侧代码不随本仓库发布。上游实现：OpenStereo（许可与来源待二次审计；本版本不带训练代码与 .so）（https://github.com/XiandaGuo/OpenStereo）。

导出后请依据训练代码核对 ONNX 的输入形状、名称、颜色顺序和预处理。

已归档的 ONNX 与归档权重位于 `assets/`；各变体路径和 SHA-256 见[资产清单](../../../../docs/manifests/conversion_assets.yaml)。

### 第 2 步：准备校准数据

按 YAML 中的两个目录分别准备配对的左右图像。`Quant.inputs` 中的 RGB 预处理参数来自旧工程的编译配置。

### 第 3 步：bstnnx_run 量化编译

使用 [BSTNNX 8.3.1 示例配置](./lightstereo_stereo_c1200_960x544_rgb.yaml)。ONNX 已放在 `assets/lightstereo/`；请自行创建 `calibration_images/lightstereo/` 并放入有代表性的图片，从仓库根目录执行：

```bash
cd samples/vision/openstereo2/conversion
bstnnx_run --config lightstereo_stereo_c1200_960x544_rgb.yaml
```

此配置是转换示例，不保证复现仓库中的已量化模型包。编译前需核对 ONNX 输入名称、形状、预处理及输出协议；已发布模型包可能由其他工具链版本生成，产物文件和目录结构由实际工具链决定。`../model/download.sh` 只检查仓库内模型文件是否齐全，不执行下载。

## 输出协议

视差图 disp（960x544，float），后处理含 disp_gt 对比与异常值剔除。以 runtime/python/pybsnn_demo.py 与 runtime/cpp/src/model_meta.h 实际解析为准。

以 runtime/python/pybsnn_demo.py 的实际解析为准。

## 转换环境

bstnnx_run 编译工具链在 x86 Docker 中运行，不随本仓库发布，请从官方渠道获取对应版本工具链包。
