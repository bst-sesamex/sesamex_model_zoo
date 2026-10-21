[English](./README.md) | 简体中文

# SesameX Model Zoo

**版本：** 见 [VERSION](./VERSION) · **变更记录：** [CHANGELOG.md](./CHANGELOG.md)

SesameX Model Zoo 面向黑芝麻智能端侧平台（Aura / Kalos），提供主流视觉模型的
部署示例。每个 sample 覆盖板端完整链路：模型转换说明、量化模型下载、Python 与
C++ 推理运行时、精度/性能评测与验证数据。

核心流程：

```text
训练（上游仓库）-> 导出 ONNX -> 校准数据 -> 量化编译
      -> 板端 C++ / Python 推理 -> FPS 与精度验证
```

## 支持平台

| 平台 | 说明 |
|---|---|
| Aura（C1200 芯片）| 主交付平台。本版本 FPS 数据实测于 Aura。 |
| Kalos（A1000 芯片） | 全部 10 个 vision sample 均随仓发布 A1000 模型包（`model/a1000/`），详见各 sample 的 `model/README`。 |

## 仓库结构

```
sesamex-model-zoo/
├── samples/vision/        # 模型 sample，每模型家族一目录
├── utils/py_utils/        # 前/后处理与可视化公共库
├── utils/tools/           # 批量编译 / 压测 / 评测工具
├── datasets/              # 数据集准备脚本（不含数据本体）
├── docs/                  # 发布规范、manifest、catalog
└── resource/imgs/         # README 用图
```

## 模型索引

下表为 Aura/Kalos 平台算法示例；FPS 由预热一次后三轮 C++ 同步推理耗时的中位数计算。

### 视觉类

| 模型 | Sample | 任务 | Aura FPS | Aura 推理耗时 | Kalos FPS | Kalos 推理耗时 |
|---|---|---|---:|---:|---:|---:|
| LightStereo (OpenStereo-2) | [openstereo2](./samples/vision/openstereo2) | 双目立体匹配/深度 | 4.58 | 218.219 ms | 4.68 | 213.816 ms |
| ResNet-18 | [classification](./samples/vision/classification) | 图像分类 | 839.40 | 1.191 ms | 477.48 | 2.094 ms |
| ResNet-34 | [classification](./samples/vision/classification) | 图像分类 | 530.60 | 1.885 ms | 293.89 | 3.403 ms |
| MobileNetV2 | [classification](./samples/vision/classification) | 图像分类 | 1686.34 | 0.593 ms | 1077.59 | 0.928 ms |
| CondLaneNet | [condlanenet](./samples/vision/condlanenet) | 车道线检测 | 148.13 | 6.751 ms | 89.19 | 11.212 ms |
| FCOS-ResNet50 | [fcos](./samples/vision/fcos) | 目标检测 | 51.29 | 19.497 ms | 35.12 | 28.471 ms |
| YOLOv10n | [yolov12](./samples/vision/yolov12) | 目标检测 | 295.92 | 3.379 ms | 161.72 | 6.184 ms |
| YOLO11n | [yolov12](./samples/vision/yolov12) | 目标检测 | 307.47 | 3.252 ms | 162.50 | 6.154 ms |
| YOLOv12n | [yolov12](./samples/vision/yolov12) | 目标检测 | 42.81 | 23.358 ms | 33.15 | 30.162 ms |
| YOLOv8n-det | [yolov8](./samples/vision/yolov8) | 目标检测 | 423.01 | 2.364 ms | 216.34 | 4.622 ms |
| YOLOv8n-seg | [yolov8](./samples/vision/yolov8) | 实例分割 | 401.39 | 2.491 ms | 208.78 | 4.790 ms |
| YOLOv8n-pose | [yolov8](./samples/vision/yolov8) | 姿态估计 | 430.66 | 2.322 ms | 216.59 | 4.617 ms |
| YOLOv5m | [yolov5](./samples/vision/yolov5) | 目标检测 | 144.96 | 6.899 ms | 84.58 | 11.823 ms |
| 人脸关键点 56 | [landmark](./samples/vision/landmark) | 人脸关键点（68 点） | 2693.00 | 0.371 ms | 1777.25 | 0.563 ms |
| 人脸关键点 224 | [landmark](./samples/vision/landmark) | 人脸关键点（68 点） | 1717.23 | 0.582 ms | 1136.79 | 0.880 ms |
| ReID MobileNetV2 | [reid](./samples/vision/reid) | 行人重识别 | 2122.39 | 0.471 ms | 1033.24 | 0.968 ms |
| ReID OSNet | [reid](./samples/vision/reid) | 行人重识别 | 374.58 | 2.670 ms | 229.48 | 4.358 ms |
| ReID ResNet-50 | [reid](./samples/vision/reid) | 行人重识别 | 511.90 | 1.954 ms | 280.14 | 3.570 ms |
| UNet 512x512 | [unet-seg](./samples/vision/unet-seg) | 语义分割 | 35.06 | 28.524 ms | 22.28 | 44.882 ms |
| UNet 512x1024 | [unet-seg](./samples/vision/unet-seg) | 语义分割 | 18.71 | 53.446 ms | 10.62 | 94.119 ms |

## 快速开始

在 aarch64 板端（Aura / Kalos）：

```bash
# 1. 初始化环境（runtime 环境 + PyBSNN wheel）
chmod +x init_env.sh
./init_env.sh

# 2. 运行 Python 推理（板端，示例：yolov8）
cd samples/vision/yolov8/runtime/python
python3 pybsnn_demo.py

# 3. 运行 C++ 推理（板端 native 编译）
cd samples/vision/yolov8/runtime/cpp
bash build.sh && sh run_demo.sh
```

各 sample 内的 `evaluator/`、`utils/tools/` 与 `conversion/` 文档。

## Sample 结构

每个 sample 遵循统一布局：

```
samples/vision/<model>/
├── README.md / README_cn.md     # 总览 + 变体索引
├── conversion/                  # 已归档 ONNX/权重 + BSTNNX 8.3.1 YAML 示例
├── evaluator/                   # 精度/性能报告 + 脚本
├── runtime/python/              # Python 运行时（pybsnn_demo.py）
├── runtime/cpp/                 # C++ 运行时（如有）
└── test_data/                   # 输入图 + 板端输出结果图
```

产物命名：`{model}_{task}_{chip}_{WxH}_{inputfmt}.{ext}`，
如 `yolov8n_det_c1200_640x640_nv12.bin`。

## 依赖

- **BSNN runtime**（`libbsnnrt.so`）：随板端系统镜像提供。
- **PyBSNN**（Python 绑定，仅 aarch64）：随仓库发布于 `pybsnn/`（C1200 wheel）；
  `init_env.sh` 从仓库内安装。
- **模型权重**：量化模型包随仓库发布于各 sample 的 `model/` 目录（C1200 与
  Kalos/A1000）。
- **转换资产**：ONNX 与已归档训练权重存于各 sample 的 `conversion/assets/`，
  文件 SHA-256 见[资产清单](./docs/manifests/conversion_assets.yaml)；校准图像需自行提供。

## 参与贡献

新增 sample 的强制目录结构、命名规范与许可规则见
[CONTRIBUTING.md](./CONTRIBUTING.md)。

## 许可

本项目基于 [Apache License 2.0](./LICENSE) 开源。第三方资产保留其原始许可，
见各 sample 文档说明。
