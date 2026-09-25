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
| Kalos（A1000 芯片）| 部分 sample 提供 A1000 芯片型号的模型，见各 sample 的 `model/README`。 |

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

下表为 Aura C1200 算法示例；FPS 由预热一次后三轮 C++ 同步推理耗时的中位数计算。

### 视觉类

| 模型 | Sample | 任务 | Aura C1200 FPS | 推理耗时 |
|---|---|---|---:|---:|
| LightStereo (OpenStereo-2) | [openstereo2](./samples/vision/openstereo2) | 双目立体匹配/深度 | 14.64 | 68.291 ms |
| ResNet-18 | [classification](./samples/vision/classification) | 图像分类 | 763.36 | 1.310 ms |
| ResNet-34 | [classification](./samples/vision/classification) | 图像分类 | 490.44 | 2.039 ms |
| MobileNetV2 | [classification](./samples/vision/classification) | 图像分类 | 1464.13 | 0.683 ms |
| CondLaneNet | [condlanenet](./samples/vision/condlanenet) | 车道线检测 | 148.54 | 6.732 ms |
| FCOS-ResNet50 | [fcos](./samples/vision/fcos) | 目标检测 | 50.79 | 19.689 ms |
| YOLOv10n | [yolov12](./samples/vision/yolov12) | 目标检测 | 207.13 | 4.828 ms |
| YOLO11n | [yolov12](./samples/vision/yolov12) | 目标检测 | 218.20 | 4.583 ms |
| YOLOv12n | [yolov12](./samples/vision/yolov12) | 目标检测 | 39.95 | 25.032 ms |
| YOLOv8n-det | [yolov8](./samples/vision/yolov8) | 目标检测 | 392.46 | 2.548 ms |
| YOLOv8n-seg | [yolov8](./samples/vision/yolov8) | 实例分割 | 370.23 | 2.701 ms |
| YOLOv8n-pose | [yolov8](./samples/vision/yolov8) | 姿态估计 | 399.36 | 2.504 ms |
| YOLOv5m | [yolov5](./samples/vision/yolov5) | 目标检测 | 145.84 | 6.857 ms |
| 人脸关键点 56 | [landmark](./samples/vision/landmark) | 人脸关键点（68 点） | 2403.85 | 0.416 ms |
| 人脸关键点 224 | [landmark](./samples/vision/landmark) | 人脸关键点（68 点） | 1506.02 | 0.664 ms |
| ReID MobileNetV2 | [reid](./samples/vision/reid) | 行人重识别 | 1785.71 | 0.560 ms |
| ReID OSNet | [reid](./samples/vision/reid) | 行人重识别 | 203.52 | 4.9135 ms |
| ReID ResNet-50 | [reid](./samples/vision/reid) | 行人重识别 | 469.04 | 2.132 ms |
| UNet 512x512 | [unet-seg](./samples/vision/unet-seg) | 语义分割 | 35.51 | 28.158 ms |
| UNet 512x1024 | [unet-seg](./samples/vision/unet-seg) | 语义分割 | 18.15 | 55.111 ms |

## 快速开始

在 aarch64 板端（Aura）：

```bash
# 1. 初始化环境（runtime 环境 + PyBSNN wheel）
chmod +x init_env.sh
./init_env.sh

# 2. 校验模型包完整性（模型已随仓库发布，示例：yolov8）
cd samples/vision/yolov8
./model/download.sh

# 3. 运行 Python 推理（板端）
cd runtime/python
python3 pybsnn_demo.py
```

模型包已随仓库发布，`download.sh` 做完整性校验。精度评测、批量压测与模型转换说明见
各 sample 内的 `evaluator/`、`utils/tools/` 与 `conversion/` 文档。

## Sample 结构

每个 sample 遵循统一布局：

```
samples/vision/<model>/
├── README.md / README_cn.md     # 总览 + 变体索引
├── conversion/                  # 已归档 ONNX/权重 + BSTNNX 8.3.1 YAML 示例
├── evaluator/                   # 精度/性能报告 + 脚本
├── model/                       # download.sh + 模型清单
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
- **模型权重**：C1200 量化模型包随仓库发布于各 sample 的 `model/c1200/`；
  `model/download.sh` 做完整性校验。
- **转换资产**：ONNX 与已归档训练权重通过 Git LFS 存于各 sample 的 `conversion/assets/`，文件 SHA-256 见[资产清单](./docs/manifests/conversion_assets.yaml)；校准图像需自行提供。

## 参与贡献

新增 sample 的强制目录结构、命名规范与许可规则见
[CONTRIBUTING.md](./CONTRIBUTING.md)。

## 许可

本项目基于 [Apache License 2.0](./LICENSE) 开源。第三方资产保留其原始许可，
见各 sample 文档说明。
