[English](./README.md) | 简体中文

# 行人重识别（MobileNetV2 / OSNet / ResNet-50）模型说明

本目录给出行人重识别（特征嵌入 + 成对余弦匹配）sample 在 Black Sesame Aura / Kalos 平台上的部署说明，
包括模型转换、模型下载、运行时推理与评测说明。

## 算法概述

- **任务**：行人重识别（成对余弦相似度匹配）
- **上游实现**：https://github.com/KaiyangZhou/deep-person-reid（Torchreid，MIT）

训练侧代码不随本仓库发布（许可与来源说明见 `conversion/README_cn.md`）。

## 支持的模型

| 模型 | 输入尺寸 | 嵌入维度 | Aura C1200 FPS | 推理耗时 |
|---|---|---|---|---|
| `mobilenetv2` | 128x256 | 1280 维 | 1785.71 | 0.560 ms |
| `osnet` | 128x256 | 1280 维 | 203.52 | 4.9135 ms |
| `resnet50` | 128x256 | 2048 维 | 469.04 | 2.132 ms |

Aura C1200 耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

## 目录结构

```text
.
├── conversion/                   # 模型转换说明（引用型）
├── evaluator/                    # 精度/性能评测说明
├── model/                        # 模型下载脚本与清单
│   └── download.sh
├── runtime/python/               # PyBSNN Python 推理
├── runtime/cpp/                  # C++ 推理（源码 + build.sh）
└── test_data/                    # 行人裁剪测试图
```

## 快速开始

```bash
# 1. 模型文件已随仓库发布（可选：完整性校验）
cd model && ./download.sh   # completeness check && cd ..

# 2. Python 推理（板端，需先执行仓库根 init_env.sh 安装 PyBSNN）
cd runtime/python
python3 pybsnn_demo.py

# 3. C++ 推理（板端 native 编译）；运行 True/False 成对测试
cd runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

## 评测说明

性能与精度口径见 [evaluator/README_cn.md](./evaluator/README_cn.md)。
