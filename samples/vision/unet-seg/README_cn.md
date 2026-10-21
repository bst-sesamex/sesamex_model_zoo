[English](./README.md) | 简体中文

# 语义分割（UNet，ESPNet-v2 骨干）模型说明

本目录给出实时语义分割 sample 在 Black Sesame Aura / Kalos 平台上的部署说明，包括模型转换、
模型下载、运行时推理与评测说明。

## 算法概述

- **任务**：语义分割（19 类，CamVid 协议）
- **上游实现**：https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks（ESPNet-v2）

训练侧代码不随本仓库发布（许可与来源说明见 `conversion/README_cn.md`）。

## 支持的模型

| 模型 | 输入尺寸 | 输出 | Aura C1200 FPS | 推理耗时 |
|---|---|---|---|---|
| `unet_512x512` | 512x512 | 512x512x19 | 35.51 | 28.158 ms |
| `unet_512x1024` | 512x1024 | 512x1024x19 | 18.15 | 55.111 ms |

Aura C1200 耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

## 目录结构

```text
.
├── conversion/                   # 模型转换说明（引用型）
├── evaluator/                    # 精度/性能评测说明
├── model/                        # 模型下载脚本与清单
├── runtime/python/               # PyBSNN Python 推理
├── runtime/cpp/                  # C++ 推理（源码 + build.sh）
└── test_data/                    # 测试图与板端输出结果图
```

## 快速开始

```bash
# 1. Python 推理（板端，需先执行仓库根 init_env.sh 安装 PyBSNN）
cd runtime/python
python3 pybsnn_demo.py

# 2. C++ 推理（板端 native 编译）；两种输入尺寸均会运行
cd runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

## 评测说明

性能与精度口径见 [evaluator/README_cn.md](./evaluator/README_cn.md)。
