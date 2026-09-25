[English](./README.md) | 简体中文

# FCOS 模型说明

本目录给出 FCOS sample 在 Black Sesame Aura / Kalos 平台上的部署说明，包括模型转换、
模型下载、运行时推理与评测说明。

## 算法概述

- **任务**：目标检测（anchor-free）
- **论文**：https://arxiv.org/pdf/1904.01355.pdf
- **官方实现**：tianzhi0549/FCOS (non-commercial) —— https://github.com/tianzhi0549/FCOS

训练侧代码不随本仓库发布（许可与来源说明见 `conversion/README_cn.md`）。

## 支持的模型

| 模型 | 输入尺寸 | 任务 | Aura C1200 FPS | 推理耗时 |
|---|---|---|---|---|
| `fcos` | 896x896 | 目标检测 | 50.79 | 19.689 ms |

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
└── test_data/                    # 测试图与板端输出结果图
```

## 快速开始

```bash
# 1. 模型文件已随仓库发布（可选：完整性校验）
cd model && ./download.sh   # completeness check && cd ..

# 2. Python 推理（板端，需先执行仓库根 init_env.sh 安装 PyBSNN）
cd runtime/python
python3 pybsnn_demo.py

# 3. C++ 推理（板端 native 编译）
cd runtime/cpp && bash build.sh
```

测试图与结果对照见 [test_data/](./test_data/)。

## 评测说明

性能与精度口径见 [evaluator/README_cn.md](./evaluator/README_cn.md)。
