[English](./README.md) | 简体中文

# OpenStereo-2 (LightStereo) 模型说明

本目录给出 OpenStereo-2 (LightStereo) sample 在 Black Sesame Aura / Kalos 平台上的部署说明，包括模型转换、
模型下载、运行时推理与评测说明。

## 算法概述

- **任务**：双目立体匹配 / 深度估计
- **论文**：LightStereo: Simplifying and Strengthening Stereo Matching
- **官方实现**：OpenStereo（来源与许可需二次审计，本版本不带训练代码） —— https://github.com/XiandaGuo/OpenStereo

训练侧代码不随本仓库发布（许可与来源说明见 `conversion/README_cn.md`）。

## 支持的模型

| 模型 | 输入尺寸 | 任务 | Aura C1200 FPS | 推理耗时 |
|---|---|---|---|---|
| `lightstereo` | 960x544 | 立体匹配 | 4.59 | 217.787 ms |

Aura C1200 耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

## 目录结构

```text
.
├── conversion/                   # 模型转换说明（引用型）
├── evaluator/                    # 精度/性能评测说明
├── model/                        # 模型下载脚本与清单
├── runtime/python/               # PyBSNN Python 推理
├── runtime/cpp/                  # C++ 推理（源码 + build.sh）
└── test_data/                    # 测试图、mask 与板端输出结果图
```

## 快速开始

```bash
# 1. Python 推理（板端，需先执行仓库根 init_env.sh 安装 PyBSNN）
cd runtime/python
python3 pybsnn_demo.py

# 2. C++ 推理（板端 native 编译）
cd runtime/cpp && bash build.sh && sh run_demo.sh
```

测试图与结果对照见 [test_data/](./test_data/)。

## 结果图口径

`test_data/0326_left_disparity.jpg` 为随仓图对在板端 C++ 链路的输出（生成时应用了
随仓的 `test_data/mask.bin`）。**白色像素为评测 mask 之外的区域**：一是超出模型视差
量程的部分（模型视差上限 `max_disp = 192` px，本场景底部 rows 的超近距地面/物体无法
表达），二是无标注像素。`run_demo.sh` 会自动加载 `test_data/mask.bin`；如需查看
原始色图，加 `-no-mask`（或移除该文件）即可。

## 评测说明

性能与精度口径见 [evaluator/README_cn.md](./evaluator/README_cn.md)。
