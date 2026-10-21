[English](./README.md) | 简体中文

# YOLOv12 / YOLO11 / YOLOv10 模型评测

本目录记录 YOLOv12 / YOLO11 / YOLOv10 在 Aura / Kalos 平台上的性能与精度口径。

## 1. 评测前提

- 板端：Aura（C1200 芯片）或 Kalos（A1000 芯片），系统已含 `libbsnnrt.so`
- Python 3 + `pybsnn`（板端执行仓库根 `init_env.sh` 自动安装）
- 精度评测需：Python 3.8+、`pycocotools`（检测类）或对应数据集评测工具

## 2. 数据集说明

- 评测数据集：COCO val2017
- 快速验证图片：`../test_data/`（单张冒烟）
- 完整数据集不入仓，见仓库根 `datasets/` 的准备脚本

## 3. 使用方法

单张冒烟（板端）：

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

批量性能（板端，全部模型）：

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. 性能 Benchmark

测试条件：Aura C1200，默认 auto 引擎；预热一次后三轮 C++ run+wait 单调时钟耗时中位数，不含前后处理。

| 模型 | 输入 | Aura C1200 FPS | Aura 耗时 |
|---|---|---|---|
| `yolov10n-det` | 640x640 | 207.13 FPS | 4.828 ms |
| `yolov11n-det` | 640x640 | 218.20 FPS | 4.583 ms |
| `yolov12n-det` | 640x640 | 39.95 FPS | 25.032 ms |

## 5. 精度 Benchmark

**本版本未发布 COCO mAP 基线（标注为未测量）。** 复现口径同 yolov8 sample（COCO val2017 + pycocotools，640x640 模型 .meta 中定义的输入张量格式，score 0.25 / IoU 0.45）。

## 6. 性能测试说明

- 复现命令：bash utils/tools/run_all_models_benchmark.sh --device C1200
- 板端建议设为 performance 模式，避免其他进程占用 CPU/DSP 带宽
- C++ 耗时为预热后 run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

## 7. 精度测试说明

- 指标定义按任务区分：检测/分割/姿态用 COCO mAP，分类用 Top-1/Top-5，车道线用 F1，立体匹配用 EPE
- 固定 shape 量化模型与官方动态 shape 结果存在口径差异，属预期；对比时保持量化前后同口径
- 校准数据规模：100~500 张代表性输入
- 未标注数值 = 该指标本版本未测量（不做推测性填充）

## 8. 验证结果汇总

3/3 模型（v10n/v11n/v12n）已通过 Aura（C1200）板端推理验证（test_data/ 含三变体结果图）。
