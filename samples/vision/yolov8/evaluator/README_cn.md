[English](./README.md) | 简体中文

# YOLOv8 模型评测


本目录记录 YOLOv8 在 Aura / Kalos 平台上的性能与精度口径。

## 1. 评测前提

- 板端：Aura（C1200 芯片）或 Kalos（A1000 芯片），系统已含 `libbsnnrt.so`
- Python 3 + `pybsnn`（板端执行仓库根 `init_env.sh` 自动安装）
- 精度评测需：Python 3.8+、`pycocotools`（检测类）或对应数据集评测工具
- 模型已随仓库发布于 `../model/`；`../model/download.sh` 做完整性校验。

## 2. 数据集说明

- 评测数据集：COCO val2017（检测/分割/姿态）
- 快速验证图片：`../test_data/`（单张冒烟）
- 完整数据集不入仓，见仓库根 `datasets/` 的准备脚本

## 3. 使用方法

单张冒烟（板端）：

```bash
cd ../runtime/python && python3 pybsnn_demo.py（det）等
```

批量性能（板端，全部模型）：

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. 性能 Benchmark

测试条件：Aura C1200，默认 auto 引擎；预热一次后三轮 C++ run+wait 单调时钟耗时中位数，不含前后处理。

| 模型 | 输入 | Aura C1200 FPS | Aura 耗时 |
|---|---|---|---|
| `yolov8n-det` | 640x640 | 392.46 FPS | 2.548 ms |
| `yolov8n-seg` | 640x640 | 370.23 FPS | 2.701 ms |
| `yolov8n-pose` | 640x640 | 399.36 FPS | 2.504 ms |

## 5. 精度 Benchmark

**本版本未发布 COCO mAP 基线（标注为未测量）。** 当前 sample 以部署链路与板端验证为主。如需复现 mAP：使用 COCO val2017 + pycocotools，保持与 runtime 示例一致的 640x640 模型 .meta 中定义的输入张量格式 预处理、score 0.25 / NMS IoU 0.45 阈值；注意固定shape 量化模型与官方动态 shape 的 mAP 口径差异属预期。

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

3/3 模型（det/seg/pose）已通过 Aura（C1200）板端推理验证（test_data/ 含输入 bus.jpg 与三任务结果图）。
