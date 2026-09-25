[English](./README.md) | 简体中文

# 语义分割（UNet）模型评测


本目录记录语义分割在 Aura / Kalos 平台上的性能与精度口径。

## 1. 评测前提

- 板端：Aura（C1200 芯片）或 Kalos（A1000 芯片），系统已含 `libbsnnrt.so`
- Python 3 + `pybsnn`（板端执行仓库根 `init_env.sh` 自动安装）
- 精度评测需：Python 3.8+、CamVid 评测工具（mIoU）
- 模型已随仓库发布于 `../model/`；`../model/download.sh` 做完整性校验。

## 2. 数据集说明

- 评测数据集：CamVid（19 类协议，源图 2048x1024）
- 快速验证图片：`../test_data/test.png`（2048x1024）与板端输出对照图
- 完整数据集不入仓，见仓库根 `datasets/` 的准备脚本

## 3. 使用方法

单张冒烟（板端）：

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

C++ demo（两种输入尺寸，叠加输出）：

```bash
cd ../runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

批量性能（板端，全部模型）：

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. 性能 Benchmark

测试条件：Aura C1200，默认 auto 引擎；预热一次后三轮 C++ run+wait 单调时钟耗时中位数，不含前后处理。
推理在模型输入分辨率上进行；2048x1024 原图经缩放/切块后送入推理。

| 模型 | 输入 | Aura C1200 FPS | Aura 耗时 |
|---|---|---|---|
| `unet_512x512` | 512x512 | 35.51 FPS | 28.158 ms |
| `unet_512x1024` | 512x1024 | 18.15 FPS | 55.111 ms |

Kalos（A1000 芯片）数据：本版本未测量。

## 5. 精度 Benchmark

**本版本未发布 CamVid mIoU 基线（标注为未测量）。** 复现口径：CamVid 19 类协议，
全图/滑窗推理后 argmax，与 float 模型在同一预处理（BGR uint8 无归一化）下对比 mIoU。

## 6. 性能测试说明

- 复现命令：bash utils/tools/run_all_models_benchmark.sh --device C1200
- 板端设为性能模式；避免 CPU/DSP 并发负载
- C++ 耗时为预热后 run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

## 7. 精度测试说明

- 指标：mIoU（19 类，CamVid 协议）
- 输出为 INT8 logits（反量化 scale 0.250）；argmax 与 scale 无关
- 校准数据：100~500 张代表性街景裁剪图
- 未列出的数值 = 本版本未测量（绝不臆造）

## 8. 验证结论

2/2 变体已通过 Aura（C1200）板端推理验证（benchmark_results 记录：
512x512x19 / 1024x512x19 输出张量，dequant scale 0.250，SUCCESS）。
