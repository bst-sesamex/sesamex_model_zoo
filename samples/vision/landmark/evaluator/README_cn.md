[English](./README.md) | 简体中文

# 人脸关键点检测模型评测


本目录记录人脸关键点检测在 Aura / Kalos 平台上的性能与精度口径。

## 1. 评测前提

- 板端：Aura（C1200 芯片）或 Kalos（A1000 芯片），系统已含 `libbsnnrt.so`
- Python 3 + `pybsnn`（板端执行仓库根 `init_env.sh` 自动安装）
- 精度评测需：Python 3.8+、300W 数据集评测工具
- 模型已随仓库发布于 `../model/`；`../model/download.sh` 做完整性校验。

## 2. 数据集说明

- 评测数据集：300W（68 点协议）
- 快速验证图片：`../test_data/`（56/224 人脸裁剪图）
- 完整数据集不入仓，见仓库根 `datasets/` 的准备脚本

## 3. 使用方法

单张冒烟（板端）：

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

C++ demo（两个尺寸）：

```bash
cd ../runtime/cpp && bash build.sh && sh run_demo.sh -C1200
```

批量性能（板端，全部模型）：

```bash
bash utils/tools/run_all_models_benchmark.sh --device C1200
```

## 4. 性能 Benchmark

测试条件：Aura C1200，默认 auto 引擎；预热一次后三轮 C++ run+wait 单调时钟耗时中位数，不含前后处理。

| 模型 | 输入 | Aura C1200 FPS | Aura 耗时 |
|---|---|---|---|
| `size56` | 56x56 | 2403.85 FPS | 0.416 ms |
| `size224` | 224x224 | 1506.02 FPS | 0.664 ms |

Kalos（A1000 芯片）数据：本版本未测量。

## 5. 精度 Benchmark

**本版本未发布 300W NME 基线（标注为未测量）。** 复现口径：300W 68 点协议，归一化误差 NME（眼距归一化），
预处理（对齐裁剪到 56/224）与 runtime 示例一致。量化模型与 float 模型需在同一协议下对比。

## 6. 性能测试说明

- 复现命令：bash utils/tools/run_all_models_benchmark.sh --device C1200
- 板端设为性能模式；避免 CPU/DSP 并发负载
- C++ 耗时为预热后 run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

## 7. 精度测试说明

- 指标：68 点 NME（归一化平均误差）
- 定形量化模型与官方动态 shape 数字存在设计性差异；float 与量化需同协议对比
- 校准数据：100~500 张代表性人脸裁剪图
- 未列出的数值 = 本版本未测量（绝不臆造）

## 8. 验证结论

2/2 变体已通过 Aura（C1200）板端推理验证（benchmark_results 记录：68 点输出，
norm range [0.117, 0.898]，SUCCESS）。
