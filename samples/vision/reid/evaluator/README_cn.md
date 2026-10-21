[English](./README.md) | 简体中文

# 行人重识别模型评测

本目录记录行人重识别在 Aura / Kalos 平台上的性能与精度口径。

## 1. 评测前提

- 板端：Aura（C1200 芯片）或 Kalos（A1000 芯片），系统已含 `libbsnnrt.so`
- Python 3 + `pybsnn`（板端执行仓库根 `init_env.sh` 自动安装）
- 精度评测需：Python 3.8+、Torchreid rank 评测工具

## 2. 数据集说明

- 评测数据集：Market-1501 / DukeMTMC-reID（标准 rank-1/mAP 协议）
- 快速验证图片：`../test_data/`（行人裁剪图，True/False 成对演示）
- 完整数据集不入仓，见仓库根 `datasets/` 的准备脚本

## 3. 使用方法

单张冒烟（板端）：

```bash
cd ../runtime/python && python3 pybsnn_demo.py
```

C++ demo（成对匹配，3 骨干 x True/False）：

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
| `mobilenetv2` | 128x256 | 1785.71 FPS | 0.560 ms |
| `osnet` | 128x256 | 203.52 FPS | 4.9135 ms |
| `resnet50` | 128x256 | 469.04 FPS | 2.132 ms |

Kalos（A1000 芯片）数据：本版本未测量。

## 5. 精度 Benchmark

**本版本未发布 Market-1501 rank-1/mAP 基线（标注为未测量）。** 复现口径：Market-1501 标准协议
（query/gallery 划分与评测准则），特征为量化模型 1280/2048 维嵌入 + 余弦距离 + C++ demo 阈值。

## 6. 性能测试说明

- 复现命令：bash utils/tools/run_all_models_benchmark.sh --device C1200
- 板端设为性能模式；避免 CPU/DSP 并发负载
- C++ 耗时为预热后 run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

## 7. 精度测试说明

- 指标：rank-1 / mAP（Market-1501 协议）
- 成对阈值（余弦距离）：mobilenetv2 0.3193、osnet 0.3677、resnet50 0.2576 —— 板端实测标定，见 conversion/README_cn.md
- 校准数据：100~500 张代表性行人裁剪图
- 未列出的数值 = 本版本未测量（绝不臆造）

## 8. 验证结论

3/3 变体已通过 Aura（C1200）板端推理验证（benchmark_results 记录：6 对 True/False
匹配结果与板端标定阈值一致，SUCCESS）。
