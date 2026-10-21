# YOLOv8 模型文件

C1200 芯片量化模型包位于 `model/c1200/<variant>-c1200/`，随仓库分发。

## 默认模型

- `yolov8n-det`（c1200）

## 支持模型

| 模型 | 芯片型号 | 文件构成 |
|---|---|---|
| `yolov8n-det` | c1200 | weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib |
| `yolov8n-seg` | c1200 | weights.bin, yolov8n-seg.Enrsd.bin, yolov8n-seg.meta, yolov8n-seg_stripped.lib |
| `yolov8n-pose` | c1200 | weights.bin, yolov8n-pose.Enrsd.bin, yolov8n-pose.meta, yolov8n-pose_stripped.lib |
> **Kalos（A1000 芯片）**：`model/a1000/<变体>-a1000/` 随仓库发布，文件构成为四件套 Enrsd 形态。

目录结构：

```text
model/
└── c1200/
    └── yolov8n-det-c1200/
        ├── weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib.Enrsd.bin    # 量化模型
        ├── weights.bin
        ├── weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib.meta
        └── weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib_stripped.lib
```

## 许可说明

模型包由 BST 内部量化流水线产出，随本仓库许可分发；训练侧代码不随本仓库发布（见 ../conversion/）。

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `yolov8n-det` | 392.46 | 2.548 ms |
| `yolov8n-pose` | 399.36 | 2.504 ms |
| `yolov8n-seg` | 370.23 | 2.701 ms |
| `yolov8n-det` | a1000 | `yolov8n-det.meta`、`yolov8n-det.Enrsd.bin`、`yolov8n-det_stripped.lib`、`weights.bin` |
| `yolov8n-seg` | a1000 | `yolov8n-seg.meta`、`yolov8n-seg.Enrsd.bin`、`yolov8n-seg_stripped.lib`、`weights.bin` |
| `yolov8n-pose` | a1000 | `yolov8n-pose.meta`、`yolov8n-pose.Enrsd.bin`、`yolov8n-pose_stripped.lib`、`weights.bin` |
