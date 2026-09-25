# YOLOv10/11/12 模型文件

本目录的 C1200 芯片量化模型包**直接随仓库发布**（`model/c1200/<variant>-c1200/`，
自 v1.0.1 起入库），无需下载；原模型服务器下载通道已停用。

## 默认模型

- `yolov10n-det`（c1200）

## 支持模型

| 模型 | 芯片型号 | 文件构成 |
|---|---|---|
| `yolov10n-det` | c1200 | weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib |
| `yolov11n-det` | c1200 | weights.bin, yolov11n-det.Enrsd.bin, yolov11n-det.meta, yolov11n-det_stripped.lib |
| `yolov12n-det` | c1200 | weights.bin, yolov12n-det.Enrsd.bin, yolov12n-det.meta, yolov12n-det_stripped.lib |
> **Kalos（A1000 芯片）**：本版本未随附 A1000 模型包；A1000 目录结构保留以兼容，包发布后补入。

## 完整性校验

```bash
./download.sh              # 全部变体完整性检查
./download.sh yolov10n-det       # 单个变体
```

脚本名沿用自早期下载制布局以保持兼容；现在仅做文件在位校验。

目录结构：

```text
model/
└── c1200/
    └── yolov10n-det-c1200/
        ├── weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib.Enrsd.bin    # 量化模型
        ├── weights.bin
        ├── weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib.meta
        └── weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib_stripped.lib
```

## 许可说明

模型包由 BST 内部量化流水线产出，随本仓库许可分发；训练侧代码不随本仓库发布（见 ../conversion/）。

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `yolov10n-det` | 207.13 | 4.828 ms |
| `yolov11n-det` | 218.20 | 4.583 ms |
| `yolov12n-det` | 39.95 | 25.032 ms |
