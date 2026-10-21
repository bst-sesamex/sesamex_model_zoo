# FCOS 模型文件

C1200 芯片量化模型包位于 `model/c1200/<variant>-c1200/`，随仓库分发。

## 默认模型

- `fcos`（c1200）

## 支持模型

| 模型 | 芯片型号 | 文件构成 |
|---|---|---|
| `fcos` | c1200 | weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib |
> **Kalos（A1000 芯片）**：`model/a1000/<变体>-a1000/` 随仓库发布，文件构成为四件套 Enrsd 形态。

目录结构：

```text
model/
└── c1200/
    └── fcos-c1200/
        ├── weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib.Enrsd.bin    # 量化模型
        ├── weights.bin
        ├── weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib.meta
        └── weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib_stripped.lib
```

## 许可说明

模型包由 BST 内部量化流水线产出，随本仓库许可分发；训练侧代码不随本仓库发布（见 ../conversion/）。

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `fcos` | 50.79 | 19.689 ms |
| `fcos` | a1000 | `fcos.meta`、`fcos.Enrsd.bin`、`fcos_stripped.lib`、`weights.bin` |
