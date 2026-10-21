# Image Classification 模型文件

C1200 模型文件位于 `model/c1200/`，文件名与下方列表一致。

- `mobilenetv2-c1200` / `resnet18-c1200` / `resnet34-c1200`：均含
  `fcos.Enrsd.bin`, `fcos.meta`, `fcos_stripped.lib`, `weights.bin`
  （文件名与发布一致；加载器按 `*.meta` 通配）。

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `mobilenetv2` | 1464.13 | 0.683 ms |
| `resnet18` | 763.36 | 1.310 ms |
| `resnet34` | 490.44 | 2.039 ms |
| `mobilenetv2` | a1000 | `mobilenetv2.meta`、`mobilenetv2.Enrsd.bin`、`mobilenetv2_stripped.lib`、`weights.bin` |
| `resnet18` | a1000 | `resnet18.meta`、`resnet18.Enrsd.bin`、`resnet18_stripped.lib`、`weights.bin` |
| `resnet34` | a1000 | `resnet34.meta`、`resnet34.Enrsd.bin`、`resnet34_stripped.lib`、`weights.bin` |
