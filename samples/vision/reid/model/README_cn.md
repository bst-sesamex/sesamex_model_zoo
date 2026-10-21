# ReID 模型文件

C1200 模型文件位于 `model/c1200/`，文件名与下方列表一致。

- `mobilenetv2-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `osnet-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `resnet50-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `mobilenetv2` | 1785.71 | 0.560 ms |
| `osnet` | 203.52 | 4.9135 ms |
| `resnet50` | 469.04 | 2.132 ms |
| `mobilenetv2` | a1000 | `mobilenetv2.meta`、`mobilenetv2.Enrsd.bin`、`mobilenetv2_stripped.lib`、`weights.bin` |
| `osnet` | a1000 | `osnet.meta`、`osnet.Enrsd.bin`、`osnet_stripped.lib`、`weights.bin` |
| `resnet50` | a1000 | `resnet50.meta`、`resnet50.Enrsd.bin`、`resnet50_stripped.lib`、`weights.bin` |
