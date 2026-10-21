# UNet 模型文件

C1200 模型文件位于 `model/c1200/`，文件名与下方列表一致。

- `unet_512x1024-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `unet_512x512-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `unet_512x1024` | 18.15 | 55.111 ms |
| `unet_512x512` | 35.51 | 28.158 ms |
| `unet_512x512` | a1000 | `unet_512x512.meta`、`unet_512x512.Enrsd.bin`、`unet_512x512_stripped.lib`、`weights.bin` |
| `unet_512x1024` | a1000 | `unet_512x1024.meta`、`unet_512x1024.Enrsd.bin`、`unet_512x1024_stripped.lib`、`weights.bin` |
