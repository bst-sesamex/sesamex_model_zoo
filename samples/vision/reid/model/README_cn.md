# ReID 模型文件

C1200 模型文件按原工程 `models/reid/Deploy/C++/prebuild/model/` 的对应模型包逐文件恢复，保留原始文件名。

- `mobilenetv2-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `osnet-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `resnet50-c1200`：`reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`

运行 `./download.sh` 检查所需文件是否在位。此检查仅验证包完整性，板端推理与精度需要另行验证。

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `mobilenetv2` | 1785.71 | 0.560 ms |
| `osnet` | 203.52 | 4.9135 ms |
| `resnet50` | 469.04 | 2.132 ms |
