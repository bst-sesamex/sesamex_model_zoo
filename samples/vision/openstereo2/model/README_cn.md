# OpenStereo-2 模型文件

C1200 模型文件按原工程 `models/openstereo2/Deploy/C++/prebuild/model/` 的对应模型包逐文件恢复，保留原始文件名。

- `lightstereo-c1200`：`lightstereo.ModelGraph.bin`, `lightstereo.ModelGraph.bin.txt`, `lightstereo.meta`, `lightstereo.sdag.bin`, `weights.bin`

运行 `./download.sh` 检查所需文件是否在位。此检查仅验证包完整性，板端推理与精度需要另行验证。

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `lightstereo` | 14.64 | 68.291 ms |
