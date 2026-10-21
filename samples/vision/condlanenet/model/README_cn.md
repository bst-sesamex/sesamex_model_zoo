# CondLaneNet 模型文件

C1200 模型文件位于 `model/c1200/`，文件名与下方列表一致。

- `condlanenet-c1200`：`condlanenet.ModelGraph.bin`, `condlanenet.ModelGraph.bin.txt`, `condlanenet.meta`, `condlanenet.sdag.bin`, `weights.bin`

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `condlanenet` | 148.54 | 6.732 ms |
| `condlanenet` | a1000 | `condlanenet.meta`、`condlanenet.Enrsd.bin`、`condlanenet_stripped.lib`、`weights.bin` |
