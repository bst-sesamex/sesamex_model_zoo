# YOLOv5 模型文件

C1200 模型文件位于 `model/c1200/`，文件名与下方列表一致。

- `yolov5m-c1200`：`weights.bin`, `yolov5m.ModelGraph.bin`, `yolov5m.ModelGraph.bin.txt`, `yolov5m.meta`, `yolov5m.sdag.bin`

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `yolov5m` | 145.84 | 6.857 ms |
| `yolov5m` | a1000 | `yolov5m.meta`、`yolov5m.Enrsd.bin`、`yolov5m_stripped.lib`、`weights.bin` |
