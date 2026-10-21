# YOLOv5 model files

C1200 model files are committed under `model/c1200/` with the runtime file names listed below.

- `yolov5m-c1200`: `weights.bin`, `yolov5m.ModelGraph.bin`, `yolov5m.ModelGraph.bin.txt`, `yolov5m.meta`, `yolov5m.sdag.bin`

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `yolov5m` | 145.84 | 6.857 ms |
| `yolov5m` | a1000 | `yolov5m.meta`, `yolov5m.Enrsd.bin`, `yolov5m_stripped.lib`, `weights.bin` |
