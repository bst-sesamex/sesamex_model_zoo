# CondLaneNet model files

C1200 model files are committed under `model/c1200/` with the runtime file names listed below.

- `condlanenet-c1200`: `condlanenet.ModelGraph.bin`, `condlanenet.ModelGraph.bin.txt`, `condlanenet.meta`, `condlanenet.sdag.bin`, `weights.bin`

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `condlanenet` | 148.54 | 6.732 ms |
| `condlanenet` | a1000 | `condlanenet.meta`, `condlanenet.Enrsd.bin`, `condlanenet_stripped.lib`, `weights.bin` |
