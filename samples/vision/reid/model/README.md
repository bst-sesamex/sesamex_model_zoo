# ReID model files

C1200 model files are committed under `model/c1200/` with the runtime file names listed below.

- `mobilenetv2-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `osnet-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `resnet50-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `mobilenetv2` | 1785.71 | 0.560 ms |
| `osnet` | 203.52 | 4.9135 ms |
| `resnet50` | 469.04 | 2.132 ms |
| `mobilenetv2` | a1000 | `mobilenetv2.meta`, `mobilenetv2.Enrsd.bin`, `mobilenetv2_stripped.lib`, `weights.bin` |
| `osnet` | a1000 | `osnet.meta`, `osnet.Enrsd.bin`, `osnet_stripped.lib`, `weights.bin` |
| `resnet50` | a1000 | `resnet50.meta`, `resnet50.Enrsd.bin`, `resnet50_stripped.lib`, `weights.bin` |
