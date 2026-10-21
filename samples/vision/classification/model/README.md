# Image Classification model files

C1200 model files are committed under `model/c1200/` with the runtime file names listed below.

- `mobilenetv2-c1200` / `resnet18-c1200` / `resnet34-c1200`: each contains
  `fcos.Enrsd.bin`, `fcos.meta`, `fcos_stripped.lib`, `weights.bin`
  (file names kept exactly as published; loaders glob `*.meta`).

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `mobilenetv2` | 1464.13 | 0.683 ms |
| `resnet18` | 763.36 | 1.310 ms |
| `resnet34` | 490.44 | 2.039 ms |
| `mobilenetv2` | a1000 | `mobilenetv2.meta`, `mobilenetv2.Enrsd.bin`, `mobilenetv2_stripped.lib`, `weights.bin` |
| `resnet18` | a1000 | `resnet18.meta`, `resnet18.Enrsd.bin`, `resnet18_stripped.lib`, `weights.bin` |
| `resnet34` | a1000 | `resnet34.meta`, `resnet34.Enrsd.bin`, `resnet34_stripped.lib`, `weights.bin` |
