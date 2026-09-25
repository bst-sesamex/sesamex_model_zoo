# ReID model files

C1200 model files are copied byte for byte from the corresponding `models/reid/Deploy/C++/prebuild/model/` package in the source project. The runtime file names are kept exactly as published.

- `mobilenetv2-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `osnet-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `resnet50-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`

Run `./download.sh` to check that every required file is present. This checks package completeness; board inference and accuracy require separate validation.

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `mobilenetv2` | 1785.71 | 0.560 ms |
| `osnet` | 203.52 | 4.9135 ms |
| `resnet50` | 469.04 | 2.132 ms |
