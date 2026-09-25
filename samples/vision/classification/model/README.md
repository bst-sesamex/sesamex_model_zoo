# Image Classification model files

C1200 model files are copied byte for byte from the corresponding `models/classification/Deploy/C++/prebuild/model/` package in the source project. The runtime file names are kept exactly as published.

- `mobilenetv2-c1200`: `fcos.Enrsd.bin`, `fcos.meta`, `fcos_stripped.lib`, `weights.bin`
- `resnet18-c1200`: `fcos.Enrsd.bin`, `fcos.meta`, `fcos_stripped.lib`, `weights.bin`
- `resnet34-c1200`: `fcos.Enrsd.bin`, `fcos.meta`, `fcos_stripped.lib`, `weights.bin`

Run `./download.sh` to check that every required file is present. This checks package completeness; board inference and accuracy require separate validation.

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `mobilenetv2` | 1464.13 | 0.683 ms |
| `resnet18` | 763.36 | 1.310 ms |
| `resnet34` | 490.44 | 2.039 ms |
