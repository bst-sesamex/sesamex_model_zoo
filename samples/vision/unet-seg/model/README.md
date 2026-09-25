# UNet model files

C1200 model files are copied byte for byte from the corresponding `models/unet-seg/Deploy/C++/prebuild/model/` package in the source project. The runtime file names are kept exactly as published.

- `unet_512x1024-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `unet_512x512-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`

Run `./download.sh` to check that every required file is present. This checks package completeness; board inference and accuracy require separate validation.

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `unet_512x1024` | 18.15 | 55.111 ms |
| `unet_512x512` | 35.51 | 28.158 ms |
