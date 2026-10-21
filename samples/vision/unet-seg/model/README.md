# UNet model files

C1200 model files are committed under `model/c1200/` with the runtime file names listed below.

- `unet_512x1024-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`
- `unet_512x512-c1200`: `reid.Enrsd.bin`, `reid.meta`, `reid_stripped.lib`, `weights.bin`

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `unet_512x1024` | 18.15 | 55.111 ms |
| `unet_512x512` | 35.51 | 28.158 ms |
| `unet_512x512` | a1000 | `unet_512x512.meta`, `unet_512x512.Enrsd.bin`, `unet_512x512_stripped.lib`, `weights.bin` |
| `unet_512x1024` | a1000 | `unet_512x1024.meta`, `unet_512x1024.Enrsd.bin`, `unet_512x1024_stripped.lib`, `weights.bin` |
