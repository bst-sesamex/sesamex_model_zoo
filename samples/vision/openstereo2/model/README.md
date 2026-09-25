# OpenStereo-2 model files

C1200 model files are copied byte for byte from the corresponding `models/openstereo2/Deploy/C++/prebuild/model/` package in the source project. The runtime file names are kept exactly as published.

- `lightstereo-c1200`: `lightstereo.ModelGraph.bin`, `lightstereo.ModelGraph.bin.txt`, `lightstereo.meta`, `lightstereo.sdag.bin`, `weights.bin`

Run `./download.sh` to check that every required file is present. This checks package completeness; board inference and accuracy require separate validation.

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `lightstereo` | 14.64 | 68.291 ms |
