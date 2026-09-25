# CondLaneNet model files

C1200 model files are copied byte for byte from the corresponding `models/condlanenet/Deploy/C++/prebuild/model/` package in the source project. The runtime file names are kept exactly as published.

- `condlanenet-c1200`: `condlanenet.ModelGraph.bin`, `condlanenet.ModelGraph.bin.txt`, `condlanenet.meta`, `condlanenet.sdag.bin`, `weights.bin`

Run `./download.sh` to check that every required file is present. This checks package completeness; board inference and accuracy require separate validation.

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `condlanenet` | 148.54 | 6.732 ms |
