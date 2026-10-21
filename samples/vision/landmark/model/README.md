# Face Landmark Detection Model Files

The quantized model packages for the C1200 chip are committed to this
repository under `model/c1200/<variant>-c1200/`.

## Default model

- `size56` (c1200)

## Supported models

| Model | Chip | Files |
|---|---|---|
| `size56` | c1200 | weights.bin, landmark.Enrsd.bin, landmark.meta, landmark_stripped.lib |
| `size224` | c1200 | weights.bin, landmark.Enrsd.bin, landmark.meta, landmark_stripped.lib |
> **Kalos (A1000 chip)**: `model/a1000/<variant>-a1000/` packages ship with the repository, using the same four-file Enrsd layout.

## Verify

```bash
```

compatibility; it only verifies file presence now.

Layout:

```text
model/
└── c1200/
    └── size56-c1200/
        ├── weights.bin, landmark.Enrsd.bin, landmark.meta, landmark_stripped.lib.Enrsd.bin    # quantized model
        ├── weights.bin
        ├── weights.bin, landmark.Enrsd.bin, landmark.meta, landmark_stripped.lib.meta
        └── weights.bin, landmark.Enrsd.bin, landmark.meta, landmark_stripped.lib_stripped.lib
```

## Licensing

Model packages are produced by the BST internal quantization pipeline and are
distributed under this repository's license; training-side code is not part of
this repository (see ../conversion/).

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `size224` | 1506.02 | 0.664 ms |
| `size56` | 2403.85 | 0.416 ms |
| `size224` | a1000 | `landmark.meta`, `landmark.Enrsd.bin`, `landmark_stripped.lib`, `weights.bin` |
| `size56` | a1000 | `landmark.meta`, `landmark.Enrsd.bin`, `landmark_stripped.lib`, `weights.bin` |
