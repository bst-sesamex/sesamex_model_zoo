# Face Landmark Detection Model Files

The quantized model packages for the C1200 chip **are committed to this
repository** under `model/c1200/<variant>-c1200/` (in-repo weights policy,
since v1.0.1). No download step is required; the former model-server channel
has been retired.

## Default model

- `size56` (c1200)

## Supported models

| Model | Chip | Files |
|---|---|---|
| `size56` | c1200 | weights.bin, landmark.Enrsd.bin, landmark.meta, landmark_stripped.lib |
| `size224` | c1200 | weights.bin, landmark.Enrsd.bin, landmark.meta, landmark_stripped.lib |
> **Kalos (A1000 chip)**: A1000 packages are not included in this release; the
> A1000 layout is kept for compatibility and packages will be added when published.

## Verify

```bash
./download.sh              # completeness check (all variants)
./download.sh size56       # one variant
```

The script name is kept from the earlier download-based layout for
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
