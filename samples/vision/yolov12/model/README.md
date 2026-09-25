# YOLOv10/11/12 Model Files

The quantized model packages for the C1200 chip **are committed to this
repository** under `model/c1200/<variant>-c1200/` (in-repo weights policy,
since v1.0.1). No download step is required; the former model-server channel
has been retired.

## Default model

- `yolov10n-det` (c1200)

## Supported models

| Model | Chip | Files |
|---|---|---|
| `yolov10n-det` | c1200 | weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib |
| `yolov11n-det` | c1200 | weights.bin, yolov11n-det.Enrsd.bin, yolov11n-det.meta, yolov11n-det_stripped.lib |
| `yolov12n-det` | c1200 | weights.bin, yolov12n-det.Enrsd.bin, yolov12n-det.meta, yolov12n-det_stripped.lib |
> **Kalos (A1000 chip)**: A1000 packages are not included in this release; the
> A1000 layout is kept for compatibility and packages will be added when published.

## Verify

```bash
./download.sh              # completeness check (all variants)
./download.sh yolov10n-det       # one variant
```

The script name is kept from the earlier download-based layout for
compatibility; it only verifies file presence now.

Layout:

```text
model/
└── c1200/
    └── yolov10n-det-c1200/
        ├── weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib.Enrsd.bin    # quantized model
        ├── weights.bin
        ├── weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib.meta
        └── weights.bin, yolov10n-det.Enrsd.bin, yolov10n-det.meta, yolov10n-det_stripped.lib_stripped.lib
```

## Licensing

Model packages are produced by the BST internal quantization pipeline and are
distributed under this repository's license; training-side code is not part of
this repository (see ../conversion/).

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `yolov10n-det` | 207.13 | 4.828 ms |
| `yolov11n-det` | 218.20 | 4.583 ms |
| `yolov12n-det` | 39.95 | 25.032 ms |
