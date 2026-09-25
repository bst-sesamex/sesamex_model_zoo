# YOLOv8 Model Files

The quantized model packages for the C1200 chip **are committed to this
repository** under `model/c1200/<variant>-c1200/` (in-repo weights policy,
since v1.0.1). No download step is required; the former model-server channel
has been retired.

## Default model

- `yolov8n-det` (c1200)

## Supported models

| Model | Chip | Files |
|---|---|---|
| `yolov8n-det` | c1200 | weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib |
| `yolov8n-seg` | c1200 | weights.bin, yolov8n-seg.Enrsd.bin, yolov8n-seg.meta, yolov8n-seg_stripped.lib |
| `yolov8n-pose` | c1200 | weights.bin, yolov8n-pose.Enrsd.bin, yolov8n-pose.meta, yolov8n-pose_stripped.lib |
> **Kalos (A1000 chip)**: A1000 packages are not included in this release; the
> A1000 layout is kept for compatibility and packages will be added when published.

## Verify

```bash
./download.sh              # completeness check (all variants)
./download.sh yolov8n-det       # one variant
```

The script name is kept from the earlier download-based layout for
compatibility; it only verifies file presence now.

Layout:

```text
model/
└── c1200/
    └── yolov8n-det-c1200/
        ├── weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib.Enrsd.bin    # quantized model
        ├── weights.bin
        ├── weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib.meta
        └── weights.bin, yolov8n-det.Enrsd.bin, yolov8n-det.meta, yolov8n-det_stripped.lib_stripped.lib
```

## Licensing

Model packages are produced by the BST internal quantization pipeline and are
distributed under this repository's license; training-side code is not part of
this repository (see ../conversion/).

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `yolov8n-det` | 392.46 | 2.548 ms |
| `yolov8n-pose` | 399.36 | 2.504 ms |
| `yolov8n-seg` | 370.23 | 2.701 ms |
