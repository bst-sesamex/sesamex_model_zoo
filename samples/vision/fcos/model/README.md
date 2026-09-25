# FCOS Model Files

The quantized model packages for the C1200 chip **are committed to this
repository** under `model/c1200/<variant>-c1200/` (in-repo weights policy,
since v1.0.1). No download step is required; the former model-server channel
has been retired.

## Default model

- `fcos` (c1200)

## Supported models

| Model | Chip | Files |
|---|---|---|
| `fcos` | c1200 | weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib |
> **Kalos (A1000 chip)**: A1000 packages are not included in this release; the
> A1000 layout is kept for compatibility and packages will be added when published.

## Verify

```bash
./download.sh              # completeness check (all variants)
./download.sh fcos       # one variant
```

The script name is kept from the earlier download-based layout for
compatibility; it only verifies file presence now.

Layout:

```text
model/
└── c1200/
    └── fcos-c1200/
        ├── weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib.Enrsd.bin    # quantized model
        ├── weights.bin
        ├── weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib.meta
        └── weights.bin, fcos.Enrsd.bin, fcos.meta, fcos_stripped.lib_stripped.lib
```

## Licensing

Model packages are produced by the BST internal quantization pipeline and are
distributed under this repository's license; training-side code is not part of
this repository (see ../conversion/).

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `fcos` | 50.79 | 19.689 ms |
