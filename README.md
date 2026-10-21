English | [简体中文](./README_cn.md)

# SesameX Model Zoo

**Version:** see [VERSION](./VERSION) · **Changelog:** [CHANGELOG.md](./CHANGELOG.md)

SesameX Model Zoo provides deployment samples for mainstream vision models on
Black Sesame edge computing platforms (Aura / Kalos). Each sample covers the
full on-device path: model conversion, quantized model packages, Python and
C++ inference runtimes, accuracy/performance evaluation and verification data.

Core workflow:

```text
Train (upstream repo) -> export ONNX -> calibration data -> quantized compile
      -> on-device C++ / Python inference -> FPS & accuracy verification
```

## Supported platforms

| Platform | Notes |
|---|---|
| Aura (C1200 chip) | Primary platform. FPS data in this release measured on Aura. |
| Kalos (A1000 chip) | All 10 vision samples ship A1000 packages under `model/a1000/`; see each sample's `model/README`. |

## Repository layout

```
sesamex-model-zoo/
├── samples/vision/        # model samples, one directory per family
├── utils/py_utils/        # shared pre/post-processing & visualization libs
├── utils/tools/           # batch build / benchmark / eval tools
├── datasets/              # dataset preparation scripts (no dataset blobs)
├── docs/                  # release policy, manifests, catalog
└── resource/imgs/         # images used by READMEs
```

## Model index

Aura C1200 FPS below is derived from the median of three warmup-once C++ synchronous inference measurements.
Conversion and evaluation details are in each sample directory.

### Vision

| Model | Sample | Task | Aura FPS | Aura Inference latency | Kalos FPS | Kalos Inference latency |
|---|---|---|---:|---:|---:|---:|
| LightStereo (OpenStereo-2) | [openstereo2](./samples/vision/openstereo2) | stereo matching / depth | 4.58 | 218.219 ms | 4.68 | 213.816 ms |
| ResNet-18 | [classification](./samples/vision/classification) | image classification | 839.40 | 1.191 ms | 477.48 | 2.094 ms |
| ResNet-34 | [classification](./samples/vision/classification) | image classification | 530.60 | 1.885 ms | 293.89 | 3.403 ms |
| MobileNetV2 | [classification](./samples/vision/classification) | image classification | 1686.34 | 0.593 ms | 1077.59 | 0.928 ms |
| CondLaneNet | [condlanenet](./samples/vision/condlanenet) | lane detection | 148.13 | 6.751 ms | 89.19 | 11.212 ms |
| FCOS-ResNet50 | [fcos](./samples/vision/fcos) | object detection | 51.29 | 19.497 ms | 35.12 | 28.471 ms |
| YOLOv10n | [yolov12](./samples/vision/yolov12) | object detection | 295.92 | 3.379 ms | 161.72 | 6.184 ms |
| YOLO11n | [yolov12](./samples/vision/yolov12) | object detection | 307.47 | 3.252 ms | 162.50 | 6.154 ms |
| YOLOv12n | [yolov12](./samples/vision/yolov12) | object detection | 42.81 | 23.358 ms | 33.15 | 30.162 ms |
| YOLOv8n-det | [yolov8](./samples/vision/yolov8) | object detection | 423.01 | 2.364 ms | 216.34 | 4.622 ms |
| YOLOv8n-seg | [yolov8](./samples/vision/yolov8) | instance segmentation | 401.39 | 2.491 ms | 208.78 | 4.790 ms |
| YOLOv8n-pose | [yolov8](./samples/vision/yolov8) | pose estimation | 430.66 | 2.322 ms | 216.59 | 4.617 ms |
| YOLOv5m | [yolov5](./samples/vision/yolov5) | object detection | 144.96 | 6.899 ms | 84.58 | 11.823 ms |
| Face Landmark 56 | [landmark](./samples/vision/landmark) | face landmark (68 pts) | 2693.00 | 0.371 ms | 1777.25 | 0.563 ms |
| Face Landmark 224 | [landmark](./samples/vision/landmark) | face landmark (68 pts) | 1717.23 | 0.582 ms | 1136.79 | 0.880 ms |
| ReID MobileNetV2 | [reid](./samples/vision/reid) | person re-identification | 2122.39 | 0.471 ms | 1033.24 | 0.968 ms |
| ReID OSNet | [reid](./samples/vision/reid) | person re-identification | 374.58 | 2.670 ms | 229.48 | 4.358 ms |
| ReID ResNet-50 | [reid](./samples/vision/reid) | person re-identification | 511.90 | 1.954 ms | 280.14 | 3.570 ms |
| UNet 512x512 | [unet-seg](./samples/vision/unet-seg) | semantic segmentation | 35.06 | 28.524 ms | 22.28 | 44.882 ms |
| UNet 512x1024 | [unet-seg](./samples/vision/unet-seg) | semantic segmentation | 18.71 | 53.446 ms | 10.62 | 94.119 ms |
## Quick start

On an aarch64 edge board (Aura / Kalos):

```bash
# 1. initialize environment (runtime env + PyBSNN wheel)
chmod +x init_env.sh
./init_env.sh

# 2. run the Python runtime (on board; example: yolov8)
cd samples/vision/yolov8/runtime/python
python3 pybsnn_demo.py

# 3. run the C++ runtime (native build on board)
cd samples/vision/yolov8/runtime/cpp
bash build.sh && sh run_demo.sh
```

For accuracy evaluation, batch benchmarking and model conversion, see the
`evaluator/`, `utils/tools/` and `conversion/` documentation inside each sample.

## Sample structure

Every sample follows the same layout:

```
samples/vision/<model>/
├── README.md / README_cn.md     # overview + variant index
├── conversion/                  # archived ONNX/checkpoints + BSTNNX 8.3.1 YAML examples
├── evaluator/                   # accuracy/performance report + scripts
├── runtime/python/              # Python runtime (pybsnn_demo.py)
├── runtime/cpp/                 # C++ runtime (where available)
└── test_data/                   # input image + on-device output image
```

Artifact naming: `{model}_{task}_{chip}_{WxH}_{inputfmt}.{ext}`,
e.g. `yolov8n_det_c1200_640x640_nv12.bin`.

## Dependencies

- **BSNN runtime** (`libbsnnrt.so`): shipped with the board system image.
- **PyBSNN** (Python bindings, aarch64 only): committed in this repository at
  `pybsnn/` (C1200 wheel); `init_env.sh` installs it from the repo.
- **Model weights**: quantized model packages are committed in-repo under
  each sample's `model/` directory (C1200 and Kalos/A1000).
- **Conversion assets**: selected ONNX and archived checkpoints are committed
  under each sample's `conversion/assets/`; see the
  [SHA-256 manifest](./docs/manifests/conversion_assets.yaml). Calibration
  images are supplied separately.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the mandatory sample layout,
naming convention and licensing rules.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).
Third-party assets keep their own licenses; see each sample's documentation.
