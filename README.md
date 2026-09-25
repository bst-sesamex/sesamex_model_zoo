English | [简体中文](./README_cn.md)

# SesameX Model Zoo

**Version:** see [VERSION](./VERSION) · **Changelog:** [CHANGELOG.md](./CHANGELOG.md)

SesameX Model Zoo provides deployment samples for mainstream vision models on
Black Sesame edge computing platforms (Aura / Kalos). Each sample covers the
full on-device path: model conversion, quantized model download, Python and
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
| Kalos (A1000 chip) | Some samples provide models for the A1000 chip; see each sample's `model/README`. |

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

| Model | Sample | Task | Aura C1200 FPS | Inference latency |
|---|---|---|---:|---:|
| LightStereo (OpenStereo-2) | [openstereo2](./samples/vision/openstereo2) | stereo matching / depth | 14.64 | 68.291 ms |
| ResNet-18 | [classification](./samples/vision/classification) | image classification | 763.36 | 1.310 ms |
| ResNet-34 | [classification](./samples/vision/classification) | image classification | 490.44 | 2.039 ms |
| MobileNetV2 | [classification](./samples/vision/classification) | image classification | 1464.13 | 0.683 ms |
| CondLaneNet | [condlanenet](./samples/vision/condlanenet) | lane detection | 148.54 | 6.732 ms |
| FCOS-ResNet50 | [fcos](./samples/vision/fcos) | object detection | 50.79 | 19.689 ms |
| YOLOv10n | [yolov12](./samples/vision/yolov12) | object detection | 207.13 | 4.828 ms |
| YOLO11n | [yolov12](./samples/vision/yolov12) | object detection | 218.20 | 4.583 ms |
| YOLOv12n | [yolov12](./samples/vision/yolov12) | object detection | 39.95 | 25.032 ms |
| YOLOv8n-det | [yolov8](./samples/vision/yolov8) | object detection | 392.46 | 2.548 ms |
| YOLOv8n-seg | [yolov8](./samples/vision/yolov8) | instance segmentation | 370.23 | 2.701 ms |
| YOLOv8n-pose | [yolov8](./samples/vision/yolov8) | pose estimation | 399.36 | 2.504 ms |
| YOLOv5m | [yolov5](./samples/vision/yolov5) | object detection | 145.84 | 6.857 ms |
| Face Landmark 56 | [landmark](./samples/vision/landmark) | face landmark (68 pts) | 2403.85 | 0.416 ms |
| Face Landmark 224 | [landmark](./samples/vision/landmark) | face landmark (68 pts) | 1506.02 | 0.664 ms |
| ReID MobileNetV2 | [reid](./samples/vision/reid) | person re-identification | 1785.71 | 0.560 ms |
| ReID OSNet | [reid](./samples/vision/reid) | person re-identification | 203.52 | 4.9135 ms |
| ReID ResNet-50 | [reid](./samples/vision/reid) | person re-identification | 469.04 | 2.132 ms |
| UNet 512x512 | [unet-seg](./samples/vision/unet-seg) | semantic segmentation | 35.51 | 28.158 ms |
| UNet 512x1024 | [unet-seg](./samples/vision/unet-seg) | semantic segmentation | 18.15 | 55.111 ms |

## Quick start

On an aarch64 board (Aura):

```bash
# 1. initialize environment (runtime env + PyBSNN wheel)
chmod +x init_env.sh
./init_env.sh

# 2. download the model package (example: yolov8)
cd samples/vision/yolov8
./model/download.sh

# 3. run the Python runtime (on board)
cd runtime/python
python3 pybsnn_demo.py
```

Model packages are committed in-repo; `download.sh` verifies completeness.
For accuracy evaluation, batch benchmarking and model conversion, see the
`evaluator/`, `utils/tools/` and `conversion/` documentation inside each sample.

## Sample structure

Every sample follows the same layout:

```
samples/vision/<model>/
├── README.md / README_cn.md     # overview + variant index
├── conversion/                  # archived ONNX/checkpoints + BSTNNX 8.3.1 YAML examples
├── evaluator/                   # accuracy/performance report + scripts
├── model/                       # download.sh + model file list
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
- **Model weights**: quantized model packages (C1200) are committed in-repo
  under each sample's `model/c1200/`; `model/download.sh` verifies completeness.
- **Conversion assets**: selected ONNX and archived checkpoints are tracked with Git LFS under each sample's `conversion/assets/`; see the [SHA-256 manifest](./docs/manifests/conversion_assets.yaml). Calibration images are supplied separately.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the mandatory sample layout,
naming convention and licensing rules.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).
Third-party assets keep their own licenses; see each sample's documentation.
