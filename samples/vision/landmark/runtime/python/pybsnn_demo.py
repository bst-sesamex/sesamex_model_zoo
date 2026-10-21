import cv2
import numpy as np
import os
import sys
import time
from pybsnn import BSNNLite


class LandmarkDemo:
    def __init__(self, model_path, input_size=(56, 56)):
        # 以 model 目录为单位加载（与 C++ demo 对齐），不在脚本里写死 .bin / .lib 单文件
        self.model = BSNNLite(model_path)
        self.input_size = input_size  # (H, W)

    def preprocess(self, img):
        h, w = img.shape[:2]
        ih, iw = self.input_size
        if (h, w) != (ih, iw):
            img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_LINEAR)
        # 与 yolov5/FCOS demo 一致：BGR HWC -> CHW，dtype 由 pybsnn 按模型 meta 处理
        return np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.uint8)

    def postprocess(self, raw_outputs, input_shape):
        # 关键点输出布局：扁平化的 [x0, y0, x1, y1, ...] 归一化坐标
        if not raw_outputs:
            return []
        out = np.asarray(raw_outputs[0]).reshape(-1)
        n_pairs = out.size // 2
        if n_pairs == 0:
            return []
        coords_norm = out[: n_pairs * 2].reshape(n_pairs, 2)
        ih, iw = input_shape
        points = []
        for nx, ny in coords_norm:
            points.append((float(nx) * iw, float(ny) * ih))
        return points

    def run(self, image_path):
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        img_input = self.preprocess(img_orig)
        _ = self.model.inference([img_input])
        t0 = time.perf_counter()
        raw_outputs = self.model.inference([img_input])
        infer_time = (time.perf_counter() - t0) * 1000
        print(f"-> bsnn model inference time : {infer_time:.3f} ms")
        points = self.postprocess(raw_outputs, self.input_size)
        return points, img_orig


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(
        base_dir,
        "size56",
        extra_candidates=(
            "model/c1200/size56",
            "model/a1000/size56-a1000",
        ),
    )
    image_path = resolve_prebuild_asset(base_dir, "crop_bgr_56.png")

    demo = LandmarkDemo(model_dir, input_size=(56, 56))
    points, img = demo.run(image_path)

    if points:
        print(f"Detected {len(points)} keypoints; first 3:")
        for i, (px, py) in enumerate(points[:3]):
            print(f" - kp[{i}] = ({px:.2f}, {py:.2f})")

    sys.exit(0)
