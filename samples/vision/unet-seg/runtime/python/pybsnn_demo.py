import os
import sys
import time

import cv2
import numpy as np

from pybsnn import BSNNLite


class UnetSegDemo:
    def __init__(self, model_path, input_size=(512, 512)):
        # `model_path` 必须是 prebuild/model/<sub> 整目录路径——BSNNLite 会从中解析 .Enrsd.bin / .meta /
        # *_stripped.lib / weights.bin 等成员（与 C++ demo 的 BSNN_MODEL(model_dir) 行为一致）。
        self.model = BSNNLite(model_path)
        # input_size = (W, H)；unet_512x512 模型的输入恰为 512x512，C++ 预处理使用 BGR raw 不归一化。
        self.input_size = input_size

    def preprocess(self, img):
        w, h = self.input_size
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        # BGR raw, no normalization (与 C++ unet_preprocess 对齐)；CHW 输出供 BSNNLite 内部按模型 dtype 处理。
        return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.uint8)

    @staticmethod
    def argmax_mask(raw_output):
        """从 BSNNLite 输出中按通道维取 argmax，返回 (H, W) uint8 类别图。

        BSNN 推理输出形状常见为 (C, W, H) 或 (C, H, W)（DWH/WHD），也可能是 (1, C, H, W)。
        这里识别最后两个维度作为空间维，取剩余维上 argmax。
        """
        arr = np.asarray(raw_output)
        # 去掉前导单维 (e.g. batch=1)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"unexpected output ndim {arr.ndim}, shape={arr.shape}")
        # 假设第一个维度是通道（DWH/CHW 风格）；如果通道在最后则先转一下
        if arr.shape[0] > arr.shape[-1] and arr.shape[-1] <= 32:
            # 形如 (H, W, C) -> 转回 (C, H, W)
            arr = arr.transpose(2, 0, 1)
        cls_map = np.argmax(arr, axis=0).astype(np.uint8)
        return cls_map

    def run(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        tensor = self.preprocess(img)
        _ = self.model.inference([tensor])
        t0 = time.perf_counter()
        raw_outputs = self.model.inference([tensor])
        infer_ms = (time.perf_counter() - t0) * 1000.0
        # 这一行被 scripts/run_all_pybsnn_benchmark.py 的 RE_BSNN_INFER_MS 捕获。
        print(f"-> bsnn model inference time : {infer_ms:.3f} ms")
        return raw_outputs, img


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(
        base_dir,
        "unet_512x512",
        legacy_candidates=(
            "model/c1200/unet_512x512",
            "model/a1000/unet_512x512-a1000",
            "model/unet_512x512",
        ),
    )
    image_path = resolve_prebuild_asset(base_dir, "test.png")

    demo = UnetSegDemo(model_dir, input_size=(512, 512))
    raw_outputs, img = demo.run(image_path)

    if not raw_outputs:
        print("no model output produced", file=sys.stderr)
        sys.exit(1)

    # 可选后处理：argmax -> 灰度类别图，缩放回原图尺寸保存。
    try:
        cls_map = UnetSegDemo.argmax_mask(raw_outputs[0])
    except Exception as exc:  # noqa: BLE001
        print(f"argmax skipped: {exc}", file=sys.stderr)
        sys.exit(0)

    h_orig, w_orig = img.shape[:2]
    cls_resized = cv2.resize(cls_map, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    out_path = os.path.join(os.path.dirname(os.path.abspath(image_path)), "pybsnn_seg_result.png")
    if not cv2.imwrite(out_path, cls_resized):
        print(f"failed to save {out_path}", file=sys.stderr)
        sys.exit(1)
    unique, counts = np.unique(cls_resized, return_counts=True)
    top = sorted(zip(unique.tolist(), counts.tolist()), key=lambda kv: -kv[1])[:5]
    print(f"Saved seg mask to {out_path} (shape={cls_resized.shape}, top labels={top})")
