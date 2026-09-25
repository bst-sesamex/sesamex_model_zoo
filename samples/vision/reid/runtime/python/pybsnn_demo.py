#!/usr/bin/env python3
r"""Minimal pybsnn demo for the reid model.

Loads the mobilenetv2 reid model directory under the sibling C++ prebuild tree
and runs a single forward pass over the canonical sample input image. Prints a
single-line BSNN inference timing message that matches
``RE_BSNN_INFER_MS = r"bsnn\s+model\s+inference\s+time\s*:\s*([\d.]+)"`` used by
``scripts/run_all_pybsnn_benchmark.py`` so the script integrates with the
existing python benchmark harness.

The script intentionally avoids pinning a specific ``.bin`` / ``.lib`` filename;
``BSNNLite`` accepts the model directory and resolves the artefacts internally,
matching how the C++ demo binary loads the model.

Errors propagate (no silent catches): a missing model directory, missing input
file, or runtime failure raises and exits non-zero, which lets
``run_all_pybsnn_benchmark.py`` surface the failure as ``FAIL(<code>)``.
"""

import os
import sys
import time

import cv2
import numpy as np

from pybsnn import BSNNLite


class ReIDDemo:
    """Single-image forward pass wrapper around BSNNLite."""

    # Conventional ReID network input geometry (height, width) used by the
    # mobilenetv2 reid model packaged under ``Deploy/C++/prebuild/model/``.
    DEFAULT_INPUT_SIZE = (256, 128)

    def __init__(self, model_path, input_size=DEFAULT_INPUT_SIZE):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model directory not found: {model_path}"
            )
        self.model = BSNNLite(model_path)
        self.input_size = input_size  # (H, W)

    def preprocess(self, img):
        """Resize BGR image to model input size and convert to CHW uint8.

        Mirrors ``models/reid/Deploy/C++/code/src/image_process.cpp::reid_preprocess``
        for the INT8/UINT8 path: INTER_LINEAR resize, BGR pixel order, no
        normalisation, channel-first layout.
        """
        h, w = self.input_size
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.uint8)

    def run(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"cv2.imread returned None for: {image_path}")

        x = self.preprocess(img)

        _ = self.model.inference([x])
        t0 = time.perf_counter()
        raw_outputs = self.model.inference([x])
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # This stdout line is the contract observed by
        # scripts/run_all_pybsnn_benchmark.py via RE_BSNN_INFER_MS.
        print(f"-> bsnn model inference time : {infer_ms:.3f} ms")

        return raw_outputs


def _flatten_outputs(raw_outputs):
    """Concatenate all output tensors into a flat float32 vector."""
    flat = []
    for arr in raw_outputs:
        flat.append(np.asarray(arr, dtype=np.float32).reshape(-1))
    if not flat:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(flat, axis=0)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(base_dir, "mobilenetv2", legacy_candidates=("model/mobilenetv2",))
    image_path = resolve_prebuild_asset(base_dir, "-1_c1s1_001351_04.jpg")

    demo = ReIDDemo(model_dir)
    raw_outputs = demo.run(image_path)

    # Optional informational diagnostic: feature dim + L2 norm of the flattened
    # output. Not consumed by RE_BSNN_INFER_MS, kept light so failures here are
    # genuine signal rather than noise.
    feat = _flatten_outputs(raw_outputs)
    if feat.size > 0:
        l2 = float(np.linalg.norm(feat))
        print(f"-> feature dim: {feat.size}, L2 norm: {l2:.6f}")

    sys.exit(0)
