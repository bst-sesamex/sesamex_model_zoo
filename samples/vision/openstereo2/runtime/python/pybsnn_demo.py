import cv2
import numpy as np
import os
import sys
from pybsnn import BSNNLite

class OpenStereo2Demo:
    def __init__(self, model_path):
        self.model = BSNNLite(model_path)
        self.input_size = (960, 544) # W, H
        self.max_disp = 192.0

    def preprocess(self, img):
        img_res = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        chw = img_rgb.transpose(2, 0, 1)
        return np.ascontiguousarray(chw, dtype=np.uint8)

    def softmax_axis2(self, x):
        # Compute softmax for logits along the last axis
        x_max = np.max(x, axis=2, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / (np.sum(e_x, axis=2, keepdims=True) + 1e-12)

    def postprocess(self, raw_outputs):
        # IMPORTANT: The BSNN Python SDK automatically dequantizes the outputs.
        # Do not multiply by dequant_scale (0.0625) again logic.
        v0 = raw_outputs[0].astype(np.float32)
        v1 = raw_outputs[1].astype(np.float32)
        
        # Apply Softmax to dequantized logits
        s0 = self.softmax_axis2(v0)
        s1 = self.softmax_axis2(v1)
        
        # Dot product consensus
        dot = np.sum(s0 * s1, axis=2)
        print(f"Dot Stats: min={dot.min():.4f}, max={dot.max():.4f}, mean={dot.mean():.4f}")
        
        disparity = dot * self.max_disp
        return disparity

    def create_colormap(self, disparity):
        h, w = disparity.shape
        disp_norm = np.clip(disparity / self.max_disp, 0, 1)
        
        bin_map = np.array([
            [0,0,114], 
            [0,0,185], 
            [1,0,114],
            [1,0,174],
            [0,1,114],
            [0,1,185],
            [1,1,114],
            [1,1,0]
        ], dtype=np.float32)
        
        bins = np.array([114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0], dtype=np.float32)
        cbins = np.cumsum(bins)
        total = cbins[-1]
        
        bins_norm = bins / total
        cbins_norm = cbins / total
        
        colormap = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(len(bins_norm)):
            start = 0 if i == 0 else cbins_norm[i-1]
            end = cbins_norm[i]
            mask = (disp_norm >= start) & (disp_norm < end)
            if i == len(bins_norm) - 1:
                mask |= (disp_norm == 1.0)
            
            ratio = (disp_norm - start) / bins_norm[i]
            ratio = ratio[..., np.newaxis]
            
            c0 = bin_map[i]
            c1 = bin_map[i+1]
            color = c0 * (1.0 - ratio) + c1 * ratio
            colormap[mask] = color[mask]
            
        colormap_bgr = (colormap * 255.0).astype(np.uint8)
        colormap_bgr = colormap_bgr[..., ::-1]
        return colormap_bgr

    def run(self, left_path, right_path):
        l_img = cv2.imread(left_path)
        r_img = cv2.imread(right_path)
        if l_img is None or r_img is None:
            return None
            
        l_in = self.preprocess(l_img)
        r_in = self.preprocess(r_img)
        
        import time
        _ = self.model.inference([l_in, r_in])
        t0 = time.perf_counter()
        raw_outputs = self.model.inference([l_in, r_in])
        infer_time = (time.perf_counter() - t0) * 1000
        print(f"-> bsnn model inference time : {infer_time:.3f} ms")
        disparity = self.postprocess(raw_outputs)
        vis = self.create_colormap(disparity)
        return vis

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(
        base_dir,
        "lightstereo",
        inner_model=True,
        legacy_candidates=(
            "main_c1200_OpenStereo-2/model",
            "model/c1200/lightstereo-c1200/model",
        ),
    )
    left_image = resolve_prebuild_asset(base_dir, "0326_left.png")
    right_image = resolve_prebuild_asset(base_dir, "0326_right.png")
    
    demo = OpenStereo2Demo(model_dir)
    vis = demo.run(left_image, right_image)
    if vis is not None:
        cv2.imwrite("pybsnn_disparity.jpg", vis)
        print("Saved result to pybsnn_disparity.jpg")
