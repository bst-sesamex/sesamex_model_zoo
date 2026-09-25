import cv2
import numpy as np
import os
import sys
from pybsnn import BSNNLite

class CondLaneNetDemo:
    def __init__(self, model_path):
        self.model = BSNNLite(model_path)
        self.crop_bbox = [0, 160, 1280, 720]
        self.input_size = (800, 320)
        self.input_pitch = 832
        self.hm_thr = 0.5
        self.nms_thr = 3.0
        self.mask_downscale = 8.0

    def preprocess(self, img):
        x1, y1, x2, y2 = self.crop_bbox
        img_cropped = img[y1:y2, x1:x2]
        # HWC (320, 800, 3) -> CHW uint8. pybsnn packs WHD with W pitch=832.
        img_resized = cv2.resize(img_cropped, self.input_size, interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(img_resized.transpose(2, 0, 1), dtype=np.uint8)

    def softmax(self, x):
        e_x = np.exp(x - np.mean(x))
        return e_x / (e_x.sum(axis=1, keepdims=True) + 1e-6)

    def nms_seeds(self, seeds):
        if not seeds: return []
        seeds = sorted(seeds, key=lambda x: x['score'], reverse=True)
        keep = []
        for s in seeds:
            too_close = False
            for k in keep:
                dist = np.sqrt((s['w'] - k['w'])**2 + (s['h'] - k['h'])**2)
                if dist < self.nms_thr:
                    too_close = True
                    break
            if not too_close:
                keep.append(s)
        return keep

    def postprocess(self, raw_outputs):
        # bsnnrt 0.13.0 / pybsnn already dequantizes to float32 HWD.
        hm = np.asarray(raw_outputs[0], dtype=np.float32)
        hm_maxpool = np.asarray(raw_outputs[1], dtype=np.float32)
        
        # Standardize HM to (H, W, D) -> (20, 50, 1)
        if hm.shape[0] != 20:
            if hm.shape[2] == 20: hm = hm.transpose(2, 1, 0)
            elif hm.shape[1] == 20: hm = hm.transpose(1, 0, 2)
        
        if hm_maxpool.shape[0] != 20:
            if hm_maxpool.shape[2] == 20: hm_maxpool = hm_maxpool.transpose(2, 1, 0)
            elif hm_maxpool.shape[1] == 20: hm_maxpool = hm_maxpool.transpose(1, 0, 2)
        
        peaks = (hm == hm_maxpool) & (hm > self.hm_thr)
        h_idxs, w_idxs, _ = np.where(peaks)

        seeds = []
        for h, w in zip(h_idxs, w_idxs):
            seeds.append({'h': h, 'w': w, 'score': hm[h, w, 0]})
        
        seeds = self.nms_seeds(seeds)

        # Standardize Multi-Channel tensors to (D, H, W) -> (1000, 40, 100)
        def to_dhw(t, expected_d):
            if t.shape[0] == expected_d: return t
            if t.shape[2] == expected_d: return t.transpose(2, 0, 1)
            if t.shape[1] == expected_d: return t.transpose(1, 2, 0)
            return t

        regs_raw = to_dhw(raw_outputs[2], 1000)
        masks_raw = to_dhw(raw_outputs[3], 1000)
        fr_raw = to_dhw(raw_outputs[4], 1000)

        lanes = []
        for s in seeds:
            h, w = s['h'], s['w']
            pos = h * 50 + w
            if pos >= 1000: continue
            
            m = masks_raw[pos] # (40, 100)
            r = regs_raw[pos]  # (40, 100)
            f = fr_raw[pos]    # (2, 40)
            
            valid_mask = (f[1] > f[0])
            valid_rows = np.where(valid_mask)[0]
            if len(valid_rows) < 2: continue
            
            m_s = self.softmax(m)
            px = np.sum(m_s * np.arange(100), axis=1)
            
            pts = []
            for row in range(valid_rows[0], valid_rows[-1] + 1):
                col = px[row]
                c_idx = int(col)
                if c_idx < 0: c_idx = 0
                if c_idx > 99: c_idx = 99
                off = r[row, c_idx]
                pts.append([(col + off) * self.mask_downscale, row * self.mask_downscale])
            lanes.append(pts)

        rx = 1280 / 800.0
        ry = 560 / 320.0
        results = []
        for l in lanes:
            transformed = [[p[0] * rx, p[1] * ry + 160] for p in l]
            results.append(transformed)
        return results

    def run(self, image_path):
        img_orig = cv2.imread(image_path)
        if img_orig is None: return None, None
        img_input = self.preprocess(img_orig)
        import time
        _ = self.model.inference([img_input])
        t0 = time.perf_counter()
        raw_outputs = self.model.inference([img_input])
        infer_time = (time.perf_counter() - t0) * 1000
        print(f"-> bsnn model inference time : {infer_time:.3f} ms")
        return self.postprocess(raw_outputs), img_orig

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(
        base_dir,
        "condlanenet",
        inner_model=True,
        legacy_candidates=("condlanenet_model_v1/model", "model/c1200/condlanenet-c1200/model"),
    )
    image_path = resolve_prebuild_asset(base_dir, "0352_19.jpg")
    
    demo = CondLaneNetDemo(model_dir)
    results, img = demo.run(image_path)
    
    if results:
        print(f"Detected {len(results)} lanes.")
        for l in results:
            pts = np.array(l, np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 0), 2)
        cv2.imwrite("pybsnn_result.jpg", img)
        print("Saved result to pybsnn_result.jpg")
        sys.exit(0)
    print("No lanes detected.")
    sys.exit(1)
