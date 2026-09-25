import cv2
import numpy as np
import os
import sys
from pybsnn import BSNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class YOLOv5Demo:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model = BSNNLite(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        orig_anchors = [
            [[10, 13], [16, 30], [33, 23]],      # stride 8
            [[30, 61], [62, 45], [59, 119]],     # stride 16
            [[116, 90], [156, 198], [373, 326]]  # stride 32
        ]
        self.anchors = np.array([orig_anchors[0], orig_anchors[1], orig_anchors[2]], dtype=np.float32)

    def preprocess(self, img):
        h, w = img.shape[:2]
        r = min(self.input_size[0] / h, self.input_size[1] / w)
        sw, sh = int(round(w * r)), int(round(h * r))
        dw, dh = (self.input_size[1] - sw) / 2, (self.input_size[0] - sh) / 2
        inter = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(inter, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.uint8)

    def postprocess(self, raw_outputs):
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for i, stride in enumerate(self.strides):
            out = raw_outputs[i] 
            shape = list(out.shape)
            if 255 in shape or 256 in shape:
                c_idx = shape.index(255) if 255 in shape else shape.index(256)
                if c_idx != len(shape) - 1:
                    axes = [d for d in range(len(shape)) if d != c_idx] + [c_idx]
                    out = out.transpose(*axes)
            
            H_d, W_d, _ = out.shape
            out = out[..., :255].reshape(H_d, W_d, 3, 85)
            out = out.transpose(2, 0, 1, 3) # -> (3, H, W, 85)
            
            out = sigmoid(out)
            conf = out[..., 4] 
            cls_p = out[..., 5:] 
            scores = conf[..., None] * cls_p
            
            class_ids = np.argmax(scores, axis=-1)
            max_scores = np.max(scores, axis=-1)
            print(f"Max score at stride {stride}: {max_scores.max()}")
            
            mask = max_scores > self.conf_threshold
            a_idxs, y_idxs, x_idxs = np.where(mask)
            
            for a, y, x in zip(a_idxs, y_idxs, x_idxs):
                score = max_scores[a, y, x]
                cid = class_ids[a, y, x]
                dx, dy, dw, dh = out[a, y, x, :4]
                
                cx = (dx * 2 - 0.5 + x) * stride
                cy = (dy * 2 - 0.5 + y) * stride
                pw = (dw * 2)**2 * self.anchors[i][a][0]
                ph = (dh * 2)**2 * self.anchors[i][a][1]
                
                all_boxes.append([cx - pw/2, cy - ph/2, pw, ph])
                all_scores.append(score)
                all_class_ids.append(cid)

        if not all_boxes: return []
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_class_ids = np.array(all_class_ids)

        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), 
                                  self.conf_threshold, self.iou_threshold)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = all_boxes[idx]
                results.append({'box': [x, y, x + w, y + h], 'score': all_scores[idx], 'class_id': all_class_ids[idx]})
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
        results = self.postprocess(raw_outputs)
        return results, img_orig

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(
        base_dir,
        "yolov5m",
        inner_model=True,
        legacy_candidates=("yolov5_c1200/model", "model/c1200/yolov5m-c1200/model"),
    )
    image_path = resolve_prebuild_asset(base_dir, "139.jpg")
    demo = YOLOv5Demo(model_dir)
    results, img = demo.run(image_path)
    if results is not None:
        print(f"Detected {len(results)} objects.")
        h_orig, w_orig = img.shape[:2]
        input_size = 640
        gain = min(input_size / h_orig, input_size / w_orig)
        pad_w = (input_size - w_orig * gain) / 2
        pad_h = (input_size - h_orig * gain) / 2
        for res in results:
            x1, y1, x2, y2 = res['box']
            x1, y1, x2, y2 = int((x1-pad_w)/gain), int((y1-pad_h)/gain), int((x2-pad_w)/gain), int((y2-pad_h)/gain)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"class {res['class_id']} {res['score']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f" - {label} at [{x1}, {y1}, {x2}, {y2}]")
        cv2.imwrite("pybsnn_result.jpg", img)
        print("Saved result to pybsnn_result.jpg")
