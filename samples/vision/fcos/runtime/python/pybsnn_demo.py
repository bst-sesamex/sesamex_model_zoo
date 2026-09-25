import cv2
import numpy as np
import os
import sys
from pybsnn import BSNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FCOSDemo:
    def __init__(self, model_path, conf_threshold=0.6, iou_threshold=0.6):
        self.model = BSNNLite(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (896, 896)
        self.strides = [8, 16, 32, 64, 128]
        self.num_classes = 80

    def preprocess(self, img):
        height, width = img.shape[:2]
        r = min(self.input_size[0] / height, self.input_size[1] / width)
        scale_w, scale_h = int(round(width * r)), int(round(height * r))
        dw, dh = (self.input_size[1] - scale_w) / 2, (self.input_size[0] - scale_h) / 2
        
        inter_img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        padded_img = cv2.copyMakeBorder(inter_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return np.ascontiguousarray(padded_img.transpose(2, 0, 1), dtype=np.uint8)

    def postprocess(self, raw_outputs):
        # raw_outputs indices for FCOS:
        # 0-4: logits, 5-9: reg, 10-14: ctr
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for level in range(5):
            stride = self.strides[level]
            cls = raw_outputs[level]
            reg = raw_outputs[level + 5]
            ctr = raw_outputs[level + 10]
            
            print(f"Level {level}: cls min={cls.min():.2f} max={cls.max():.2f}, reg min={reg.min():.2f} max={reg.max():.2f}")
            ctr_s = sigmoid(ctr)
            cls_s = sigmoid(cls)
            scores = np.sqrt(cls_s * ctr_s)
            
            mask = scores > self.conf_threshold
            y_idxs, x_idxs, c_idxs = np.where(mask)
            
            if len(y_idxs) == 0: continue
            
            # Print top box for each level
            best_idx = np.argmax(scores[mask])
            by, bx, bc = y_idxs[best_idx], x_idxs[best_idx], c_idxs[best_idx]
            bl, bt, br, bb = reg[by, bx]
            print(f"  Best in L{level}: score={scores[by, bx, bc]:.2f} box=[{bl:.1f}, {bt:.1f}, {br:.1f}, {bb:.1f}] at stride {stride}")
            
            mask = scores > self.conf_threshold
            y_idxs, x_idxs, c_idxs = np.where(mask)
            
            if len(y_idxs) == 0: continue

            for y, x, cid in zip(y_idxs, x_idxs, c_idxs):
                score = scores[y, x, cid]
                l, t, r, b = reg[y, x]
                
                loc_x = x * stride + stride / 2.0
                loc_y = y * stride + stride / 2.0
                
                x1 = loc_x - l
                y1 = loc_y - t
                x2 = loc_x + r
                y2 = loc_y + b

                x0 = max(0, x1)
                y0 = max(0, y1)
                x1_c = min(x2, self.input_size[1])
                y1_c = min(y2, self.input_size[0])
                
                if x1_c > x0 and y1_c > y0:
                    all_boxes.append([x0, y0, x1_c - x0, y1_c - y0]) # [x, y, w, h]
                    all_scores.append(float(score))
                    all_class_ids.append(int(cid + 1))

        if not all_boxes: return []

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_class_ids = np.array(all_class_ids)

        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), 
                                  self.conf_threshold, self.iou_threshold)
        
        per_class_results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                per_class_results.append({
                    'box': all_boxes[idx],
                    'score': all_scores[idx],
                    'class_id': all_class_ids[idx]
                })

        # Class-agnostic NMS (matches C++ to remove quantization noise across classes)
        if not per_class_results: return []
        boxes_agnostic = [r['box'] for r in per_class_results]
        scores_agnostic = [r['score'] for r in per_class_results]
        
        indices_agnostic = cv2.dnn.NMSBoxes(boxes_agnostic, scores_agnostic, 
                                            self.conf_threshold, self.iou_threshold)
        
        results = []
        if len(indices_agnostic) > 0:
            for idx in indices_agnostic.flatten():
                x, y, w, h = per_class_results[idx]['box']
                results.append({
                    'box': [x, y, x + w, y + h],
                    'score': per_class_results[idx]['score'],
                    'class_id': per_class_results[idx]['class_id']
                })
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

    model_dir = resolve_model_dir(base_dir, "fcos")
    image_path = resolve_prebuild_asset(base_dir, "000000000139.jpg")
    
    demo = FCOSDemo(model_dir)
    results, img = demo.run(image_path)
    
    if results is not None:
        print(f"Detected {len(results)} objects.")
        input_size = 896
        height, width = img.shape[:2]
        gain = min(input_size / height, input_size / width)
        pad_w = (input_size - width * gain) / 2
        pad_h = (input_size - height * gain) / 2
        
        for res in results:
            x1, y1, x2, y2 = res['box']
            x1 = int((x1 - pad_w) / gain)
            y1 = int((y1 - pad_h) / gain)
            x2 = int((x2 - pad_w) / gain)
            y2 = int((y2 - pad_h) / gain)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"class {res['class_id']} {res['score']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f" - {label} at [{x1}, {y1}, {x2}, {y2}]")
        
        cv2.imwrite("pybsnn_result.jpg", img)
        print("Saved result to pybsnn_result.jpg")
