import cv2
import numpy as np
import os
import sys
from pybsnn import BSNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-6)

class YOLOv8SegDemo:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model = BSNNLite(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        self.dfl_conv = np.arange(16, dtype=np.float32)
        self.num_classes = 80
        self.num_masks = 32
        self.proto_size = (160, 160)

    def preprocess(self, img):
        height, width = img.shape[:2]
        r = min(self.input_size[0] / height, self.input_size[1] / width)
        scale_w, scale_h = int(round(width * r)), int(round(height * r))
        dw, dh = (self.input_size[1] - scale_w) / 2, (self.input_size[0] - scale_h) / 2
        
        inter_img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        padded_img = cv2.copyMakeBorder(inter_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img_rgb.transpose(2, 0, 1), dtype=np.uint8)

    def postprocess(self, raw_outputs):
        # raw_outputs[0,1,2] -> Det (144 depth: 64 DFL + 80 cls)
        # raw_outputs[3,4,5] -> Mask Coeffs (32 depth)
        # raw_outputs[6] -> Proto (160x160x32, UINT8)
        
        det_heads = raw_outputs[0:3]
        coeff_heads = raw_outputs[3:6]
        proto = raw_outputs[6].astype(np.float32) # (160, 160, 32)
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_coeffs = []

        for i in range(len(self.strides)):
            stride = self.strides[i]
            det_out = det_heads[i]
            coeff_out = coeff_heads[i]
            
            # Det head: (H, W, 144)
            reg = det_out[:, :, :64]
            cls = det_out[:, :, 64:]
            
            probs = sigmoid(cls)
            mask = np.max(probs, axis=-1) > self.conf_threshold
            y_idxs, x_idxs = np.where(mask)
            
            if len(y_idxs) == 0:
                continue

            for y, x in zip(y_idxs, x_idxs):
                score = np.max(probs[y, x])
                class_id = np.argmax(probs[y, x])
                
                # Box decode
                ltrb_raw = reg[y, x].reshape(4, 16)
                ltrb_soft = softmax(ltrb_raw, axis=1)
                ltrb = np.dot(ltrb_soft, self.dfl_conv)
                
                grid_x, grid_y = x + 0.5, y + 0.5
                x1 = (grid_x - ltrb[0]) * stride
                y1 = (grid_y - ltrb[1]) * stride
                x2 = (grid_x + ltrb[2]) * stride
                y2 = (grid_y + ltrb[3]) * stride
                
                # Mask coeffs
                coeff = coeff_out[y, x].astype(np.float32)
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_class_ids.append(class_id)
                all_coeffs.append(coeff)

        if not all_boxes:
            return []

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_coeffs = np.array(all_coeffs)
        
        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), 
                                  self.conf_threshold, self.iou_threshold)
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                # Process mask for each detected object
                coeff = all_coeffs[idx]
                # proto is (160, 160, 32), coeff is (32,)
                # mask = sigmoid(sum(coeff_i * proto_i))
                mask_raw = sigmoid(np.dot(proto, coeff)) # (160, 160)
                
                results.append({
                    'box': all_boxes[idx],
                    'score': all_scores[idx],
                    'class_id': all_class_ids[idx],
                    'mask_raw': mask_raw # 160x160 map
                })
        
        return results

    def draw_seg(self, img, results):
        height, width = img.shape[:2]
        gain = min(640 / height, 640 / width)
        pad_w = (640 - width * gain) / 2
        pad_h = (640 - height * gain) / 2

        # Color palette for masks
        palette = np.random.randint(0, 255, (self.num_classes + 1, 3), dtype=np.uint8)

        # Base mask for visualization
        mask_overlay = np.zeros_like(img, dtype=np.uint8)

        for res in results:
            x1, y1, x2, y2 = res['box']
            mask_raw = res['mask_raw']
            class_id = res['class_id']
            color = palette[class_id].tolist()
            
            # Rescale box back to original
            ox1 = int((x1 - pad_w) / gain)
            oy1 = int((y1 - pad_h) / gain)
            ox2 = int((x2 - pad_w) / gain)
            oy2 = int((y2 - pad_h) / gain)
            
            cv2.rectangle(img, (ox1, oy1), (ox2, oy2), color, 2)
            
            # Process mask
            # 1. Resize mask back to 640x640 (clipped by padding)
            mask_full = cv2.resize(mask_raw, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            # 2. Threshold
            mask_bool = mask_full > 0.5
            
            # 3. Crop mask by detection box in 640x640 space
            box_mask = np.zeros_like(mask_full, dtype=bool)
            ix1, iy1, ix2, iy2 = [max(0, int(round(v))) for v in [x1, y1, x2, y2]]
            box_mask[iy1:iy2, ix1:ix2] = True
            mask_final = mask_bool & box_mask
            
            # 4. Filter out padding from mask
            mask_final = mask_final[int(round(pad_h)):int(round(640-pad_h)), 
                                   int(round(pad_w)):int(round(640-pad_w))]
            
            # 5. Resize to original image size
            if mask_final.size > 0:
                mask_res = cv2.resize(mask_final.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                mask_overlay[mask_res > 0] = color

        # Alpha blend
        img = cv2.addWeighted(img, 0.6, mask_overlay, 0.4, 0)
        return img

    def run(self, image_path):
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            return None, None
            
        img_input = self.preprocess(img_orig)
        import time
        _ = self.model.inference([img_input])
        t0 = time.perf_counter()
        raw_outputs = self.model.inference([img_input])
        infer_time = (time.perf_counter() - t0) * 1000
        print(f"-> bsnn model inference time : {infer_time:.3f} ms")
        results = self.postprocess(raw_outputs)
        img_vis = self.draw_seg(img_orig.copy(), results)
        return results, img_vis

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(base_dir, "yolov8n-seg")
    image_path = resolve_prebuild_asset(base_dir, "bus.jpg")
    if not os.path.exists(image_path): image_path = "./input.jpg"

    print(f"Running YOLOv8 Seg Demo...")
    demo = YOLOv8SegDemo(model_dir)
    results, img_vis = demo.run(image_path)
    
    if results is not None:
        cv2.imwrite("pybsnn_seg_result.jpg", img_vis)
        print(f"Detected {len(results)} objects with masks. Saved to pybsnn_seg_result.jpg")
