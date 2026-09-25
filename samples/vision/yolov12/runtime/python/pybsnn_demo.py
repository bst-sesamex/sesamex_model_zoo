import cv2
import numpy as np
import os
import sys
from pybsnn import BSNNLite

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class YOLOv12Demo:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model = BSNNLite(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (640, 640)
        self.num_classes = 80
        self.reg_max = 16
        self.strides = [8, 16, 32]
        self.dfl_conv = np.arange(16, dtype=np.float32)

    def preprocess(self, img):
        height, width = img.shape[:2]
        r = min(self.input_size[0] / height, self.input_size[1] / width)
        scale_w, scale_h = int(round(width * r)), int(round(height * r))
        dw, dh = (self.input_size[1] - scale_w) / 2, (self.input_size[0] - scale_h) / 2
        
        inter_img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        padded_img = cv2.copyMakeBorder(inter_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # HWC (BGR) -> HWC (RGB)
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        # HWC (RGB) -> CHW (RGB)
        return np.ascontiguousarray(img_rgb.transpose(2, 0, 1), dtype=np.uint8)

    def postprocess(self, raw_outputs):
        # raw_outputs structure: [head0, head1, head2] (fused)
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for i, stride in enumerate(self.strides):
            out = raw_outputs[i]
            # SDK returns float32, no manual dequantization needed
            out_float = out
            
            # cls part is last 80 channels
            cls_scores = sigmoid(out_float[:, :, 64:])
            
            # Filter by threshold
            mask = np.max(cls_scores, axis=-1) > self.conf_threshold
            y_idxs, x_idxs = np.where(mask)
            
            if len(y_idxs) == 0:
                continue

            for y, x in zip(y_idxs, x_idxs):
                score = np.max(cls_scores[y, x])
                class_id = np.argmax(cls_scores[y, x])
                
                # reg part is first 64 channels (4 * 16)
                ltrb_raw = out_float[y, x, :64].reshape(4, 16)
                ltrb_soft = softmax(ltrb_raw, axis=1)
                ltrb = np.dot(ltrb_soft, self.dfl_conv) # shape (4,)
                
                grid_x, grid_y = x + 0.5, y + 0.5
                x1 = (grid_x - ltrb[0]) * stride
                y1 = (grid_y - ltrb[1]) * stride
                x2 = (grid_x + ltrb[2]) * stride
                y2 = (grid_y + ltrb[3]) * stride
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_class_ids.append(class_id)

        if not all_boxes:
            return []

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_class_ids = np.array(all_class_ids)

        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), 
                                  self.conf_threshold, self.iou_threshold)
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                results.append({
                    'box': all_boxes[idx],
                    'score': all_scores[idx],
                    'class_id': all_class_ids[idx]
                })
        
        return results

    def run(self, image_path):
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            print(f"Error: Could not read image {image_path}")
            return None, None
            
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

    model_dir = resolve_model_dir(base_dir, "yolov12n-det")
    image_path = resolve_prebuild_asset(base_dir, "000000000139.jpg")
    if not os.path.exists(image_path):
        image_path = "./input.jpg"

    print(f"Running YOLOv12 Demo...")
    print(f"Model: {model_dir}")
    print(f"Image: {image_path}")
    
    yolo = YOLOv12Demo(model_dir)
    results, img = yolo.run(image_path)
    
    if results is None:
        sys.exit(1)
        
    print(f"Detected {len(results)} objects.")
    
    # Draw results
    height, width = img.shape[:2]
    gain = min(640 / height, 640 / width)
    pad_w = (640 - width * gain) / 2
    pad_h = (640 - height * gain) / 2
    
    for res in results:
        x1, y1, x2, y2 = res['box']
        # Rescale back to original image
        x1 = int((x1 - pad_w) / gain)
        y1 = int((y1 - pad_h) / gain)
        x2 = int((x2 - pad_w) / gain)
        y2 = int((y2 - pad_h) / gain)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"class {res['class_id']} {res['score']:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f" - {label} at [{x1}, {y1}, {x2}, {y2}]")

    output_path = "pybsnn_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"Saved result to {output_path}")
    
    if len(results) == 0:
        print("Note: No objects detected.")
    else:
        print("Test Passed: Objects detected successfully.")
    sys.exit(0)
