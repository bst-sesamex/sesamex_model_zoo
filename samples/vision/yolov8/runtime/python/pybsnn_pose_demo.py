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

class YOLOv8PoseDemo:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model = BSNNLite(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        self.dfl_conv = np.arange(16, dtype=np.float32)
        
        # COCO skeleton Connections
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        self.pose_palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
        ], dtype=np.uint8)

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
        # raw_outputs[0,1,2] -> Det (65 depth: 64 DFL + 1 cls)
        # raw_outputs[3,4,5] -> Pose (51 depth: 17 * 3)
        all_boxes = []
        all_scores = []
        all_kpts = []

        for i in range(len(self.strides)):
            stride = self.strides[i]
            det_out = raw_outputs[i]
            kpt_out = raw_outputs[i + 3]
            
            # Det head: (H, W, 65)
            reg = det_out[:, :, :64]
            cls = det_out[:, :, 64:] # 1 channel for person
            
            probs = sigmoid(cls[..., 0])
            mask = probs > self.conf_threshold
            y_idxs, x_idxs = np.where(mask)
            
            if len(y_idxs) == 0:
                continue

            for y, x in zip(y_idxs, x_idxs):
                score = probs[y, x]
                
                # Box decode (DFL)
                ltrb_raw = reg[y, x].reshape(4, 16)
                ltrb_soft = softmax(ltrb_raw, axis=1)
                ltrb = np.dot(ltrb_soft, self.dfl_conv)
                
                grid_x, grid_y = x + 0.5, y + 0.5
                x1 = (grid_x - ltrb[0]) * stride
                y1 = (grid_y - ltrb[1]) * stride
                x2 = (grid_x + ltrb[2]) * stride
                y2 = (grid_y + ltrb[3]) * stride
                
                # Keypoint decode
                # kpt_out[y, x] is (51,) -> 17 * (x, y, conf)
                raw_kpts = kpt_out[y, x].reshape(17, 3)
                kpts = np.zeros((17, 3), dtype=np.float32)
                for kp in range(17):
                    kpts[kp, 0] = (raw_kpts[kp, 0] * 2.0 + (grid_x - 0.5)) * stride
                    kpts[kp, 1] = (raw_kpts[kp, 1] * 2.0 + (grid_y - 0.5)) * stride
                    kpts[kp, 2] = sigmoid(raw_kpts[kp, 2])
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_kpts.append(kpts)

        if not all_boxes:
            return []

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        
        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), 
                                  self.conf_threshold, self.iou_threshold)
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                results.append({
                    'box': all_boxes[idx],
                    'score': all_scores[idx],
                    'keypoints': all_kpts[idx]
                })
        
        return results

    def draw_pose(self, img, results):
        height, width = img.shape[:2]
        gain = min(640 / height, 640 / width)
        pad_w = (640 - width * gain) / 2
        pad_h = (640 - height * gain) / 2

        for res in results:
            x1, y1, x2, y2 = res['box']
            kpts = res['keypoints']
            
            # Rescale box
            x1 = int((x1 - pad_w) / gain)
            y1 = int((y1 - pad_h) / gain)
            x2 = int((x2 - pad_w) / gain)
            y2 = int((y2 - pad_h) / gain)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Rescale keypoints
            rescaled_kpts = []
            for i in range(17):
                kx = int((kpts[i, 0] - pad_w) / gain)
                ky = int((kpts[i, 1] - pad_h) / gain)
                conf = kpts[i, 2]
                rescaled_kpts.append([kx, ky, conf])
                if conf > 0.5:
                    cv2.circle(img, (kx, ky), 5, (0, 255, 255), -1)

            # Draw skeleton
            for i, sk in enumerate(self.skeleton):
                p1 = rescaled_kpts[sk[0] - 1]
                p2 = rescaled_kpts[sk[1] - 1]
                if p1[2] > 0.5 and p2[2] > 0.5:
                    color = tuple(map(int, self.pose_palette[i % len(self.pose_palette)]))
                    cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, 2)
                    
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
        img_vis = self.draw_pose(img_orig.copy(), results)
        return results, img_vis

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(base_dir, "../../../../../utils/py_utils/deploy")))
    from pybsnn_paths import resolve_model_dir, resolve_prebuild_asset

    model_dir = resolve_model_dir(base_dir, "yolov8n-pose")
    image_path = resolve_prebuild_asset(base_dir, "bus.jpg")
    if not os.path.exists(image_path): image_path = "./input.jpg"

    print(f"Running YOLOv8 Pose Demo...")
    demo = YOLOv8PoseDemo(model_dir)
    results, img_vis = demo.run(image_path)
    
    if results is not None:
        cv2.imwrite("pybsnn_pose_result.jpg", img_vis)
        print(f"Detected {len(results)} persons. Saved to pybsnn_pose_result.jpg")
