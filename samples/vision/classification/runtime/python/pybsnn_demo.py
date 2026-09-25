import cv2
import numpy as np
import os
import sys
from pybsnn import BSNNLite

class ClassificationDemo:
    def __init__(self, model_path):
        self.model = BSNNLite(model_path)
        self.input_size = (224, 224)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def preprocess(self, img):
        # Resize to 224x224
        resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        # Keep BGR as the model expects
        return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.uint8)

    def postprocess(self, raw_outputs):
        # SDK returns float32, no manual dequantization needed
        logits = raw_outputs[0].flatten()
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Get Top-5
        top5_idx = np.argsort(probs)[-5:][::-1]
        results = []
        for idx in top5_idx:
            results.append({
                'class_id': int(idx),
                'class_name': self.classes[idx] if idx < len(self.classes) else f"class_{idx}",
                'score': float(probs[idx])
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

    model_dir = resolve_model_dir(base_dir, "resnet34", legacy_candidates=("model/resnet34-c1200",))
    image_path = resolve_prebuild_asset(base_dir, "bird_303.png")
    if not os.path.exists(image_path):
        image_path = "./input.png"

    print(f"Running Classification Demo...")
    demo = ClassificationDemo(model_dir)
    results, img = demo.run(image_path)
    
    if results is None:
        sys.exit(1)
        
    print(f"Top-5 Predictions:")
    for i, res in enumerate(results):
        print(f"  {i+1}: {res['class_name']} ({res['score']:.4f})")

    # Save result image
    cv2.imwrite("pybsnn_result.jpg", img)
    print(f"Saved result to pybsnn_result.jpg")

    if len(results) > 0:
        print("Test Passed: Predictions generated successfully.")
        sys.exit(0)
    else:
        print("Test Failed: No predictions.")
        sys.exit(1)
