import numpy as np
import onnxruntime as ort
from PIL import Image

class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img, imgsz):
        """Resizes image while maintaining aspect ratio with padding."""
        w, h = img.size
        scale = min(imgsz / w, imgsz / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = img.resize((nw, nh), Image.BILINEAR)
        
        # Create a gray background (standard YOLO padding)
        new_img = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2
        new_img.paste(img_resized, (pad_x, pad_y))
        
        return new_img, scale, (pad_x, pad_y)

    def predict(self, image_path: str) -> list[dict]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        
        x, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        x = (np.array(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        
        out = self.session.run(None, {self.input_name: x})[0]
        out = out[0]  # Shape: (300, 6) -> [x1, y1, x2, y2, score, cls]

        results = []
        for x1, y1, x2, y2, score, cls in out:
            if score < self.conf:
                continue
            
            # Map predictions back to original pixels
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            
            # Clip to image bounds
            x1 = max(0.0, min(orig_w, x1))
            y1 = max(0.0, min(orig_h, y1))
            x2 = max(0.0, min(orig_w, x2))
            y2 = max(0.0, min(orig_h, y2))
            
            results.append({
                "xmin": float(x1), "ymin": float(y1),
                "xmax": float(x2), "ymax": float(y2),
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
        return results