import os
import cv2
import numpy as np
import torch
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class OCREngine:
    """
    Enhanced Vietnamese OCR Engine using PaddleOCR detection + VietOCR recognition.
    This is the default OCR engine for the pipeline, optimized for Vietnamese text.
    Uses PaddleOCR for text region detection and VietOCR for text recognition.
    """
    def __init__(self, output_dir='./paddle_output', device=None):
        """
        Initialize the Enhanced Vietnamese OCR Engine.
        Args:
            output_dir: Directory to save detection outputs
            device: 'cuda' or 'cpu'. If None, auto-detects CUDA availability
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize PaddleOCR detector (detection only)
        try:
            self.detector = PaddleOCR(use_angle_cls=False, lang='en', det=True, rec=False, gpu=(self.device=='cuda'))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR detector: {e}")

        # Initialize VietOCR recognizer
        try:
            config = Cfg.load_config_from_name('vgg_transformer')
            config['cnn']['pretrained'] = False
            config['device'] = self.device
            config['predictor']['beamsearch'] = True
            self.recognizer = Predictor(config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VietOCR recognizer: {e}")

    @staticmethod
    def group_boxes_to_lines(boxes, y_threshold=10):
        if len(boxes) == 0:
            return []
        box_centers = [np.mean([pt[1] for pt in box]) for box in boxes]
        sorted_idx = np.argsort(box_centers)
        boxes_sorted = [boxes[i] for i in sorted_idx]
        centers_sorted = [box_centers[i] for i in sorted_idx]
        lines = []
        current_line = [boxes_sorted[0]]
        current_y = centers_sorted[0]
        for i in range(1, len(boxes_sorted)):
            if abs(centers_sorted[i] - current_y) <= y_threshold:
                current_line.append(boxes_sorted[i])
                current_y = np.mean([current_y, centers_sorted[i]])
            else:
                lines.append(current_line)
                current_line = [boxes_sorted[i]]
                current_y = centers_sorted[i]
        if current_line:
            lines.append(current_line)
        merged_lines = []
        for line in lines:
            all_points = np.vstack(line)
            x_min = np.min(all_points[:, 0])
            y_min = np.min(all_points[:, 1])
            x_max = np.max(all_points[:, 0])
            y_max = np.max(all_points[:, 1])
            merged_lines.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
        return merged_lines

    def paddle_detect(self, image_path):
        # Use PaddleOCR to detect text boxes only
        result = self.detector.ocr(image_path, cls=False)
        # result[0] is a list of (box, _) pairs
        # Handle case where result[0] might be None or empty
        if not result or result[0] is None:
            return {"boxes": []}
        boxes = [np.array(line[0]).astype(int) for line in result[0] if line]
        return {"boxes": boxes}

    def run(self, image_path, visualize=False):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        prediction_result = self.paddle_detect(image_path)
        boxes = prediction_result['boxes']
        boxes_lines = self.group_boxes_to_lines(boxes, y_threshold=10)
        regions = boxes_lines
        if len(regions) == 0:
            print("No text detected!")
            return []
        def get_region_center(box):
            box = np.array(box)
            center_y = np.mean(box[:, 1])
            center_x = np.mean(box[:, 0])
            return (int(center_y // 30), center_x)
        regions_sorted = sorted(regions, key=get_region_center)
        results = []
        img_pil = Image.fromarray(img_rgb)
        for idx, region in enumerate(regions_sorted):
            try:
                box = np.array(region, dtype=np.int32)
                x_min = max(0, np.min(box[:, 0]) - 5)
                x_max = min(w, np.max(box[:, 0]) + 5)
                y_min = max(0, np.min(box[:, 1]) - 5)
                y_max = min(h, np.max(box[:, 1]) + 5)
                if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                    continue
                cropped = img_pil.crop((x_min, y_min, x_max, y_max))
                text = self.recognizer.predict(cropped)
                if text.strip():
                    results.append({
                        'id': idx + 1,
                        'text': text,
                        'box': box.tolist(),
                        'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                        'confidence': 0.95
                    })
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(regions_sorted)} regions...")
            except Exception as e:
                print(f"Error processing region {idx}: {e}")
                continue
        print(f"\n✓ Successfully extracted {len(results)} text segments")
        if visualize and len(results) > 0:
            self.visualize_results(img_rgb, results)
        return results

    @staticmethod
    def visualize_results(img_rgb, results):
        from PIL import ImageDraw, ImageFont
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        for result in results:
            box = np.array(result['box'])
            text = result['text']
            idx = result['id']
            points = [tuple(p) for p in box]
            draw.polygon(points, outline='#00FF00', width=2)
            label = f"[{idx}] {text[:30]}..." if len(text) > 30 else f"[{idx}] {text}"
            x = int(np.min(box[:, 0]))
            y = int(np.min(box[:, 1])) - 20
            try:
                bbox = draw.textbbox((x, y), label, font=font)
                draw.rectangle(bbox, fill='white', outline='red')
            except:
                pass
            draw.text((x, y), label, fill='red', font=font)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 20))
        plt.imshow(img_pil)
        plt.axis('off')
        plt.title(f'Detected {len(results)} text regions')
        plt.show()