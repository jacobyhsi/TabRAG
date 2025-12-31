from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances

import os
import cv2
import torch
import shutil
import pymupdf

from ditod import add_vit_config

CONFIG_PATH = './object_detection/publaynet_configs/cascade/cascade_dit_large.yaml'
WEIGHTS_PATH = ['MODEL.WEIGHTS', 'publaynet_dit-l_cascade.pth']
visualize = True # Toggle on to save a visualization of identified layout sections

class LayoutProcessor:
    def __init__(self):
        # Step 1: instantiate config
        self.cfg = get_cfg()
        add_vit_config(self.cfg)
        self.cfg.merge_from_file(CONFIG_PATH)
        
        # Step 2: add model weights URL to config
        self.cfg.merge_from_list(WEIGHTS_PATH)
        
        # Step 3: set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device

        # Step 4: define model
        self.predictor = DefaultPredictor(self.cfg)

    def extract_high_res_image(self, path, page_number, output_path, dpi=288):
        ext = os.path.splitext(path)[1].lower()

        if ext in [".pdf"]:
            # Handle PDF
            doc = pymupdf.open(path)
            page = doc[page_number]
            zoom = dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(output_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            # Handle image - just copy
            shutil.copy(path, output_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def preprocess(self, file_path, threshold=0.3, visualize=True):
        """
        Convert one provided PDF or image file to layout-segmented images.
        PDF mode → loops over pages.
        Image mode → single page only.
        """
        ext = os.path.splitext(file_path)[1].lower()
        layout_components = {}

        # -------------------------
        # Handle single images
        # -------------------------
        if ext in [".jpg", ".jpeg", ".png"]:
            filename = os.path.splitext(file_path)[0]
            processed_path = os.path.join('processed', filename.split('datasets/')[1])
            os.makedirs(processed_path, exist_ok=True)

            img = cv2.imread(file_path)
            if img is None:
                print(f"[WARN] Could not read image: {file_path}")
                return layout_components
            img_h, img_w, _ = img.shape

            md = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
            md.set(thing_classes=["text", "title", "list", "table", "figure"])

            with torch.no_grad():
                output = self.predictor(img)["instances"]
            output_dict = output.to('cpu').get_fields()
            keep = output_dict['scores'] > threshold

            layout_components[0] = []
            for i in range(len(keep)):
                if keep[i]:
                    bbox = output_dict['pred_boxes'][i]
                    score = output_dict['scores'][i]
                    pred_class = output_dict['pred_classes'][i]
                    x_min, y_min, x_max, y_max = map(round, bbox.tensor.cpu().numpy()[0])

                    # clamp to image bounds
                    x_min = max(0, min(x_min, img_w - 1))
                    x_max = max(0, min(x_max, img_w))
                    y_min = max(0, min(y_min, img_h - 1))
                    y_max = max(0, min(y_max, img_h))

                    if x_max <= x_min or y_max <= y_min:
                        continue

                    bbox_scaled = [x_min, y_min, x_max, y_max]
                    cropped = img[y_min:y_max, x_min:x_max]
                    if cropped.size == 0:
                        continue

                    cv2.imwrite(f"{processed_path}/{i}.png", cropped)
                    layout_components[0].append({
                        "idx": i,
                        "bbox": bbox_scaled,
                        "score": score.item(),
                        "class": pred_class.item(),
                        "path": f"{processed_path}/{i}.png",
                        "text": ""  # no PDF text
                    })

            if visualize:
                self._save_visualization(img, output, keep, md, filename, 0)

        else:
            raise ValueError(f"Unsupported file format: {ext}. Please preprocess PDFs to images first.")

        return layout_components

    def _save_visualization(self, img, output, keep, md, filename, page_number):
        """Helper to save visualization images."""
        output_dict = output.to('cpu').get_fields()
        image_size = output.image_size
        thresholded_output = Instances(
            image_size=image_size,
            pred_boxes=output_dict['pred_boxes'][keep],
            scores=output_dict['scores'][keep],
            pred_classes=output_dict['pred_classes'][keep],
            pred_masks=output_dict['pred_masks'][keep]
        )
        v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
        result = v.draw_instance_predictions(thresholded_output)
        result_image = result.get_image()[:, :, ::-1]
        os.makedirs(f'visualizations/{filename}', exist_ok=True)
        cv2.imwrite(f'visualizations/{filename}/{page_number}.jpg', result_image)