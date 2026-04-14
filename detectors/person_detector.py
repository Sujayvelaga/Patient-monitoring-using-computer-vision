"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PERSON DETECTOR — YOLOv8 for person + bottle detection                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import logging
from typing import List, Optional
from ultralytics import YOLO

log = logging.getLogger(__name__)


class PersonDetector:
    """
    YOLOv8 object detector for persons and IV bottles.
    Uses ultralytics YOLO with built-in GPU/CPU auto-selection.
    """

    # COCO class IDs: 0=Person, 39=Bottle
    PERSON_CLASS = 0
    BOTTLE_CLASS = 39

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.45,
                 device: str = ""):
        self.confidence_threshold = confidence_threshold
        self.device = device

        log.info(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        log.info("YOLOv8 model loaded")

        self._last_results = None

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run YOLO detection.

        Returns:
            dict with keys:
              - "persons": list of {"bbox": [x1,y1,x2,y2], "confidence": float}
              - "bottles": list of {"bbox": [x1,y1,x2,y2], "confidence": float}
              - "raw_results": ultralytics Results object
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        self._last_results = results

        persons = []
        bottles = []

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()

                    if cls_id == self.PERSON_CLASS:
                        persons.append({
                            "bbox": bbox,
                            "confidence": conf
                        })
                    elif cls_id == self.BOTTLE_CLASS:
                        bottles.append({
                            "bbox": bbox,
                            "confidence": conf
                        })

        return {
            "persons": persons,
            "bottles": bottles,
            "raw_results": results
        }

    @property
    def last_results(self):
        return self._last_results
