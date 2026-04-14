"""
╔══════════════════════════════════════════════════════════════════════════╗
║  POSE ESTIMATOR — YOLOv8-Pose for 17-keypoint human pose               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import logging
from ultralytics import YOLO

log = logging.getLogger(__name__)


class PoseEstimator:
    """
    YOLOv8-Pose for multi-person 17-keypoint pose estimation.
    Outputs per-person keypoints with confidence scores.
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt",
                 confidence_threshold: float = 0.45,
                 device: str = ""):
        self.confidence_threshold = confidence_threshold
        self.device = device

        log.info(f"Loading YOLOv8-Pose model: {model_path}")
        self.model = YOLO(model_path)
        log.info("YOLOv8-Pose model loaded")

    def estimate(self, frame: np.ndarray) -> list:
        """
        Run pose estimation.

        Returns:
            list of dicts, each with:
              - "bbox": [x1, y1, x2, y2]
              - "bbox_confidence": float
              - "keypoints": (17, 3) array [x, y, confidence]
              - "visible_count": int — number of visible keypoints
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )

        poses = []

        if results and len(results) > 0:
            r = results[0]
            if r.keypoints is not None and r.boxes is not None:
                kps = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
                boxes = r.boxes

                for i in range(len(kps)):
                    keypoints = kps[i]  # (17, 3) — x, y, conf

                    # Count visible keypoints (confidence > 0.3)
                    visible = int(np.sum(keypoints[:, 2] > 0.3))

                    bbox = boxes[i].xyxy[0].cpu().numpy().tolist()
                    bbox_conf = float(boxes[i].conf[0])

                    poses.append({
                        "bbox": bbox,
                        "bbox_confidence": bbox_conf,
                        "keypoints": keypoints,
                        "visible_count": visible
                    })

        return poses

    def estimate_with_tracking(self, frame: np.ndarray,
                                persist: bool = True) -> list:
        """
        Run pose estimation with built-in ByteTrack tracking.

        Returns:
            list of dicts (same as estimate) plus:
              - "track_id": int or None
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False
        )

        poses = []

        if results and len(results) > 0:
            r = results[0]
            if r.keypoints is not None and r.boxes is not None:
                kps = r.keypoints.data.cpu().numpy()
                boxes = r.boxes

                for i in range(len(kps)):
                    keypoints = kps[i]
                    visible = int(np.sum(keypoints[:, 2] > 0.3))

                    bbox = boxes[i].xyxy[0].cpu().numpy().tolist()
                    bbox_conf = float(boxes[i].conf[0])

                    track_id = None
                    if boxes[i].id is not None:
                        track_id = int(boxes[i].id[0])

                    poses.append({
                        "bbox": bbox,
                        "bbox_confidence": bbox_conf,
                        "keypoints": keypoints,
                        "visible_count": visible,
                        "track_id": track_id
                    })

        return poses
