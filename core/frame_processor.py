"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FRAME PROCESSOR — Preprocessing & Environmental Quality Assessment    ║
║  Supports Hybrid Vision Intelligence (Novel 1)                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


class FrameProcessor:
    """Preprocessing pipeline and environmental quality scoring."""

    def __init__(self, target_width: int = 960, target_height: int = 540):
        self.target_width = target_width
        self.target_height = target_height
        self._quality_score = 1.0

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame for processing."""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        return resized

    def assess_quality(self, frame: np.ndarray) -> float:
        """
        Compute environmental quality score [0.0 - 1.0].
        Used by Hybrid Intelligence to decide DL vs classical.

        Factors:
          - Brightness: too dark or overexposed = bad
          - Blur: motion blur or out of focus = bad
          - Contrast: low contrast = hard to detect features
        """
        if frame is None:
            return 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness score (ideal: 80-180)
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 40:
            brightness_score = mean_brightness / 40.0
        elif mean_brightness > 220:
            brightness_score = max(0, (255 - mean_brightness) / 35.0)
        else:
            brightness_score = 1.0

        # Blur score (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 20:
            blur_score = 0.1
        elif laplacian_var < 100:
            blur_score = laplacian_var / 100.0
        else:
            blur_score = 1.0

        # Contrast score (standard deviation of pixel values)
        std_dev = float(np.std(gray))
        if std_dev < 15:
            contrast_score = std_dev / 15.0
        elif std_dev > 80:
            contrast_score = 1.0
        else:
            contrast_score = min(1.0, std_dev / 50.0)

        # Weighted composite
        self._quality_score = (
            0.4 * brightness_score +
            0.35 * blur_score +
            0.25 * contrast_score
        )
        return self._quality_score

    @property
    def quality_score(self) -> float:
        return self._quality_score

    def get_quality_breakdown(self, frame: np.ndarray) -> dict:
        """Detailed quality breakdown for dashboard display."""
        if frame is None:
            return {"brightness": 0, "blur": 0, "contrast": 0, "overall": 0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return {
            "brightness": float(np.mean(gray)),
            "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "contrast": float(np.std(gray)),
            "overall": self._quality_score
        }
