"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BASE DETECTOR — Abstract interface for all detection modules          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """Standardized result from any detector."""
    status: str                        # e.g., "NORMAL", "FALL DETECTED", "LOW"
    confidence: float = 0.0            # [0.0, 1.0]
    detail: str = ""                   # Human-readable detail
    severity: str = "LOW"              # LOW, MEDIUM, HIGH, CRITICAL
    should_alert: bool = False         # Whether to trigger alert
    metadata: dict = None              # Additional module-specific data

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDetector(ABC):
    """Abstract base class for all detection modules."""

    def __init__(self, name: str):
        self.name = name
        self._enabled = True
        self._last_result = DetectionResult(status="INITIALIZING")

    @abstractmethod
    def detect(self, frame: np.ndarray, context: dict) -> DetectionResult:
        """
        Run detection on a frame.

        Args:
            frame: BGR image
            context: shared context dict with keys like:
                - "keypoints": (17, 3) array for tracked patient
                - "patient_bbox": (x1, y1, x2, y2)
                - "patient_id": int
                - "all_detections": list of all YOLO detections
                - "quality_score": float environmental quality

        Returns:
            DetectionResult with status, confidence, and metadata
        """
        pass

    @abstractmethod
    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detection results onto frame."""
        pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def last_result(self) -> DetectionResult:
        return self._last_result

    def process(self, frame: np.ndarray, context: dict) -> DetectionResult:
        """Full pipeline: detect + store result."""
        if not self._enabled:
            return DetectionResult(status="DISABLED", confidence=0.0)
        result = self.detect(frame, context)
        self._last_result = result
        return result
