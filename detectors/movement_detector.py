"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MOVEMENT DETECTOR — Semantic No-Movement Detection (Novel 4)          ║
║  Tracks 17 joint coordinates with weighted importance                  ║
║  Detects true body stasis ignoring background noise                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import numpy as np
import logging
from collections import deque
from detectors.base_detector import BaseDetector, DetectionResult
from utils.geometry import weighted_joint_displacement, KEYPOINT_NAMES

log = logging.getLogger(__name__)


class MovementDetector(BaseDetector):
    """
    NOVEL CONTRIBUTION #4: Semantic No-Movement Detection.

    Unlike pixel-difference methods that trigger on fan/curtain movement:
    1. Tracks 17 body joint coordinates from pose estimation
    2. Computes weighted displacement (core joints weighted 2x)
    3. Uses sliding window for temporal smoothing
    4. Only flags "no movement" if actual body joints remain static
    """

    # Default joint weights (core joints more important)
    DEFAULT_WEIGHTS = {
        "nose": 1.0,
        "left_eye": 0.5, "right_eye": 0.5,
        "left_ear": 0.5, "right_ear": 0.5,
        "left_shoulder": 2.0, "right_shoulder": 2.0,
        "left_elbow": 1.5, "right_elbow": 1.5,
        "left_wrist": 1.5, "right_wrist": 1.5,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_knee": 1.5, "right_knee": 1.5,
        "left_ankle": 1.0, "right_ankle": 1.0,
    }

    def __init__(self, config=None):
        super().__init__("movement_detector")

        cfg = config or {}
        self.no_movement_seconds = cfg.get("no_movement_seconds", 30.0)
        self.motion_threshold = cfg.get("motion_threshold", 4.0)
        self.window_size = cfg.get("sliding_window_size", 60)
        custom_weights = cfg.get("joint_weights", self.DEFAULT_WEIGHTS)

        # Build weight array matching COCO 17-keypoint order
        self._weights = np.array([
            custom_weights.get(name, 1.0) for name in KEYPOINT_NAMES
        ])

        # State
        self._prev_joints = None
        self._displacement_history = deque(maxlen=self.window_size)
        self._score_history = deque(maxlen=200)
        self._last_motion_time = time.time()
        self._elapsed_still = 0.0
        self._current_score = 0.0
        self._is_still = False

    def detect(self, frame: np.ndarray, context: dict) -> DetectionResult:
        keypoints = context.get("keypoints")

        if keypoints is None or len(keypoints) < 17:
            return DetectionResult(
                status="NO PERSON",
                confidence=0.0,
                detail="No pose data available"
            )

        # Extract joint positions (only visible ones)
        curr_joints = keypoints[:, :2].copy()  # (17, 2) — x, y
        visibility = keypoints[:, 2]  # (17,) — confidence per joint

        # Mask invisible joints
        visible_mask = visibility > 0.3
        visible_count = int(np.sum(visible_mask))

        if visible_count < 4:
            return DetectionResult(
                status="LOW VISIBILITY",
                confidence=0.1,
                detail=f"Only {visible_count}/17 joints visible"
            )

        if self._prev_joints is None:
            self._prev_joints = curr_joints.copy()
            return DetectionResult(
                status="INITIALIZING",
                confidence=0.0,
                detail="Collecting baseline data"
            )

        # Compute weighted displacement (only for visible joints)
        effective_weights = self._weights * visible_mask.astype(float)
        displacement = weighted_joint_displacement(
            self._prev_joints, curr_joints, effective_weights)

        self._current_score = displacement
        self._displacement_history.append(displacement)
        self._score_history.append({
            "time": time.time(),
            "score": displacement
        })
        self._prev_joints = curr_joints.copy()

        # Check if there's significant movement
        if displacement > self.motion_threshold:
            self._last_motion_time = time.time()

        self._elapsed_still = time.time() - self._last_motion_time
        no_movement = self._elapsed_still >= self.no_movement_seconds

        # ── Confidence Calculation ──
        # Based on: how long still + consistency of stillness + visibility
        time_ratio = min(self._elapsed_still / self.no_movement_seconds, 1.0)
        consistency = 0.0
        if len(self._displacement_history) > 5:
            recent = list(self._displacement_history)[-10:]
            still_frames = sum(1 for d in recent if d < self.motion_threshold)
            consistency = still_frames / len(recent)

        visibility_factor = min(visible_count / 10.0, 1.0)

        confidence = (
            0.4 * time_ratio +
            0.35 * consistency +
            0.25 * visibility_factor
        ) if no_movement else (0.1 * time_ratio)

        self._is_still = no_movement

        if no_movement:
            severity = "CRITICAL" if self._elapsed_still > self.no_movement_seconds * 2 \
                else "HIGH"
            return DetectionResult(
                status="NO MOVEMENT",
                confidence=confidence,
                detail=(f"No movement for {self._elapsed_still:.0f}s "
                        f"(threshold: {self.no_movement_seconds:.0f}s). "
                        f"Score: {displacement:.2f}"),
                severity=severity,
                should_alert=True,
                metadata={
                    "elapsed": self._elapsed_still,
                    "threshold": self.no_movement_seconds,
                    "score": displacement,
                    "visible_joints": visible_count,
                    "progress": time_ratio,
                }
            )
        else:
            return DetectionResult(
                status="ACTIVE",
                confidence=confidence,
                detail=f"Movement score: {displacement:.2f}, "
                       f"still for {self._elapsed_still:.0f}s",
                severity="LOW",
                metadata={
                    "elapsed": self._elapsed_still,
                    "threshold": self.no_movement_seconds,
                    "score": displacement,
                    "visible_joints": visible_count,
                    "progress": time_ratio,
                }
            )

    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        import cv2
        from utils.drawing import COLORS

        h, w = frame.shape[:2]
        meta = result.metadata or {}
        progress = meta.get("progress", 0.0)
        elapsed = meta.get("elapsed", 0.0)
        threshold = meta.get("threshold", self.no_movement_seconds)

        # Progress bar at bottom
        bar_w = w - 20
        bar_x = 10
        bar_y = h - 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                       (35, 35, 35), -1)

        r_val = int(255 * progress)
        g_val = int(200 * (1 - progress))
        fill_w = int(bar_w * progress)
        color = (0, g_val, r_val)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8),
                       color, -1)

        # Label
        label = f"Still: {elapsed:.0f}s / {threshold:.0f}s"
        cv2.putText(frame, label, (bar_x, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["white"], 1)

        # Motion score sparkline
        scores = [s["score"] for s in self._score_history]
        if len(scores) > 1:
            max_s = max(scores) or 1
            spark_w = min(len(scores), 200)
            for i, s in enumerate(scores[-spark_w:]):
                bh_px = int(15 * s / max_s)
                bx = 10 + i
                c = COLORS["red"] if self._is_still else (
                    COLORS["green"] if s > self.motion_threshold else COLORS["dim"])
                cv2.line(frame, (bx, h - 22), (bx, h - 22 - bh_px), c, 1)

        if result.status == "NO MOVEMENT":
            from utils.drawing import draw_alert_overlay
            frame = draw_alert_overlay(frame, "NO MOVEMENT", result.severity)

        return frame

    def reset(self):
        """Reset movement tracking."""
        self._prev_joints = None
        self._displacement_history.clear()
        self._score_history.clear()
        self._last_motion_time = time.time()
        self._elapsed_still = 0.0
        self._is_still = False
        log.info("Movement detector reset")

    @property
    def elapsed_still(self) -> float:
        return self._elapsed_still

    @property
    def current_score(self) -> float:
        return self._current_score
