"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FALL DETECTOR — Spine-Angle Based Fall Detection (Novel 3)            ║
║  Uses pose keypoints to calculate spine orientation and detect rapid   ║
║  vertical → horizontal transitions with temporal validation            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import numpy as np
import logging
from collections import deque
from detectors.base_detector import BaseDetector, DetectionResult
from utils.geometry import (spine_angle, midpoint, angle_change_rate,
                             KEYPOINT_INDEX)

log = logging.getLogger(__name__)


class FallDetector(BaseDetector):
    """
    NOVEL CONTRIBUTION #3: Pose-Based Fall Detection using Spine Angle.

    Instead of using bounding box aspect ratio (fragile), this detector:
    1. Computes spine vector from mid-hip → mid-shoulder
    2. Calculates angle from horizontal (standing ≈ 90°, fallen ≈ 0°)
    3. Detects rapid vertical→horizontal transitions (Δangle/Δtime)
    4. Requires sustained horizontal position before confirming fall
    5. Outputs confidence based on keypoint visibility + angle certainty
    """

    def __init__(self, config=None):
        super().__init__("fall_detector")

        # Config
        cfg = config or {}
        self.vertical_min = cfg.get("vertical_angle_min", 60.0)
        self.vertical_max = cfg.get("vertical_angle_max", 120.0)
        self.fall_threshold = cfg.get("fall_angle_threshold", 35.0)
        self.confirm_frames = cfg.get("fall_confirm_frames", 8)
        self.rapid_transition = cfg.get("rapid_transition_deg_per_sec", 100.0)
        self.min_keypoints = cfg.get("fall_confidence_min_keypoints", 4)
        self.recovery_frames = cfg.get("recovery_frames", 15)

        # State
        self._angle_history = deque(maxlen=60)
        self._time_history = deque(maxlen=60)
        self._fall_counter = 0
        self._recovery_counter = 0
        self._posture = "UNKNOWN"
        self._posture_history = deque(maxlen=30)
        self._current_angle = 0.0
        self._is_fallen = False
        self._fall_start_time = None
        self._last_mid_shoulder = None
        self._last_mid_hip = None

    def detect(self, frame: np.ndarray, context: dict) -> DetectionResult:
        keypoints = context.get("keypoints")
        bbox = context.get("patient_bbox")
        
        is_horizontal_bbox = False
        if bbox is not None:
            w = max(1.0, bbox[2] - bbox[0])
            h = max(1.0, bbox[3] - bbox[1])
            is_horizontal_bbox = (w / h) > 1.2

        if keypoints is None or len(keypoints) < 17:
            if not is_horizontal_bbox:
                self._fall_counter = max(0, self._fall_counter - 1)
                if self._fall_counter == 0:
                    self._is_fallen = False
                return DetectionResult(
                    status="NO PERSON",
                    confidence=0.0,
                    detail="No pose keypoints available",
                    severity="LOW"
                )
            # If is_horizontal_bbox is true, we skip the return to process it as a fall

        if keypoints is not None and len(keypoints) >= 17:
            # Extract key joints
            l_shoulder = keypoints[KEYPOINT_INDEX["left_shoulder"]]
        r_shoulder = keypoints[KEYPOINT_INDEX["right_shoulder"]]
        l_hip = keypoints[KEYPOINT_INDEX["left_hip"]]
        r_hip = keypoints[KEYPOINT_INDEX["right_hip"]]

        # Check minimum visibility
        if keypoints is not None and len(keypoints) >= 17:
            core_visibility = [l_shoulder[2], r_shoulder[2], l_hip[2], r_hip[2]]
            visible_core = sum(1 for v in core_visibility if v > 0.3)
        else:
            visible_core = 0

        if visible_core < 2 and not is_horizontal_bbox:
            return DetectionResult(
                status="LOW VISIBILITY",
                confidence=0.1,
                detail=f"Only {visible_core}/4 core joints visible",
                severity="LOW"
            )

        # Calculate midpoints
        if keypoints is not None and len(keypoints) >= 17:
            mid_shoulder = midpoint((l_shoulder[0], l_shoulder[1]), (r_shoulder[0], r_shoulder[1]))
            mid_hip = midpoint((l_hip[0], l_hip[1]), (r_hip[0], r_hip[1]))
            angle = spine_angle(mid_shoulder, mid_hip)
        else:
            # Fallback if keypoints are missing but bbox indicates horizontal fall
            mid_shoulder = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1])
            mid_hip = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[3])
            angle = 10.0

        self._last_mid_shoulder = mid_shoulder
        self._last_mid_hip = mid_hip
        self._current_angle = angle

        now = time.time()
        self._angle_history.append(angle)
        self._time_history.append(now)

        # Classify posture
        if angle >= self.vertical_min:
            posture = "STANDING"
        elif angle >= self.fall_threshold:
            posture = "SITTING"
        else:
            posture = "LYING"

        self._posture_history.append(posture)
        # Majority vote for stable posture
        if self._posture_history:
            self._posture = max(set(self._posture_history),
                                key=self._posture_history.count)

        # ── Fall Detection Logic ──

        # Check for rapid transition (degrees per second)
        transition_rate = 0.0
        if len(self._angle_history) >= 2 and len(self._time_history) >= 2:
            transition_rate = angle_change_rate(
                list(self._angle_history), list(self._time_history))

        is_horizontal = (angle < self.fall_threshold) or is_horizontal_bbox
        rapid_drop = transition_rate > self.rapid_transition

        if is_horizontal:
            self._fall_counter += 1
            self._recovery_counter = 0
            if self._fall_start_time is None:
                self._fall_start_time = now
        else:
            self._recovery_counter += 1
            if self._recovery_counter >= self.recovery_frames:
                self._fall_counter = 0
                self._is_fallen = False
                self._fall_start_time = None

        # Confirm fall: sustained horizontal position
        confirmed = self._fall_counter >= self.confirm_frames
        was_fallen = self._is_fallen
        self._is_fallen = confirmed

        # ── Confidence Calculation ──
        # Based on: keypoint visibility, angle certainty, transition speed
        visibility_conf = min(visible_core / 4.0, 1.0)
        angle_conf = max(0, (self.fall_threshold - angle) / self.fall_threshold) \
            if is_horizontal else 0.0
        temporal_conf = min(self._fall_counter / self.confirm_frames, 1.0)

        # Boost confidence if rapid transition was detected
        transition_boost = min(transition_rate / self.rapid_transition, 0.3) \
            if rapid_drop else 0.0

        confidence = (
            0.3 * visibility_conf +
            0.3 * angle_conf +
            0.3 * temporal_conf +
            0.1 * min(1.0, transition_boost)
        ) if is_horizontal else visibility_conf * 0.1

        # ── Build Result ──
        if confirmed:
            fall_duration = now - self._fall_start_time if self._fall_start_time else 0
            severity = "CRITICAL" if confidence > 0.8 else "HIGH"
            return DetectionResult(
                status="FALL DETECTED",
                confidence=confidence,
                detail=(f"Spine angle {angle:.0f}° (threshold {self.fall_threshold}°), "
                        f"sustained {self._fall_counter} frames, "
                        f"duration {fall_duration:.1f}s"),
                severity=severity,
                should_alert=not was_fallen,  # Alert only on new fall
                metadata={
                    "angle": angle,
                    "posture": self._posture,
                    "fall_counter": self._fall_counter,
                    "transition_rate": transition_rate,
                    "mid_shoulder": mid_shoulder,
                    "mid_hip": mid_hip,
                    "fall_duration": fall_duration,
                }
            )
        else:
            return DetectionResult(
                status=f"{self._posture}",
                confidence=confidence,
                detail=f"Spine angle: {angle:.0f}°, Posture: {self._posture}",
                severity="LOW",
                metadata={
                    "angle": angle,
                    "posture": self._posture,
                    "fall_counter": self._fall_counter,
                    "transition_rate": transition_rate,
                    "mid_shoulder": mid_shoulder,
                    "mid_hip": mid_hip,
                }
            )

    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        from utils.drawing import draw_spine_angle_indicator, draw_alert_overlay

        meta = result.metadata
        if meta and meta.get("mid_shoulder") and meta.get("mid_hip"):
            draw_spine_angle_indicator(
                frame,
                meta["mid_shoulder"],
                meta["mid_hip"],
                meta.get("angle", 0),
                is_fallen=result.status == "FALL DETECTED"
            )

        if result.status == "FALL DETECTED":
            frame = draw_alert_overlay(frame, "FALL DETECTED", result.severity)

        return frame

    def reset(self):
        """Reset fall detection state."""
        self._fall_counter = 0
        self._recovery_counter = 0
        self._is_fallen = False
        self._fall_start_time = None
        self._angle_history.clear()
        self._time_history.clear()
        self._posture_history.clear()
        self._posture = "UNKNOWN"
        log.info("Fall detector reset")

    @property
    def current_angle(self) -> float:
        return self._current_angle

    @property
    def posture(self) -> str:
        return self._posture

    @property
    def is_fallen(self) -> bool:
        return self._is_fallen
