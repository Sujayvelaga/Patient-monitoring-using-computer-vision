"""
╔══════════════════════════════════════════════════════════════════════════╗
║  TRACKER — ByteTrack Multi-Person Tracking + Identity Lock (Novel 2)   ║
║  Assigns persistent IDs; locks onto patient; filters visitors          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import logging
from typing import Optional, Dict, List
from collections import deque

log = logging.getLogger(__name__)


class TrackedPerson:
    """State for a single tracked person."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.positions = deque(maxlen=100)   # (timestamp, centroid_x, centroid_y)
        self.is_patient = False
        self.bbox = None
        self.keypoints = None
        self.bbox_confidence = 0.0
        self.visible_keypoints = 0

    def update(self, pose_data: dict):
        """Update with new detection data."""
        self.last_seen = time.time()
        self.bbox = pose_data.get("bbox")
        self.keypoints = pose_data.get("keypoints")
        self.bbox_confidence = pose_data.get("bbox_confidence", 0.0)
        self.visible_keypoints = pose_data.get("visible_count", 0)

        if self.bbox:
            cx = (self.bbox[0] + self.bbox[2]) / 2
            cy = (self.bbox[1] + self.bbox[3]) / 2
            self.positions.append((self.last_seen, cx, cy))

    @property
    def centroid(self):
        if self.bbox:
            return ((self.bbox[0] + self.bbox[2]) / 2,
                    (self.bbox[1] + self.bbox[3]) / 2)
        return None

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def is_active(self) -> bool:
        return (time.time() - self.last_seen) < 3.0


class IdentityTracker:
    """
    Multi-person tracker with patient identity lock.

    NOVEL CONTRIBUTION #2:
    - Maintains tracked persons from ByteTrack results
    - Allows operator to "lock" a specific person as the patient
    - All downstream modules only receive locked patient data
    - Prevents false alerts from nurses/visitors
    """

    def __init__(self, track_buffer: float = 5.0):
        self.persons: Dict[int, TrackedPerson] = {}
        self.locked_patient_id: Optional[int] = None
        self.track_buffer = track_buffer  # Seconds before removing lost tracks
        self._lock_candidates = []
        self._last_lost_log_time = 0.0  # Throttle "patient lost" warnings

    def update(self, tracked_poses: list):
        """
        Update tracker with pose estimation results (with track IDs).

        Args:
            tracked_poses: list from PoseEstimator.estimate_with_tracking()
        """
        seen_ids = set()

        for pose in tracked_poses:
            track_id = pose.get("track_id")
            if track_id is None:
                continue

            seen_ids.add(track_id)

            if track_id not in self.persons:
                self.persons[track_id] = TrackedPerson(track_id)
                log.info(f"New person detected: ID #{track_id}")

            self.persons[track_id].update(pose)
            self.persons[track_id].is_patient = (
                track_id == self.locked_patient_id)

        # Clean up stale tracks
        stale = [tid for tid, p in self.persons.items()
                 if (time.time() - p.last_seen) > self.track_buffer
                 and tid not in seen_ids]
        for tid in stale:
            if tid == self.locked_patient_id:
                now = time.time()
                if now - self._last_lost_log_time > 10.0:
                    log.warning(f"Locked patient #{tid} lost! "
                                f"Keeping lock for re-identification.")
                    self._last_lost_log_time = now
            else:
                del self.persons[tid]

    def lock_patient(self, track_id: int) -> bool:
        """Lock onto a specific person as the patient."""
        if track_id in self.persons:
            # Unlock previous
            if self.locked_patient_id is not None:
                prev = self.persons.get(self.locked_patient_id)
                if prev:
                    prev.is_patient = False

            self.locked_patient_id = track_id
            self.persons[track_id].is_patient = True
            log.info(f"Patient locked: ID #{track_id}")
            return True
        log.warning(f"Cannot lock ID #{track_id}: not found")
        return False

    def lock_nearest_to(self, x: float, y: float) -> Optional[int]:
        """Lock the person nearest to (x, y) click coordinates."""
        best_id = None
        best_dist = float('inf')

        for tid, person in self.persons.items():
            if not person.is_active or person.centroid is None:
                continue
            cx, cy = person.centroid
            dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None:
            self.lock_patient(best_id)
        return best_id

    def unlock_patient(self):
        """Remove patient lock."""
        if self.locked_patient_id is not None:
            prev = self.persons.get(self.locked_patient_id)
            if prev:
                prev.is_patient = False
            log.info(f"Patient unlocked: ID #{self.locked_patient_id}")
            self.locked_patient_id = None

    def auto_lock_first(self) -> Optional[int]:
        """Automatically lock the first/longest-present person."""
        if self.locked_patient_id is not None:
            return self.locked_patient_id

        active = [(tid, p) for tid, p in self.persons.items()
                  if p.is_active]
        if active:
            # Lock the person who has been present longest
            tid, _ = max(active, key=lambda x: x[1].duration)
            self.lock_patient(tid)
            return tid
        return None

    @property
    def patient(self) -> Optional[TrackedPerson]:
        """Get locked patient data. Returns None if no patient locked."""
        if self.locked_patient_id is not None:
            p = self.persons.get(self.locked_patient_id)
            if p and p.is_active:
                return p
        return None

    @property
    def active_persons(self) -> List[TrackedPerson]:
        """Get all currently active persons."""
        return [p for p in self.persons.values() if p.is_active]

    @property
    def person_count(self) -> int:
        return len([p for p in self.persons.values() if p.is_active])

    def get_context(self) -> dict:
        """Build context dict for downstream detectors."""
        patient = self.patient
        ctx = {
            "patient_id": self.locked_patient_id,
            "person_count": self.person_count,
            "all_persons": self.active_persons,
        }

        if patient and patient.keypoints is not None:
            ctx["keypoints"] = patient.keypoints
            ctx["patient_bbox"] = patient.bbox
            ctx["patient_confidence"] = patient.bbox_confidence
            ctx["patient_centroid"] = patient.centroid
            ctx["visible_keypoints"] = patient.visible_keypoints
        else:
            ctx["keypoints"] = None
            ctx["patient_bbox"] = None
            ctx["patient_confidence"] = 0.0
            ctx["patient_centroid"] = None
            ctx["visible_keypoints"] = 0

        return ctx

    def get_status(self) -> dict:
        """Status dict for dashboard."""
        return {
            "locked_patient_id": self.locked_patient_id,
            "active_count": self.person_count,
            "persons": {
                tid: {
                    "is_patient": p.is_patient,
                    "centroid": p.centroid,
                    "duration": round(p.duration, 1),
                    "confidence": round(p.bbox_confidence, 2),
                }
                for tid, p in self.persons.items()
                if p.is_active
            }
        }
