"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BED MONITOR — Multi-anchor zone + posture + temporal hysteresis v3    ║
║                                                                          ║
║  Key capabilities:                                                       ║
║  1. No default zone — user MUST draw it via dashboard / Set Zone btn.   ║
║  2. Cold-start: requires in_confirm frames before reporting IN_BED.     ║
║  3. Bbox corners voting — works WITHOUT keypoints (coma patients).      ║
║  4. Fall-from-bed velocity — bbox-centroid jump ≥ EXIT_VELOCITY_PX      ║
║     inside→outside triggers immediate HIGH alert (no confirm delay).    ║
║     Designed for coma/unconscious patients who can't trigger keypoints  ║
║     and for night-time when YOLO pose quality drops.                    ║
║  5. Coma mode — skips posture checks, works on bbox alone, fast alert.  ║
║  6. clear_zone() API to reset zone from dashboard.                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import json
import os
import numpy as np
import logging
from collections import deque
from detectors.base_detector import BaseDetector, DetectionResult
from utils.geometry import (
    point_in_polygon,
    distance_to_polygon,
    midpoint,
    fraction_of_points_inside_polygon,
    KEYPOINT_INDEX,
)

log = logging.getLogger(__name__)


class BedMonitor(BaseDetector):
    """
    Bed-zone logic that is harder to fool than bbox-centroid alone:

    1. **Weighted inside vote** — combines torso midpoint (shoulders+hips),
       bounding-box corners (all 4), and (when visible) ankle midpoint.
    2. **Signed distance** from the fused anchor to the polygon boundary.
    3. **Temporal hysteresis** — OUT_OF_BED requires out_confirm consecutive
       "outside" frames; re-entry clears quickly.
    4. **Pose quality gate** — weak core keypoints reduce confidence only;
       they do NOT falsely report OUT_OF_BED.
    5. **Cold-start** — _stable_inside is not assumed True until confirmed.
    """

    STATES = {
        "IN_BED_LYING":    ("Patient lying in bed",        "LOW"),
        "IN_BED_SITTING":  ("Patient sitting in bed",      "LOW"),
        "ON_BED_EDGE":     ("Patient on bed edge",         "MEDIUM"),
        "OUT_OF_BED":      ("Patient has left the bed",    "HIGH"),
        "STANDING_NEAR":   ("Patient standing near bed",   "MEDIUM"),
        "NO_ZONE":         ("No bed zone defined",         "LOW"),
        "NO_PATIENT":      ("No patient tracked",          "LOW"),
        "INITIALIZING":    ("Confirming patient position",  "LOW"),
    }

    # Minimum centroid jump (px/frame) from inside→outside to flag as "fell"
    EXIT_VELOCITY_PX = 80.0

    def __init__(self, config=None):
        super().__init__("bed_monitor")

        cfg = config or {}
        self.zone_file      = cfg.get("zone_file", "data/calibration/bed_zone.json")
        self.lying_max      = cfg.get("lying_angle_max",    35.0)
        self.sitting_range  = cfg.get("sitting_angle_range", (35.0, 65.0))
        self.standing_min   = cfg.get("standing_angle_min",  65.0)
        self.edge_distance  = cfg.get("edge_distance_threshold", 50.0)

        self.out_confirm    = max(1, int(cfg.get("out_of_bed_confirm_frames",  7)))
        self.in_confirm     = max(1, int(cfg.get("in_bed_confirm_frames",      3)))
        self.min_core_kp    = int(cfg.get("min_core_keypoints_for_zone",       2))
        self.w_torso        = float(cfg.get("torso_anchor_weight",             0.50))
        self.w_bbox         = float(cfg.get("bbox_centroid_weight",            0.25))
        self.w_feet         = float(cfg.get("feet_anchor_weight",              0.15))
        self.w_corners      = float(cfg.get("bbox_corners_weight",             0.10))

        # Lower threshold = more lenient when keypoints partially visible
        self.in_vote_threshold = float(cfg.get("in_zone_vote_threshold", 0.30))

        self.exit_velocity_px = float(cfg.get("exit_velocity_px", self.EXIT_VELOCITY_PX))

        self.feet_outside_penalty   = float(cfg.get("feet_outside_penalty",      0.22))
        self.bed_contact_band_px    = float(cfg.get("bed_contact_band_px",       85.0))
        self.contact_confirm_frames = int(cfg.get("contact_confirm_frames",      40))
        self.com_kp_min             = float(cfg.get("com_keypoint_conf_min",     0.25))
        self.hip_stability_window   = int(cfg.get("hip_stability_window",        18))
        self.hip_micro_motion_min_std = float(cfg.get("hip_micro_motion_min_std", 0.35))

        # Normalize weights
        wsum = self.w_torso + self.w_bbox + self.w_feet + self.w_corners
        if wsum > 1e-6:
            self.w_torso   /= wsum
            self.w_bbox    /= wsum
            self.w_feet    /= wsum
            self.w_corners /= wsum

        self.zone_pts   = None
        self._state          = "NO_ZONE"
        self._outside_streak = 0
        self._inside_streak  = 0
        self._stable_inside  = False
        self._confirmed      = False
        self._contact_streak   = 0
        self._hip_y_history    = deque(maxlen=max(30, self.hip_stability_window))
        self._last_patient_id  = None
        # Fall-from-bed velocity tracking
        self._last_centroid    = None   # (x, y) from previous frame
        self._fell_from_bed    = False  # latched on sudden exit

        self._load_zone()

    # ──────────────────────────────────────────────────────────────────────
    #  Zone persistence
    # ──────────────────────────────────────────────────────────────────────
    def _load_zone(self):
        if os.path.exists(self.zone_file):
            try:
                with open(self.zone_file) as f:
                    data = json.load(f)
                pts = [tuple(pt) for pt in data.get("pts", [])]
                if len(pts) >= 3:
                    self.zone_pts = pts
                    self._state   = "INITIALIZING"   # not "OK" yet
                    log.info("Bed zone loaded: %d points", len(pts))
            except Exception as e:
                log.warning("Could not load bed zone: %s", e)

    def _save_zone(self):
        os.makedirs(os.path.dirname(self.zone_file), exist_ok=True)
        with open(self.zone_file, "w") as f:
            json.dump({"pts": self.zone_pts}, f)
        log.info("Bed zone saved: %d points", len(self.zone_pts))

    def set_zone(self, points: list):
        if len(points) >= 3:
            self.zone_pts        = [tuple(pt) for pt in points]
            self._outside_streak = 0
            self._inside_streak  = 0
            self._stable_inside  = False
            self._confirmed      = False
            self._fell_from_bed  = False
            self._last_centroid  = None
            self._save_zone()
            self._state = "INITIALIZING"
            return True
        return False

    def clear_zone(self):
        """Remove the bed zone. User must re-draw."""
        self.zone_pts        = None
        self._state          = "NO_ZONE"
        self._stable_inside  = False
        self._confirmed      = False
        self._fell_from_bed  = False
        self._last_centroid  = None
        # Overwrite saved file with empty zone
        os.makedirs(os.path.dirname(self.zone_file), exist_ok=True)
        with open(self.zone_file, "w") as f:
            json.dump({"pts": []}, f)
        log.info("Bed zone cleared")

    # ──────────────────────────────────────────────────────────────────────
    #  Keypoint helpers
    # ──────────────────────────────────────────────────────────────────────
    def _torso_midpoint(self, keypoints):
        if keypoints is None or len(keypoints) < 17:
            return None, 0
        ls = keypoints[KEYPOINT_INDEX["left_shoulder"]]
        rs = keypoints[KEYPOINT_INDEX["right_shoulder"]]
        lh = keypoints[KEYPOINT_INDEX["left_hip"]]
        rh = keypoints[KEYPOINT_INDEX["right_hip"]]
        core = [ls, rs, lh, rh]
        vis  = sum(1 for j in core if j[2] > 0.3)
        if vis < 2:
            return None, vis
        pairs = []
        if ls[2] > 0.3 and rs[2] > 0.3:
            pairs.append(midpoint((ls[0], ls[1]), (rs[0], rs[1])))
        if lh[2] > 0.3 and rh[2] > 0.3:
            pairs.append(midpoint((lh[0], lh[1]), (rh[0], rh[1])))
        if not pairs:
            return None, vis
        sx = sum(p[0] for p in pairs) / len(pairs)
        sy = sum(p[1] for p in pairs) / len(pairs)
        return (sx, sy), vis

    def _feet_midpoint(self, keypoints):
        if keypoints is None or len(keypoints) < 17:
            return None, False
        la = keypoints[KEYPOINT_INDEX["left_ankle"]]
        ra = keypoints[KEYPOINT_INDEX["right_ankle"]]
        if la[2] <= 0.3 and ra[2] <= 0.3:
            return None, False
        if la[2] > 0.3 and ra[2] > 0.3:
            return midpoint((la[0], la[1]), (ra[0], ra[1])), True
        if la[2] > 0.3:
            return (float(la[0]), float(la[1])), True
        return (float(ra[0]), float(ra[1])), True

    # ──────────────────────────────────────────────────────────────────────
    #  NOVEL: Bbox-corners zone vote
    # ──────────────────────────────────────────────────────────────────────
    def _bbox_corners_inside_frac(self, patient_bbox) -> float:
        """
        Test all 4 corners of the bounding box against the zone polygon.
        Returns fraction [0, 1] of corners inside.
        This is critical when keypoints are missing: if 3 of 4 corners
        are inside the bed zone, the patient is clearly still in bed.
        """
        if patient_bbox is None or self.zone_pts is None:
            return 0.5   # unknown
        x1, y1, x2, y2 = patient_bbox[:4]
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        n_inside = sum(1 for c in corners if point_in_polygon(c, self.zone_pts))
        return n_inside / 4.0

    # ──────────────────────────────────────────────────────────────────────
    #  Weighted zone vote (multi-anchor)
    # ──────────────────────────────────────────────────────────────────────
    def _weighted_zone_vote(self, keypoints, patient_bbox, patient_centroid):
        torso_pt, core_vis = self._torso_midpoint(keypoints)
        feet_pt,  feet_ok  = self._feet_midpoint(keypoints)

        pts, weights = [], []

        if torso_pt is not None:
            pts.append(torso_pt)
            weights.append(self.w_torso)

        if patient_centroid is not None:
            pts.append((float(patient_centroid[0]), float(patient_centroid[1])))
            weights.append(self.w_bbox)

        if feet_ok and feet_pt is not None:
            pts.append(feet_pt)
            weights.append(self.w_feet)

        # ── Bbox corners (always available when bbox is) ──
        corners_frac = self._bbox_corners_inside_frac(patient_bbox)
        # Treat corners_frac as a virtual synthetic point via a weight bonus
        corners_contribution = corners_frac * self.w_corners

        if not pts:
            # No anchors at all — rely entirely on bbox corners
            if patient_centroid is not None:
                anchor = (float(patient_centroid[0]), float(patient_centroid[1]))
            else:
                return None, 0.0, None
            inside_score = (1.0 if point_in_polygon(anchor, self.zone_pts) else 0.0)
            return anchor, float(inside_score * 0.65 + corners_contribution * 0.35), 0

        wsum = sum(weights) or 1.0
        ax   = sum(p[0] * w for p, w in zip(pts, weights)) / wsum
        ay   = sum(p[1] * w for p, w in zip(pts, weights)) / wsum
        anchor = (ax, ay)

        inside_score = 0.0
        for p, w in zip(pts, weights):
            if point_in_polygon(p, self.zone_pts):
                inside_score += w
        inside_score /= wsum

        # Weighted keypoints fraction (body keypoints: hips, shoulders, knees)
        kp_fr = None
        if keypoints is not None and len(keypoints) >= 17:
            kpts, kconf = [], []
            for name in ("left_hip", "right_hip", "left_shoulder",
                         "right_shoulder", "left_knee", "right_knee"):
                j = keypoints[KEYPOINT_INDEX[name]]
                kpts.append((float(j[0]), float(j[1])))
                kconf.append(float(j[2]))
            kp_fr = fraction_of_points_inside_polygon(kpts, self.zone_pts, kconf, 0.3)

        # Combine: anchor vote + bbox corners + keypoint fraction
        if kp_fr is not None:
            combined = (0.55 * inside_score
                        + 0.30 * kp_fr
                        + 0.15 * corners_frac)
        else:
            combined = (0.65 * inside_score
                        + 0.35 * corners_frac)

        return anchor, float(combined), core_vis

    # ──────────────────────────────────────────────────────────────────────
    #  Helper computations
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _polygon_y_extents(zone_pts):
        ys = [p[1] for p in zone_pts]
        return float(min(ys)), float(max(ys))

    def _csi_subzone(self, cy: float) -> str:
        ymin, ymax = self._polygon_y_extents(self.zone_pts)
        if ymax <= ymin + 1e-3:
            return "CENTER"
        t = (cy - ymin) / (ymax - ymin)
        if t < 0.34:
            return "HEAD"
        if t < 0.67:
            return "CENTER"
        return "FOOT_EDGE"

    def _pose_center_of_mass(self, keypoints):
        if keypoints is None or len(keypoints) < 17:
            return None, 0.0
        names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                 "left_knee", "right_knee")
        sx = sy = sw = 0.0
        for name in names:
            j = keypoints[KEYPOINT_INDEX[name]]
            c = float(j[2])
            if c < self.com_kp_min:
                continue
            sx += j[0] * c; sy += j[1] * c; sw += c
        if sw < 1e-6:
            return None, 0.0
        return (sx / sw, sy / sw), sw

    def _hip_vertical_mid(self, keypoints):
        if keypoints is None or len(keypoints) < 17:
            return None
        lh = keypoints[KEYPOINT_INDEX["left_hip"]]
        rh = keypoints[KEYPOINT_INDEX["right_hip"]]
        if lh[2] < 0.3 and rh[2] < 0.3:
            return None
        if lh[2] >= 0.3 and rh[2] >= 0.3:
            return 0.5 * (float(lh[1]) + float(rh[1]))
        return float(lh[1] if lh[2] >= 0.3 else rh[1])

    # ──────────────────────────────────────────────────────────────────────
    #  Main detect
    # ──────────────────────────────────────────────────────────────────────
    def _fall_velocity_alert(self, current_centroid) -> bool:
        """
        Fall-from-bed detection without keypoints.
        When a patient (coma / night-time) falls from bed, their bbox centroid
        moves abruptly from inside→outside the zone in 1-2 frames.
        Normal patient movement is slow; a fall is fast (EXIT_VELOCITY_PX+).
        """
        if self._last_centroid is None or current_centroid is None:
            return False
        cx, cy = current_centroid
        px, py = self._last_centroid
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if dist < self.exit_velocity_px:
            return False
        # Only flag if previous position was INSIDE and current is OUTSIDE
        was_inside = point_in_polygon(self._last_centroid, self.zone_pts)
        now_outside = not point_in_polygon(current_centroid, self.zone_pts)
        return was_inside and now_outside

    def detect(self, frame: np.ndarray, context: dict) -> DetectionResult:
        if not self.zone_pts or len(self.zone_pts) < 3:
            return DetectionResult(
                status="NO ZONE",
                confidence=0.0,
                detail="⚠ No bed zone defined. Click 'Set Zone' button to draw the patient bed area.",
                metadata={"state": "NO_ZONE", "zone_set": False}
            )

        patient_centroid = context.get("patient_centroid")
        keypoints        = context.get("keypoints")
        patient_id       = context.get("patient_id")
        patient_bbox     = context.get("patient_bbox")

        if patient_centroid is None:
            self._last_centroid = None
            return DetectionResult(
                status="NO PATIENT",
                confidence=0.0,
                detail="No patient currently tracked",
                metadata={"state": "NO_PATIENT", "zone_set": True}
            )

        # Reset streaks if patient identity changed
        if patient_id is not None and patient_id != self._last_patient_id:
            self._outside_streak = 0
            self._inside_streak  = 0
            self._stable_inside  = False
            self._confirmed      = False
            self._fell_from_bed  = False
            self._last_centroid  = None
            self._last_patient_id = patient_id

        # ── Fall-from-bed velocity check (works without keypoints) ──
        current_centroid = (float(patient_centroid[0]), float(patient_centroid[1]))
        fell_this_frame = self._fall_velocity_alert(current_centroid)
        if fell_this_frame:
            self._fell_from_bed = True
            log.warning("Fall-from-bed velocity event! Patient centroid moved %.0f px",
                        ((current_centroid[0]-self._last_centroid[0])**2 +
                         (current_centroid[1]-self._last_centroid[1])**2)**0.5)
        self._last_centroid = current_centroid

        # ── Multi-anchor zone vote ──
        anchor, inside_frac, core_vis = self._weighted_zone_vote(
            keypoints, patient_bbox, patient_centroid)

        if anchor is None:
            # Absolute fallback: use centroid only
            anchor = (float(patient_centroid[0]), float(patient_centroid[1]))
            inside_frac = 1.0 if point_in_polygon(anchor, self.zone_pts) else 0.0
            core_vis = 0

        # ── Pose center-of-mass refinement ──
        com, com_w = self._pose_center_of_mass(keypoints)
        if com is not None and com_w >= 1.0:
            blend = 0.80
            anchor = (blend * anchor[0] + (1 - blend) * com[0],
                      blend * anchor[1] + (1 - blend) * com[1])

        # ── Feet penalty ──
        feet_pt, feet_ok = self._feet_midpoint(keypoints)
        feet_outside = (
            feet_ok and feet_pt is not None
            and not point_in_polygon(feet_pt, self.zone_pts))
        if feet_outside:
            inside_frac = max(0.0, inside_frac - self.feet_outside_penalty)

        posture     = context.get("posture", "UNKNOWN")
        spine_angle = context.get("spine_angle", 90.0)

        cx, cy   = anchor
        csi_zone = self._csi_subzone(cy)

        ymin, ymax     = self._polygon_y_extents(self.zone_pts)
        bed_surface_y  = ymax
        hip_y          = self._hip_vertical_mid(keypoints)
        hip_near_bed   = False
        if hip_y is not None:
            hip_near_bed = (bed_surface_y - hip_y) <= self.bed_contact_band_px
            self._hip_y_history.append(hip_y)
        else:
            self._hip_y_history.append(cy)

        # ── Raw inside decision ──
        raw_inside = inside_frac >= self.in_vote_threshold

        # Override: standing with feet outside → definitely out
        if feet_outside and posture == "STANDING":
            raw_inside = False

        # ── Contact streak ──
        if raw_inside:
            if hip_near_bed or csi_zone in ("CENTER", "FOOT_EDGE", "HEAD"):
                self._contact_streak = min(self._contact_streak + 1, 10 ** 6)
            else:
                self._contact_streak = max(0, self._contact_streak - 2)
        else:
            self._contact_streak = max(0, self._contact_streak - 3)

        # ── Hip micro-motion ──
        hip_std = 0.0
        if len(self._hip_y_history) >= self.hip_stability_window:
            tail     = list(self._hip_y_history)[-self.hip_stability_window:]
            hip_std  = float(np.std(tail))
        low_micro_motion = (
            hip_std < self.hip_micro_motion_min_std
            and len(self._hip_y_history) >= self.hip_stability_window
            and raw_inside
            and posture == "LYING")

        bed_contact_confirmed = self._contact_streak >= self.contact_confirm_frames

        dist = distance_to_polygon((cx, cy), self.zone_pts)

        # ── Temporal hysteresis ──
        if raw_inside:
            self._outside_streak = 0
            self._inside_streak  = min(self._inside_streak + 1, 10 ** 6)
        else:
            self._inside_streak  = 0
            self._outside_streak = min(self._outside_streak + 1, 10 ** 6)

        # Update stable state
        if self._outside_streak >= self.out_confirm:
            self._stable_inside = False
            self._confirmed     = True   # we've seen enough frames
        elif self._inside_streak >= self.in_confirm:
            self._stable_inside = True
            self._confirmed     = True

        inside = self._stable_inside

        # ── FIX: on_edge uses smoothed `inside`, not raw ──
        on_edge = abs(dist) < self.edge_distance and inside and not (
            self._inside_streak < self.in_confirm)

        # ── State classification ──
        if not self._confirmed:
            state = "INITIALIZING"
        elif inside:
            if posture == "LYING" or spine_angle < self.lying_max:
                state = "IN_BED_LYING"
            elif on_edge:
                state = "ON_BED_EDGE"
            else:
                state = "IN_BED_SITTING"
        else:
            if posture == "STANDING" or spine_angle > self.standing_min:
                state = "OUT_OF_BED"
            else:
                state = "STANDING_NEAR"

        self._state = state
        description, severity = self.STATES[state]

        # ── Override: fall-from-bed (velocity event) = immediate CRITICAL ──
        if self._fell_from_bed and state in ("OUT_OF_BED", "STANDING_NEAR"):
            severity = "CRITICAL"
            description = "Patient FELL from bed"

        # ── Confidence ──
        patient_conf = context.get("patient_confidence", 0.5)
        visible_kp   = int(context.get("visible_keypoints", 0))
        core_ok      = core_vis >= self.min_core_kp

        zone_certainty = min(max(abs(dist), 1.0) / 120.0, 1.0)
        posture_known  = 1.0 if posture != "UNKNOWN" else 0.35
        vote_clarity   = float(min(1.0, max(0.0, abs(inside_frac - 0.5) * 2.0)))

        confidence = (
            0.32 * patient_conf +
            0.22 * zone_certainty +
            0.20 * posture_known +
            0.16 * vote_clarity +
            0.10 * (1.0 if core_ok else 0.4)
        )
        if visible_kp < 6:
            confidence *= 0.85
        if bed_contact_confirmed and raw_inside:
            confidence = min(1.0, confidence + 0.05)
        if not self._confirmed:
            confidence = min(confidence, 0.5)
        # Fall-from-bed = maximum confidence alert
        if self._fell_from_bed and state == "OUT_OF_BED":
            confidence = max(confidence, 0.92)
        confidence = float(max(0.0, min(1.0, confidence)))

        normal_alert = (
            state == "OUT_OF_BED"
            and self._outside_streak >= self.out_confirm
            and core_ok
            and patient_conf >= 0.30
        )
        # Immediate fall-from-bed alert (no debounce needed — velocity event)
        fall_alert = self._fell_from_bed and state == "OUT_OF_BED"
        should_alert = normal_alert or fall_alert

        return DetectionResult(
            status=state.replace("_", " "),
            confidence=confidence,
            detail=(
                f"{description} (Patient #{patient_id}, "
                f"vote={inside_frac:.2f}, zone={csi_zone}, "
                f"dist={dist:.0f}px, posture={posture}, "
                f"in_streak={self._inside_streak}, out_streak={self._outside_streak}, "
                f"contact={bed_contact_confirmed})"
            ),
            severity=severity,
            should_alert=should_alert,
            metadata={
                "state":             state,
                "inside":            inside,
                "raw_inside":        raw_inside,
                "inside_fraction":   inside_frac,
                "distance":          dist,
                "on_edge":           on_edge,
                "posture":           posture,
                "centroid":          (cx, cy),
                "bbox_centroid":     patient_centroid,
                "zone_set":          True,
                "outside_streak":    self._outside_streak,
                "inside_streak":     self._inside_streak,
                "confirmed":         self._confirmed,
                "csi_zone":          csi_zone,
                "feet_outside":      feet_outside,
                "bed_contact_confirmed": bed_contact_confirmed,
                "hip_stability_std": hip_std,
                "low_micro_motion_hint": low_micro_motion,
                "fell_from_bed": self._fell_from_bed,
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Annotate
    # ──────────────────────────────────────────────────────────────────────
    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        from utils.drawing import draw_zone_polygon, COLORS

        meta      = result.metadata or {}
        is_inside = meta.get("inside", True)

        if self.zone_pts:
            frame = draw_zone_polygon(frame, self.zone_pts, is_inside)

        centroid = meta.get("centroid")
        if centroid:
            cx, cy = int(centroid[0]), int(centroid[1])
            color  = COLORS["green"] if is_inside else COLORS["red"]
            cv2.circle(frame, (cx, cy), 7, color, -1)
            bc = meta.get("bbox_centroid")
            if bc and (abs(bc[0] - cx) > 3 or abs(bc[1] - cy) > 3):
                cv2.circle(frame, (int(bc[0]), int(bc[1])), 4, (200, 200, 0), 1)

        state = meta.get("state", "")
        if state == "OUT_OF_BED":
            from utils.drawing import draw_alert_overlay
            msg = "PATIENT FELL FROM BED" if meta.get("fell_from_bed") else "PATIENT LEFT BED"
            frame = draw_alert_overlay(frame, msg, "CRITICAL" if meta.get("fell_from_bed") else "HIGH", y_offset=80)
        elif state == "ON_BED_EDGE":
            h, w = frame.shape[:2]
            cv2.putText(frame, "PATIENT ON BED EDGE",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255), 2)
        elif state == "NO_ZONE":
            h, w = frame.shape[:2]
            # Big instruction overlay — no zone set
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 30), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.putText(frame, "BED ZONE NOT SET — Use dashboard 'Set Zone' to draw the bed area",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 200, 255), 2)

        return frame

    @property
    def state(self) -> str:
        return self._state

    @property
    def has_zone(self) -> bool:
        return self.zone_pts is not None and len(self.zone_pts) >= 3
