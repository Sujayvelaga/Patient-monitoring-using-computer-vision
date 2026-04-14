import cv2
import json
import os
import time
import numpy as np
import logging
from collections import deque
from detectors.base_detector import BaseDetector, DetectionResult

log = logging.getLogger(__name__)

# ── Histogram shape constants ──────────────────────────────────────────────────
# HSV: [16, 16, 16] bins  →  4096 values
# Gradient (ang × mag): [16, 16] bins  →  256 values
# Combined flat vector length:
_HSV_BINS   = (16, 16, 16)   # 4 096
_GRAD_BINS  = (16, 16)       # 256
_HIST_SIZE  = 4096 + 256     # 4 352


class SalineDetector(BaseDetector):
    """
    Saline monitor using Calibration-based Bhattacharyya Histogram Distance Comparison.

    Detection pipeline:
      1. User draws ROI representing the full bottle.
      2. Calibration stores the HSV + gradient histogram of this full bottle in JSON.
      3. For each frame, we compute the same combined histogram of the same ROI.
      4. Compare current histogram to the reference using Bhattacharyya distance.
      5. Map distance to a fluid level estimate:
            distance = 0.0  →  100 % (perfectly matches the full-bottle reference)
            distance = empty_threshold  →  0 %

    FIX NOTES (v2):
      • Do NOT apply a second NORM_MINMAX after vstack — it destroys the relative
        weighting between HSV and gradient parts.  Each sub-histogram is normalised
        independently before concatenation.
      • Level mapping uses a power-curve so that early, slow drift (top of bag)
        still reads as "high" and the last stretch maps naturally to LOW/EMPTY.
      • Smoothing uses an EMA with a configurable alpha instead of a raw median
        on a cold deque that distorts the first ~30 frames.
      • `set_manual_roi` no longer overwrites calibration histogram on disk.
    """

    def __init__(self, config=None):
        super().__init__("saline_detector")
        cfg = config or {}

        # ── Thresholds ──────────────────────────────────────────────────────────
        self.low_threshold = self._normalize_distance_threshold(
            cfg.get("low_threshold", 0.30), default_value=0.30, name="low_threshold"
        )
        self.empty_threshold = self._normalize_distance_threshold(
            cfg.get("empty_threshold", 0.55), default_value=0.55, name="empty_threshold"
        )
        self.no_bottle_threshold = self._normalize_distance_threshold(
            cfg.get("no_bottle_threshold", 0.80),
            default_value=0.80,
            name="no_bottle_threshold",
        )
        self.confirm_frames = int(cfg.get("confirm_frames", 4))
        self.calibration_frames = int(cfg.get("calibration_frames", 7))
        self.auto_tune_enabled = bool(cfg.get("auto_tune_enabled", True))
        self.auto_tune_frames = int(cfg.get("auto_tune_frames", 60))
        if self.empty_threshold < self.low_threshold:
            log.warning(
                "Saline thresholds out of order (low=%.3f, empty=%.3f). Adjusting.",
                self.low_threshold,
                self.empty_threshold,
            )
            self.empty_threshold = self.low_threshold + 0.05
        if self.no_bottle_threshold < self.empty_threshold:
            log.warning(
                "Saline no_bottle_threshold too low (%.3f). Adjusting above empty_threshold.",
                self.no_bottle_threshold,
            )
            self.no_bottle_threshold = min(0.98, self.empty_threshold + 0.10)

        # ── ROI & Calibration ───────────────────────────────────────────────────
        self.roi_padding      = cfg.get("roi_padding", 10)
        self.calibration_file = cfg.get(
            "calibration_file", "data/calibration/saline_calibration.json"
        )

        # ── Smoothing ───────────────────────────────────────────────────────────
        # EMA alpha: higher = reacts faster, lower = more stable.
        # history_size kept for backward compat but EMA is primary now.
        self.ema_alpha    = cfg.get("ema_alpha", 0.15)
        self.history_size = cfg.get("history_size", 30)

        # ── Internal State ──────────────────────────────────────────────────────
        self._manual_roi      = None
        self._reference_hist  = None   # shape (4352,) float32
        self._roi_set         = False

        self._level        = 100.0
        self._distance_ema = 0.0
        self._ema_ready    = False          # False until first real frame
        self._distance_history = deque(maxlen=self.history_size)
        self._status       = "NOT SET"

        self._tracking_id    = None
        self._camera_shifted = False
        self._draw_mode      = False
        self._pending_status = None
        self._pending_count = 0
        self._stable_status = "NOT SET"
        self._tune_distances = deque(maxlen=max(10, self.auto_tune_frames))
        self._recent_hists = deque(maxlen=max(5, self.calibration_frames * 2))

        self._load_calibration()

    @staticmethod
    def _normalize_distance_threshold(value, *, default_value: float, name: str) -> float:
        """
        Accept either ratio thresholds (0..1) or legacy percent values (0..100).
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            log.warning("Invalid saline %s=%r, using default %.3f", name, value, default_value)
            return default_value
        if v <= 0:
            log.warning("Non-positive saline %s=%.3f, using default %.3f", name, v, default_value)
            return default_value
        if v > 1.0:
            v = v / 100.0
        return max(0.01, min(0.99, v))

    # ── Calibration I/O ────────────────────────────────────────────────────────

    def _load_calibration(self):
        if not os.path.exists(self.calibration_file):
            return
        try:
            with open(self.calibration_file) as f:
                data = json.load(f)

            saved_roi = tuple(data["roi"]) if data.get("roi") else None
            if saved_roi and len(saved_roi) == 4:
                self._manual_roi = saved_roi
                self._roi_set    = True

            hist_data = data.get("hist")
            if hist_data:
                arr = np.array(hist_data, dtype=np.float32).flatten()
                if arr.size == _HIST_SIZE:
                    self._reference_hist = arr
                else:
                    log.warning(
                        "Discarding saline calibration: expected %d values, got %d.",
                        _HIST_SIZE, arr.size,
                    )

            log.info(
                "Saline calibration loaded: ROI=%s, hist=%s",
                self._manual_roi,
                "yes" if self._reference_hist is not None else "no",
            )
        except Exception as e:
            log.warning("Could not load saline calibration: %s", e)

    def _save_calibration(self, *, include_hist: bool = True):
        """
        Persist ROI (and optionally the reference histogram) to disk.

        Parameters
        ----------
        include_hist : bool
            Set to False when saving only an ROI update so that a previously
            valid calibration histogram is not erased.
        """
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)

        # Read whatever is currently on disk so we don't wipe unrelated keys.
        existing = {}
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file) as f:
                    existing = json.load(f)
            except Exception:
                pass

        if self._manual_roi:
            existing["roi"] = list(self._manual_roi)

        if include_hist and self._reference_hist is not None:
            existing["hist"] = self._reference_hist.flatten().tolist()

        with open(self.calibration_file, "w") as f:
            json.dump(existing, f)

    # ── Histogram Computation ──────────────────────────────────────────────────

    @staticmethod
    def _compute_combined_hist(crop: np.ndarray) -> np.ndarray:
        """
        Build a combined HSV + gradient histogram for `crop`.

        Returns a flat float32 vector of length _HIST_SIZE (4352).

        Design decisions:
          • Each sub-histogram is normalised to [0, 1] independently before
            concatenation.  A second global normalisation is NOT applied because
            it would destroy the relative scale between colour and structure cues.
          • Gradient magnitude is clipped to [0, 255] in absolute units (not
            MINMAX per-frame) so that empty, near-transparent ROIs with only
            background noise don't get artificially inflated structure histograms.
        """
        # ── HSV colour histogram ────────────────────────────────────────────
        hsv     = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_c  = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            list(_HSV_BINS),
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist_c, hist_c, 0, 1, cv2.NORM_MINMAX)

        # ── Gradient orientation + magnitude histogram ──────────────────────
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Absolute clip — do NOT use MINMAX here (inflates noise in empty bottles).
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        ang = np.clip(ang / 2, 0, 179).astype(np.uint8)   # map [0°,360°] → [0,179]

        merged  = cv2.merge([ang, mag])
        hist_g  = cv2.calcHist(
            [merged], [0, 1], None,
            list(_GRAD_BINS),
            [0, 180, 0, 256],
        )
        cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)

        # ── Concatenate ─────────────────────────────────────────────────────
        combined = np.concatenate(
            [hist_c.flatten(), hist_g.flatten()]
        ).astype(np.float32)

        # Sanity check (should never fire in production).
        assert combined.size == _HIST_SIZE, (
            f"Combined histogram size mismatch: {combined.size} != {_HIST_SIZE}"
        )
        return combined

    # ── ROI Helper ─────────────────────────────────────────────────────────────

    def _inner_roi(self, frame: np.ndarray, roi: tuple):
        """Return (crop, adjusted_roi) with padding applied."""
        x1, y1, x2, y2 = [int(v) for v in roi]
        x1 = max(0, x1 + self.roi_padding)
        y1 = max(0, y1 + self.roi_padding)
        x2 = min(frame.shape[1], x2 - self.roi_padding)
        y2 = min(frame.shape[0], y2 - self.roi_padding)
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        return np.array([]), (x1, y1, x2, y2)

    # ── Level Estimation ───────────────────────────────────────────────────────

    def _distance_to_level(self, dist: float) -> float:
        """
        Map Bhattacharyya distance [0, empty_threshold] → level [100, 0].

        A power curve (gamma < 1) keeps the reading high during the early,
        slow drift phase (bag is full) and compresses the LOW/EMPTY zone so
        that the last visible portion of liquid maps naturally below 25 %.

              level = (1 - dist / empty_threshold) ^ gamma * 100

        gamma = 0.6 was chosen empirically; tune via config if needed.
        """
        gamma = 0.6
        if dist >= self.empty_threshold:
            return 0.0
        fraction = 1.0 - (dist / self.empty_threshold)
        return float(min(100.0, max(0.0, (fraction ** gamma) * 100.0)))

    def _commit_status(self, proposed: str) -> str:
        """
        Commit status only after consecutive-frame confirmation.
        """
        if self._stable_status in ("NOT SET", "UNCALIBRATED", "ERROR"):
            self._stable_status = proposed
            self._pending_status = proposed
            self._pending_count = 1
            return self._stable_status

        if proposed == self._stable_status:
            self._pending_status = proposed
            self._pending_count = 0
            return self._stable_status

        if proposed == self._pending_status:
            self._pending_count += 1
        else:
            self._pending_status = proposed
            self._pending_count = 1

        if self._pending_count >= max(1, self.confirm_frames):
            self._stable_status = proposed
            self._pending_count = 0
        return self._stable_status

    def _auto_tune_thresholds(self, raw_distance: float):
        """
        Learn baseline distance from early stable frames and auto-tune thresholds.
        """
        if not self.auto_tune_enabled:
            return
        if len(self._tune_distances) >= self.auto_tune_frames:
            return
        self._tune_distances.append(float(raw_distance))
        if len(self._tune_distances) < max(20, self.auto_tune_frames // 2):
            return

        vals = np.array(self._tune_distances, dtype=np.float32)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        tuned_low = np.clip(mean + 2.0 * std, 0.12, 0.65)
        tuned_empty = np.clip(mean + 3.6 * std, tuned_low + 0.06, 0.90)
        tuned_no_bottle = np.clip(mean + 5.2 * std, tuned_empty + 0.08, 0.98)

        self.low_threshold = float(tuned_low)
        self.empty_threshold = float(tuned_empty)
        self.no_bottle_threshold = float(tuned_no_bottle)

    # ── detect() ───────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, context: dict) -> DetectionResult:
        if not self._roi_set or self._manual_roi is None:
            return DetectionResult(
                status="NOT SET",
                confidence=0.0,
                detail="Click 'Set Saline' to draw IV bottle ROI",
                metadata={"level": -1, "using_yolo": False, "roi_set": False},
            )

        if self._reference_hist is None:
            return DetectionResult(
                status="UNCALIBRATED",
                confidence=0.0,
                detail="ROI set but no calibration histogram found. Please recalibrate.",
                metadata={
                    "level": -1, "using_yolo": False,
                    "roi_set": True, "bbox": self._manual_roi,
                },
            )

        # ── Step 1: Extract current histogram ─────────────────────────────
        crop, valid_roi = self._inner_roi(frame, self._manual_roi)
        if crop.size == 0:
            return DetectionResult(
                status="ERROR",
                confidence=0.0,
                detail="Invalid ROI boundaries",
                metadata={
                    "level": -1, "using_yolo": False,
                    "roi_set": True, "bbox": self._manual_roi,
                },
            )

        hist_flat = self._compute_combined_hist(crop)
        self._recent_hists.append(hist_flat.copy())

        # ── Step 2: Compare to reference ──────────────────────────────────
        raw_distance = float(
            cv2.compareHist(self._reference_hist, hist_flat, cv2.HISTCMP_BHATTACHARYYA)
        )
        grad_ref = self._reference_hist[4096:]
        grad_cur = hist_flat[4096:]
        gradient_distance = float(
            cv2.compareHist(grad_ref, grad_cur, cv2.HISTCMP_BHATTACHARYYA)
        )

        # EMA smoothing — warm-start on first frame so we don't read 0.0.
        if not self._ema_ready:
            self._distance_ema = raw_distance
            self._ema_ready    = True
        else:
            self._distance_ema = (
                self.ema_alpha * raw_distance
                + (1.0 - self.ema_alpha) * self._distance_ema
            )

        # Keep history for debug / UI sparkline.
        self._distance_history.append(raw_distance)

        # ── Step 3: Map distance → level ──────────────────────────────────
        self._level = self._distance_to_level(self._distance_ema)

        # ── Step 4: Determine status ───────────────────────────────────────
        if gradient_distance >= self.no_bottle_threshold:
            # Missing bottle is an operator/setup state, not a saline depletion event.
            proposed_status = "NO BOTTLE"
            level_value = -1.0
        elif self._distance_ema >= self.empty_threshold:
            proposed_status = "EMPTY"
            level_value = self._level
        elif self._distance_ema >= self.low_threshold:
            proposed_status = "LOW"
            level_value = self._level
        else:
            proposed_status = "OK"
            level_value = self._level

        self._auto_tune_thresholds(raw_distance)
        status = self._commit_status(proposed_status)
        if status == "EMPTY":
            severity, should_alert = "CRITICAL", True
        elif status == "LOW":
            severity, should_alert = "HIGH", True
        elif status == "NO BOTTLE":
            severity, should_alert = "MEDIUM", False
        else:
            severity, should_alert = "LOW", False

        self._status = status

        return DetectionResult(
            status=status,
            confidence=1.0,
            detail=(
                f"Dist: {self._distance_ema:.3f} | GradDist: {gradient_distance:.3f} | "
                f"Level: {level_value:.1f}%"
            ),
            severity=severity,
            should_alert=should_alert,
            metadata={
                "level":        level_value,
                "using_yolo":   False,
                "bbox":         self._manual_roi,
                "roi_set":      True,
                "distance":     self._distance_ema,
                "distance_raw": raw_distance,
                "gradient_distance": gradient_distance,
                "low_threshold": self.low_threshold,
                "empty_threshold": self.empty_threshold,
                "no_bottle_threshold": self.no_bottle_threshold,
                "confirm_frames": self.confirm_frames,
                "status_pending": self._pending_status,
                "status_pending_count": self._pending_count,
            },
        )

    # Alias so SalinePipeline can call either name.
    def process(self, frame: np.ndarray, context: dict) -> DetectionResult:
        return self.detect(frame, context)

    # ── annotate() ─────────────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        from utils.drawing import draw_saline_indicator
        meta  = result.metadata or {}
        bbox  = meta.get("bbox")
        level = meta.get("level", -1)
        if bbox and level >= 0:
            draw_saline_indicator(frame, bbox, level, result.status)
        return frame

    # ── ROI Management ─────────────────────────────────────────────────────────

    def set_manual_roi(self, roi: tuple):
        x1, y1, x2, y2 = [int(v) for v in roi]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            log.warning("Saline ROI too small (%dx%d), ignored", x2 - x1, y2 - y1)
            return
        self._manual_roi = (x1, y1, x2, y2)
        self._roi_set    = True
        self._reset_smoothing()

        # Persist ROI only — preserve any valid histogram already on disk.
        self._save_calibration(include_hist=False)
        log.info("Manual saline ROI set: %s", self._manual_roi)

    def clear_saline_roi(self):
        self._manual_roi     = None
        self._roi_set        = False
        self._reference_hist = None
        self._reset_smoothing()
        self._level = 100.0

        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, "w") as f:
            json.dump({}, f)
        log.info("Saline ROI and calibration cleared.")

    def enter_draw_mode(self):
        self._draw_mode = True

    def confirm_draw_roi(self, x1: int, y1: int, x2: int, y2: int):
        self._draw_mode = False
        self.set_manual_roi((x1, y1, x2, y2))
        log.info("Saline ROI confirmed from draw: (%d,%d)-(%d,%d)", x1, y1, x2, y2)

    def cancel_draw_mode(self):
        self._draw_mode = False

    # ── Calibration ────────────────────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Lock the current ROI crop as the 100 % reference histogram.
        Must be called when the IV bag is visually full.
        """
        if self._manual_roi is None:
            log.warning("No ROI set — cannot calibrate.")
            return False

        crop, _ = self._inner_roi(frame, self._manual_roi)
        if crop.size == 0:
            log.warning("Crop is empty — cannot calibrate.")
            return False

        current_hist = self._compute_combined_hist(crop)
        hists = [current_hist]
        if self._recent_hists:
            n_take = max(0, self.calibration_frames - 1)
            hists.extend(list(self._recent_hists)[-n_take:])
        self._reference_hist = np.mean(np.stack(hists, axis=0), axis=0).astype(np.float32)
        self._reset_smoothing()
        self._level = 100.0
        self._stable_status = "OK"
        self._pending_status = None
        self._pending_count = 0
        self._tune_distances.clear()

        self._save_calibration(include_hist=True)
        log.info(
            "Saline reference histogram locked. Hist size=%d, ROI=%s",
            self._reference_hist.size, self._manual_roi,
        )
        return True

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _reset_smoothing(self):
        self._distance_history.clear()
        self._distance_ema = 0.0
        self._ema_ready    = False

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def level(self) -> float:
        return self._level

    @property
    def status(self) -> str:
        return self._status

    @property
    def method_levels(self) -> dict:
        return {}

    @property
    def tracking_id(self):
        return self._tracking_id

    @property
    def camera_shifted(self) -> bool:
        return self._camera_shifted

    @property
    def roi_set(self) -> bool:
        return self._roi_set