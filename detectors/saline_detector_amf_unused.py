"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SALINE — Adaptive Multi-Method Fusion (AMF) v3                        ║
║                                                                          ║
║  Production-ready IV saline monitoring with 4-method fluid detection:   ║
║  1. Canny Edge Detection — horizontal edge at fluid boundary            ║
║  2. Hough Line Transform — dominant horizontal line in ROI              ║
║  3. Intensity Gradient — liquid→air transition via column derivative    ║
║  4. GDS (Gradient Direction Scan) — Sobel-Y/total ratio for meniscus   ║
║                                                                          ║
║  Robustness features:                                                    ║
║  • CLAHE adaptive histogram equalization for lighting invariance        ║
║  • Median filtering for noise reduction                                  ║
║  • ORB feature matching for camera-shift self-recalibration             ║
║  • Physics-constrained monotonic drain model                             ║
║  • Temporal smoothing (moving average) for stable readings               ║
║  • Weighted fusion with IQR-based outlier rejection                     ║
║  • Fallback chain: YOLO → ORB-shifted → EMA → Manual → Last known     ║
║  • Transparent water bottle detection support                            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import json
import os
import time
import numpy as np
import logging
from collections import deque
from detectors.base_detector import BaseDetector, DetectionResult
from utils.geometry import iou_xyxy, bbox_area_xyxy, centroid_from_xyxy

log = logging.getLogger(__name__)

# Physics constraints — saline only drains
PHYSICS_MAX_RISE_PCT = 8.0
PHYSICS_MAX_DROP_PCT = 25.0


class SalineDetector(BaseDetector):
    """
    Adaptive Multi-Method Fusion (AMF) v3 saline monitor.

    Detection pipeline:
      1. YOLO bottle detection (COCO class 39 + transparent bottles)
      2. Shape filtering (aspect ratio, area, contour)
      3. ROI extraction with adaptive padding
      4. CLAHE + median filter preprocessing
      5. 4-method fluid level estimation (Canny, Hough, Gradient, GDS)
      6. Weighted fusion with outlier rejection
      7. Physics-constrained temporal smoothing
      8. ORB-based camera shift recalibration
      9. Alert generation with confirmation streaks

    Public API (preserved from v2):
      - detect(frame, context) → DetectionResult
      - annotate(frame, result) → frame
      - set_manual_roi(roi)
      - calibrate(frame) → bool
      - level (property)
      - status (property)
    """

    def __init__(self, config=None):
        super().__init__("saline_detector")

        cfg = config or {}

        # ── Thresholds ──
        self.low_threshold      = cfg.get("low_threshold", 30.0)
        self.critical_threshold = cfg.get("critical_threshold", 20.0)
        self.empty_threshold    = cfg.get("empty_threshold", 10.0)

        # ── Detection ──
        self.iv_model_path       = cfg.get("iv_model_path", "")
        self.bottle_class_id     = cfg.get("bottle_class_id", 39)
        self.detect_transparent  = cfg.get("detect_transparent", True)

        # ── Edge detection ──
        self.canny_low          = cfg.get("canny_low", 50)
        self.canny_high         = cfg.get("canny_high", 150)
        self.use_adaptive_canny = cfg.get("use_adaptive_canny", True)

        # ── ROI ──
        self.roi_padding         = cfg.get("roi_padding", 10)
        self.meniscus_center_w   = cfg.get("meniscus_center_width_frac", 0.55)

        # ── Smoothing ──
        self.history_size       = cfg.get("history_size", 30)
        self.bbox_ema_alpha     = cfg.get("bbox_ema_alpha", 0.4)

        # ── Calibration ──
        self.calibration_file   = cfg.get("calibration_file",
                                          "data/calibration/saline_calibration.json")
        self.fallback_frames    = cfg.get("yolo_fallback_frames", 30)

        # ── Bottle plausibility ──
        self.min_bottle_area_px  = cfg.get("min_bottle_area_px", 900)
        self.bottle_ar_min       = cfg.get("bottle_aspect_ratio_min", 1.15)
        self.bottle_ar_max       = cfg.get("bottle_aspect_ratio_max", 5.0)
        self.max_center_jump_px  = cfg.get("max_bbox_center_jump_px", 100.0)
        self.min_bbox_iou_ema    = cfg.get("min_bbox_iou_with_ema", 0.08)

        # ── Fusion & integrity ──
        self.methods_agreement_max_diff = cfg.get("methods_agreement_max_diff", 28.0)
        self.integrity_cap       = cfg.get("low_integrity_confidence_cap", 0.45)
        self.alert_confirm_frames = max(1, int(cfg.get("alert_confirm_frames", 5)))

        # ── Preprocessing (CLAHE + median) ──
        self.use_clahe          = cfg.get("use_clahe", True)
        self.clahe_clip         = cfg.get("clahe_clip_limit", 3.0)
        self.clahe_tile         = cfg.get("clahe_tile_size", 8)
        self.use_median         = cfg.get("use_median_filter", True)
        self.median_k           = cfg.get("median_kernel_size", 5)

        # ── ORB self-recalibration ──
        self.orb_features        = cfg.get("orb_features", 500)
        self.orb_match_thresh    = cfg.get("orb_match_threshold", 30.0)
        self.cam_shift_thresh    = cfg.get("camera_shift_threshold_px", 15.0)

        # ── Optical flow ──
        self.use_optical_flow_lock = cfg.get("use_optical_flow_lock", True)
        self.flow_max_shift_px     = float(cfg.get("flow_max_shift_px", 18.0))

        # ── Hough ──
        self.hough_meniscus      = cfg.get("hough_meniscus", True)
        self.hough_min_len_frac  = float(cfg.get("hough_min_line_length_frac", 0.35))

        # ── 4-method fusion weights ──
        self.w_canny    = cfg.get("canny_weight", 0.25)
        self.w_hough    = cfg.get("hough_weight", 0.25)
        self.w_gradient = cfg.get("gradient_weight", 0.25)
        self.w_gds      = cfg.get("gds_weight", 0.25)

        # ── Temporal constraints ──
        self.temporal_jump_pct    = float(cfg.get("temporal_jump_level_pct", 22.0))
        self.temporal_jump_window = max(3, int(cfg.get("temporal_jump_window", 5)))
        self.stale_std_max        = float(cfg.get("stale_level_std_max", 0.35))
        self.stale_min_frames     = int(cfg.get("stale_level_min_frames", 90))

        # ── Alert timing ──
        self.no_bottle_timeout   = cfg.get("no_bottle_timeout_sec", 10.0)

        # ── GDS parameters ──
        self.gds_ratio_threshold  = 0.55
        self.gds_min_energy       = 8.0
        self.gds_smoothing_k      = 9
        self.gds_scan_margin_frac = 0.07

        # ══════════════════════════════════════════════════════════════
        #  Internal State
        # ══════════════════════════════════════════════════════════════
        self._yolo_roi       = None
        self._manual_roi     = None
        self._ema_bbox       = None
        self._last_bbox      = None
        self._level_history  = deque(maxlen=self.history_size)
        self._instant_levels = deque(maxlen=max(64, self.stale_min_frames))
        self._level          = 100.0
        self._physics_level  = 100.0
        self._status         = "NOT SET"
        self._yolo_miss_count = 0
        self._using_yolo     = False
        self._reference_hist = None
        self._low_streak     = 0
        self._empty_streak   = 0
        self._critical_streak = 0
        self._integrity_ok   = True
        self._last_raw_level = 100.0
        self._tracking_id    = None
        self._last_bottle_time = time.time()

        # Whether user has explicitly set a ROI (manual or calibrated)
        self._roi_set = False

        # Whether we have a fine-tuned IV model (not COCO generic)
        self._has_iv_model = bool(self.iv_model_path)

        # ORB recalibration state
        self._orb = cv2.ORB_create(nfeatures=self.orb_features)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_orb_kp = None
        self._prev_orb_desc = None
        self._prev_orb_roi = None
        self._camera_shifted = False

        # CLAHE instance
        if self.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_tile, self.clahe_tile))
        else:
            self._clahe = None

        # Optical flow state
        self._prev_frame_gray = None

        # Per-method last values for dashboard
        self._method_levels = {
            "canny": None, "hough": None,
            "gradient": None, "gds": None
        }

        self._load_calibration()

    # ══════════════════════════════════════════════════════════════════
    #  Calibration Persistence
    # ══════════════════════════════════════════════════════════════════
    def _load_calibration(self):
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file) as f:
                    data = json.load(f)
                saved_roi = tuple(data["roi"]) if data.get("roi") else None
                if saved_roi and len(saved_roi) == 4:
                    self._manual_roi = saved_roi
                    self._ema_bbox = saved_roi
                    self._last_bbox = saved_roi
                    self._roi_set = True   # User had set this previously
                hist_data = data.get("hist")
                if hist_data:
                    arr = np.array(hist_data, dtype=np.float32)
                    self._reference_hist = arr.flatten()
                log.info("Saline calibration loaded: ROI=%s, roi_set=%s",
                         self._manual_roi, self._roi_set)
            except Exception as e:
                log.warning("Could not load saline calibration: %s", e)

    def _save_calibration(self):
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        data = {}
        roi = self._yolo_roi or self._manual_roi
        if roi:
            data["roi"] = list(roi)
        if self._reference_hist is not None:
            data["hist"] = self._reference_hist.flatten().tolist()
        with open(self.calibration_file, "w") as f:
            json.dump(data, f)

    # ══════════════════════════════════════════════════════════════════
    #  Preprocessing Pipeline (CLAHE + Median)
    # ══════════════════════════════════════════════════════════════════
    def _preprocess_roi(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply robustness preprocessing to grayscale ROI crop.
        1. CLAHE — adaptive histogram equalization for lighting invariance
        2. Median filter — salt-and-pepper noise removal
        Returns preprocessed grayscale image.
        """
        processed = gray.copy()

        # Step 1: CLAHE for adaptive contrast
        if self._clahe is not None:
            processed = self._clahe.apply(processed)

        # Step 2: Median filter for noise
        if self.use_median:
            k = self.median_k
            if k % 2 == 0:
                k += 1
            processed = cv2.medianBlur(processed, k)

        return processed

    # ══════════════════════════════════════════════════════════════════
    #  Bottle Detection & Selection
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _bbox_aspect_ratio(box):
        x1, y1, x2, y2 = box
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        return h / w

    def _plausible_bottle(self, box) -> bool:
        """Check if bounding box matches IV bottle shape constraints."""
        if bbox_area_xyxy(box) < self.min_bottle_area_px:
            return False
        ar = self._bbox_aspect_ratio(box)
        return self.bottle_ar_min <= ar <= self.bottle_ar_max

    def _score_bottle_candidate(self, box, conf: float) -> float:
        """Score a bottle candidate by confidence, shape, and proximity to EMA."""
        if not self._plausible_bottle(box):
            return -1.0
        ar = self._bbox_aspect_ratio(box)
        ar_mid = (self.bottle_ar_min + self.bottle_ar_max) / 2.0
        ar_pen = abs(ar - ar_mid) / max(ar_mid, 0.01)
        score = conf * 2.0 - 0.15 * ar_pen
        if self._ema_bbox is not None:
            score += 0.5 * iou_xyxy(tuple(box), tuple(self._ema_bbox))
            cx, cy = centroid_from_xyxy(*box)
            ecx, ecy = centroid_from_xyxy(*self._ema_bbox)
            dist = ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5
            if dist > self.max_center_jump_px:
                score -= 1.0
        return score

    def _pick_bottle(self, bottles: list):
        """Select best bottle from candidates using scoring function."""
        if not bottles:
            return None
        best, best_score = None, -1e9
        for b in bottles:
            box = tuple(b["bbox"])
            s = self._score_bottle_candidate(box, float(b.get("confidence", 0)))
            if s > best_score:
                best_score = s
                best = b
        if best_score < 0:
            return max(bottles, key=lambda x: float(x.get("confidence", 0)))
        return best

    def _smooth_bbox(self, raw_box: tuple) -> tuple:
        """EMA smooth bounding box to reduce jitter."""
        if self._ema_bbox is None:
            self._ema_bbox = raw_box
            return raw_box
        a = self.bbox_ema_alpha
        out = tuple((1 - a) * self._ema_bbox[i] + a * raw_box[i] for i in range(4))
        cx, cy = centroid_from_xyxy(*raw_box)
        ecx, ecy = centroid_from_xyxy(*self._ema_bbox)
        if ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5 > self.max_center_jump_px:
            iou = iou_xyxy(raw_box, tuple(self._ema_bbox))
            if iou < self.min_bbox_iou_ema:
                log.debug("Saline: rejecting abrupt bottle jump (IoU=%.2f)", iou)
                return tuple(self._ema_bbox)
        self._ema_bbox = out
        return out

    # ══════════════════════════════════════════════════════════════════
    #  ORB Feature-Matching Self-Recalibration (Novel)
    # ══════════════════════════════════════════════════════════════════
    def _orb_recalibrate(self, frame_bgr: np.ndarray, roi: tuple) -> tuple:
        """
        ORB-based camera shift detection and ROI correction.

        Algorithm:
          1. Extract ORB keypoints/descriptors in current ROI
          2. Match against previous frame's keypoints
          3. Compute median displacement of good matches
          4. If displacement > threshold → camera shifted → adjust ROI
          5. Store current keypoints for next frame

        Returns adjusted ROI tuple.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in roi]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return roi

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y1:y2, x1:x2]

        # Detect ORB keypoints in ROI
        kp, desc = self._orb.detectAndCompute(roi_gray, None)

        if kp is None or desc is None or len(kp) < 5:
            self._prev_orb_kp = kp
            self._prev_orb_desc = desc
            self._prev_orb_roi = (x1, y1, x2, y2)
            self._camera_shifted = False
            return roi

        # Match with previous frame
        if (self._prev_orb_desc is not None and
                self._prev_orb_kp is not None and
                len(self._prev_orb_desc) >= 5):
            try:
                matches = self._bf_matcher.match(self._prev_orb_desc, desc)
                matches = sorted(matches, key=lambda m: m.distance)
                # Keep only good matches below threshold
                good = [m for m in matches if m.distance < self.orb_match_thresh]

                if len(good) >= 4:
                    # Compute displacement vectors
                    prev_pts = np.array([self._prev_orb_kp[m.queryIdx].pt for m in good])
                    curr_pts = np.array([kp[m.trainIdx].pt for m in good])
                    displacements = curr_pts - prev_pts
                    dx = float(np.median(displacements[:, 0]))
                    dy = float(np.median(displacements[:, 1]))
                    shift_mag = (dx**2 + dy**2) ** 0.5

                    if shift_mag > self.cam_shift_thresh:
                        # Camera shifted — adjust ROI
                        self._camera_shifted = True
                        nx1 = int(np.clip(x1 + dx, 0, w - 5))
                        ny1 = int(np.clip(y1 + dy, 0, h - 5))
                        nx2 = int(np.clip(x2 + dx, nx1 + 4, w))
                        ny2 = int(np.clip(y2 + dy, ny1 + 4, h))
                        roi = (nx1, ny1, nx2, ny2)

                        # Update EMA bbox when not driven by YOLO
                        if not self._using_yolo and self._ema_bbox is not None:
                            ex1, ey1, ex2, ey2 = self._ema_bbox
                            self._ema_bbox = (
                                int(np.clip(ex1 + dx, 0, w - 5)),
                                int(np.clip(ey1 + dy, 0, h - 5)),
                                int(np.clip(ex2 + dx, 5, w)),
                                int(np.clip(ey2 + dy, 5, h)),
                            )
                        log.debug("ORB recalibration: shift=(%.1f, %.1f)px", dx, dy)
                    else:
                        self._camera_shifted = False
            except Exception:
                self._camera_shifted = False

        # Store for next frame
        self._prev_orb_kp = kp
        self._prev_orb_desc = desc
        self._prev_orb_roi = (x1, y1, x2, y2)
        return roi

    # ══════════════════════════════════════════════════════════════════
    #  Optical-Flow ROI Lock (Lucas-Kanade Fallback)
    # ══════════════════════════════════════════════════════════════════
    def _apply_flow_lock(self, frame_bgr: np.ndarray, roi: tuple) -> tuple:
        """LK optical flow for sub-pixel ROI tracking when ORB has few features."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if not self.use_optical_flow_lock:
            self._prev_frame_gray = gray.copy()
            return roi

        h, w = gray.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in roi]
        x1 = max(0, min(x1, w - 2)); x2 = max(x1 + 4, min(x2, w))
        y1 = max(0, min(y1, h - 2)); y2 = max(y1 + 4, min(y2, h))

        if self._prev_frame_gray is None or self._prev_frame_gray.shape != gray.shape:
            self._prev_frame_gray = gray.copy()
            return roi

        prev = self._prev_frame_gray
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=42, qualityLevel=0.2,
                                      minDistance=4, blockSize=5, mask=mask)
        if p0 is None or len(p0) < 5:
            self._prev_frame_gray = gray.copy()
            return roi

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev, gray, p0, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03))
        self._prev_frame_gray = gray.copy()

        if p1 is None:
            return roi
        good = st.ravel() == 1
        if good.sum() < 4:
            return roi

        d = np.asarray(p1[good], dtype=np.float32).reshape(-1, 2) - \
            np.asarray(p0[good], dtype=np.float32).reshape(-1, 2)
        dx = float(np.clip(np.median(d[:, 0]), -self.flow_max_shift_px, self.flow_max_shift_px))
        dy = float(np.clip(np.median(d[:, 1]), -self.flow_max_shift_px, self.flow_max_shift_px))

        nx1 = int(np.clip(x1 + dx, 0, w - 5))
        ny1 = int(np.clip(y1 + dy, 0, h - 5))
        nx2 = int(np.clip(x2 + dx, nx1 + 4, w))
        ny2 = int(np.clip(y2 + dy, ny1 + 4, h))

        if not self._using_yolo and self._ema_bbox is not None:
            ex1, ey1, ex2, ey2 = [int(round(v)) for v in self._ema_bbox]
            self._ema_bbox = (
                int(np.clip(ex1 + dx, 0, w - 5)),
                int(np.clip(ey1 + dy, 0, h - 5)),
                int(np.clip(ex2 + dx, 5, w)),
                int(np.clip(ey2 + dy, 5, h)),
            )

        return (nx1, ny1, nx2, ny2)

    # ══════════════════════════════════════════════════════════════════
    #  ROI Helpers
    # ══════════════════════════════════════════════════════════════════
    def _inner_roi(self, frame: np.ndarray, roi: tuple):
        """Crop inner ROI with padding from frame."""
        x1, y1, x2, y2 = [int(v) for v in roi]
        x1 = max(0, x1 + self.roi_padding)
        y1 = max(0, y1 + self.roi_padding)
        x2 = min(frame.shape[1], x2 - self.roi_padding)
        y2 = min(frame.shape[0], y2 - self.roi_padding)
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def _adaptive_canny(self, gray: np.ndarray):
        """Adaptive Canny edge detection using median-based thresholds."""
        if not self.use_adaptive_canny:
            return cv2.Canny(gray, self.canny_low, self.canny_high)
        med = np.median(gray)
        lo = int(max(10, 0.66 * med))
        hi = int(min(255, 1.33 * med))
        if hi <= lo + 5:
            hi = lo + 30
        return cv2.Canny(gray, lo, hi)

    # ══════════════════════════════════════════════════════════════════
    #  METHOD 1: Canny Edge Detection (Fluid Boundary)
    # ══════════════════════════════════════════════════════════════════
    def _estimate_canny(self, frame: np.ndarray, roi: tuple):
        """
        Detect fluid boundary using Canny edge detection.

        Algorithm:
          1. Crop ROI → grayscale → CLAHE + median preprocessing
          2. Gaussian blur → adaptive Canny edge detection
          3. Focus on central band to avoid bottle sidewalls
          4. Scan for strong horizontal edges (fluid boundary)
          5. Use row-wise edge density to find meniscus line
          6. Map position to fluid level percentage

        Returns float level [0,100] or None.
        """
        crop, _ = self._inner_roi(frame, roi)
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_roi(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Central band — avoid bottle sidewalls
        frac = 0.5 * self.meniscus_center_w
        cx0 = max(0, int(w * (0.5 - frac)))
        cx1 = min(w, int(w * (0.5 + frac)))
        if cx1 - cx0 < 4:
            cx0, cx1 = 0, w
        band = gray[:, cx0:cx1]

        edges = self._adaptive_canny(band)

        # Row-wise edge density — find the strongest horizontal edge row
        margin = max(2, int(h * 0.08))
        row_density = np.sum(edges > 0, axis=1).astype(np.float64)

        # Smooth the density curve
        k = max(3, min(9, h // 20 | 1))
        if k % 2 == 0:
            k += 1
        row_density = cv2.GaussianBlur(
            row_density.reshape(-1, 1), (1, k), 0).ravel()

        # Find peak edge density in safe zone (skip top/bottom margin)
        scan_zone = row_density[margin:h - margin]
        if len(scan_zone) < 4:
            return None

        threshold = np.mean(scan_zone) + 0.5 * np.std(scan_zone)
        # Scan top→bottom for first strong horizontal edge
        for r_rel in range(len(scan_zone)):
            if scan_zone[r_rel] > threshold:
                r = margin + r_rel
                level = 100.0 * (h - r) / float(h)
                return float(max(0.0, min(100.0, level)))

        # Fallback: use peak
        if scan_zone.max() > 0:
            peak_rel = int(np.argmax(scan_zone))
            r = margin + peak_rel
            level = 100.0 * (h - r) / float(h)
            return float(max(0.0, min(100.0, level)))

        return None

    # ══════════════════════════════════════════════════════════════════
    #  METHOD 2: Hough Line Transform (Horizontal Line Detection)
    # ══════════════════════════════════════════════════════════════════
    def _estimate_hough(self, frame: np.ndarray, roi: tuple):
        """
        Detect fluid surface using Hough Line Transform.

        Algorithm:
          1. Crop ROI → preprocess → Canny edge detection
          2. HoughLinesP to find line segments
          3. Filter by slope ≈ 0° (near-horizontal only)
          4. Filter by minimum length (fraction of ROI width)
          5. Take median y-position of qualifying lines
          6. Map to fluid level percentage

        Returns float level [0,100] or None.
        """
        if not self.hough_meniscus:
            return None
        crop, _ = self._inner_roi(frame, roi)
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_roi(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape
        if h < 20 or w < 12:
            return None

        edges = self._adaptive_canny(gray)
        min_len = max(int(w * self.hough_min_len_frac), 10)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=max(10, min_len // 2),
            minLineLength=min_len,
            maxLineGap=max(6, w // 20),
        )
        if lines is None:
            return None

        y_candidates = []
        margin = int(h * 0.08)
        for ln in lines:
            lx1, ly1, lx2, ly2 = ln[0]
            dy_abs = abs(ly2 - ly1)
            dx_abs = abs(lx2 - lx1)
            # Must be near-horizontal: slope ≈ 0°
            if dx_abs < min_len * 0.80:
                continue
            if dy_abs > max(4, h // 22):
                continue
            ym = 0.5 * (ly1 + ly2)
            if margin <= ym <= h - margin:
                y_candidates.append(ym)

        if not y_candidates:
            return None
        y_med = float(np.median(y_candidates))
        level = 100.0 * (h - y_med) / float(h)
        return float(max(0.0, min(100.0, level)))

    # ══════════════════════════════════════════════════════════════════
    #  METHOD 3: Intensity Gradient (Liquid→Air Transition)
    # ══════════════════════════════════════════════════════════════════
    def _estimate_gradient(self, frame: np.ndarray, roi: tuple):
        """
        Detect fluid boundary via intensity gradient analysis.

        Algorithm:
          1. Crop ROI → preprocess → central band extraction
          2. Compute column-wise mean intensity profile (top→bottom)
          3. Smooth profile with Gaussian
          4. Compute first derivative (gradient) of intensity profile
          5. Find maximum gradient magnitude → liquid/air boundary
          6. The transition from liquid (darker) to air (lighter) or
             vice versa creates a strong gradient spike
          7. Map position to fluid level percentage

        Returns float level [0,100] or None.
        """
        crop, _ = self._inner_roi(frame, roi)
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 24 or w < 10:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_roi(gray)

        # Central band
        frac = 0.5 * self.meniscus_center_w
        cx0 = max(0, int(w * (0.5 - frac)))
        cx1 = min(w, int(w * (0.5 + frac)))
        if cx1 - cx0 < 4:
            cx0, cx1 = 0, w
        band = gray[:, cx0:cx1]

        # Column-wise mean intensity for each row
        intensity_profile = np.mean(band, axis=1).astype(np.float64)

        # Smooth the profile
        k = max(3, min(11, h // 15 | 1))
        if k % 2 == 0:
            k += 1
        intensity_profile = cv2.GaussianBlur(
            intensity_profile.reshape(-1, 1), (1, k), 0).ravel()

        # Compute gradient (first derivative)
        gradient = np.abs(np.diff(intensity_profile))

        # Scan within margins
        margin = max(3, int(h * 0.08))
        if h - 2 * margin < 4:
            return None
        scan = gradient[margin:h - margin - 1]
        if len(scan) < 4:
            return None

        # Find strongest gradient transition
        peak_rel = int(np.argmax(scan))
        peak_val = scan[peak_rel]

        # Require minimum gradient strength
        if peak_val < np.mean(scan) + 0.3 * np.std(scan):
            return None

        r = margin + peak_rel
        level = 100.0 * (h - r) / float(h)
        return float(max(0.0, min(100.0, level)))

    # ══════════════════════════════════════════════════════════════════
    #  METHOD 4: GDS — Gradient Direction Scan (Primary, Novel)
    # ══════════════════════════════════════════════════════════════════
    def _estimate_gds(self, frame: np.ndarray, roi: tuple):
        """
        Gradient Direction Scan — primary meniscus detector.

        Physics basis: the air/liquid interface creates a STRONG HORIZONTAL
        gradient (Sobel-Y ≫ Sobel-X), while bottle edges, labels, and caps
        create predominantly vertical gradients.

        Algorithm:
          1. Crop ROI → central band
          2. Compute per-row: ratio = ΣSobel_Y / (ΣSobel_Y + ΣSobel_X + ε)
          3. Smooth ratio curve → scan top→bottom
          4. First row where ratio > threshold AND energy > min → meniscus
          5. Map to fluid level %

        Returns float level [0,100] or None.
        """
        crop, _ = self._inner_roi(frame, roi)
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 24 or w < 12:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_roi(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Central band
        frac = 0.5 * self.meniscus_center_w
        cx0 = max(0, int(w * (0.5 - frac)))
        cx1 = min(w, int(w * (0.5 + frac)))
        if cx1 - cx0 < 4:
            cx0, cx1 = 0, w
        band = gray[:, cx0:cx1]

        # Sobel in X and Y directions
        sx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)

        row_sx = np.sum(np.abs(sx), axis=1).astype(np.float64)
        row_sy = np.sum(np.abs(sy), axis=1).astype(np.float64)

        total_energy = row_sx + row_sy + 1e-6
        ratio = row_sy / total_energy  # 1.0 = purely horizontal edges

        # Smooth ratio curve
        k = self.gds_smoothing_k
        k = k if k % 2 == 1 else k + 1
        k = max(3, min(k, h // 3 | 1))
        if k > 1:
            ratio_s = cv2.GaussianBlur(
                ratio.reshape(-1, 1).astype(np.float32), (1, k), 0).ravel()
        else:
            ratio_s = ratio

        # Scan top→bottom within safe margins
        margin = max(2, int(h * self.gds_scan_margin_frac))
        for r in range(margin, h - margin):
            if (ratio_s[r] > self.gds_ratio_threshold and
                    row_sy[r] > self.gds_min_energy):
                level = 100.0 * (h - r) / float(h)
                return float(max(0.0, min(100.0, level)))

        # Fallback: largest ratio peak in scan zone
        scan_zone = ratio_s[margin:h - margin]
        if len(scan_zone) > 0 and scan_zone.max() > self.gds_ratio_threshold * 0.75:
            peak_rel = int(np.argmax(scan_zone))
            peak = margin + peak_rel
            if row_sy[peak] > self.gds_min_energy * 0.5:
                level = 100.0 * (h - peak) / float(h)
                return float(max(0.0, min(100.0, level)))

        return None

    # ══════════════════════════════════════════════════════════════════
    #  Weighted Fusion with IQR Outlier Rejection
    # ══════════════════════════════════════════════════════════════════
    def _fuse_methods(self, canny_lv, hough_lv, grad_lv, gds_lv):
        """
        Weighted fusion of 4 detection methods with outlier rejection.

        Strategy:
          1. Collect valid readings with their configured weights
          2. If ≥3 methods agree, use IQR to reject outliers
          3. Compute weighted average of remaining readings
          4. Track method agreement for integrity assessment

        Returns (fused_level, integrity_ok, spread)
        """
        readings = []
        weights = []
        if canny_lv is not None:
            readings.append(canny_lv)
            weights.append(self.w_canny)
        if hough_lv is not None:
            readings.append(hough_lv)
            weights.append(self.w_hough)
        if grad_lv is not None:
            readings.append(grad_lv)
            weights.append(self.w_gradient)
        if gds_lv is not None:
            readings.append(gds_lv)
            weights.append(self.w_gds)

        if not readings:
            return 50.0, False, 0.0

        readings = np.array(readings, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        spread = float(readings.max() - readings.min())

        # IQR outlier rejection when ≥3 methods available
        if len(readings) >= 3:
            q1 = np.percentile(readings, 25)
            q3 = np.percentile(readings, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (readings >= lower) & (readings <= upper)
            if mask.sum() >= 2:
                readings = readings[mask]
                weights = weights[mask]

        # Normalize weights
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum

        fused = float(np.sum(readings * weights))
        integrity = spread <= self.methods_agreement_max_diff
        return fused, integrity, spread

    # ══════════════════════════════════════════════════════════════════
    #  Physics-Constrained Level Update
    # ══════════════════════════════════════════════════════════════════
    def _apply_physics(self, instant_level: float) -> tuple:
        """
        Saline only drains — apply monotonic drain constraints.
        Max rise per frame = 8% (bag can't refill).
        Max drop per frame = 25% (can't drain instantly).
        Returns (constrained_level, physics_violated).
        """
        prev = self._physics_level
        delta = instant_level - prev
        violated = False

        if delta > PHYSICS_MAX_RISE_PCT:
            constrained = prev + PHYSICS_MAX_RISE_PCT
            violated = True
        elif delta < -PHYSICS_MAX_DROP_PCT:
            constrained = prev - PHYSICS_MAX_DROP_PCT
            violated = True
        else:
            constrained = instant_level

        constrained = float(max(0.0, min(100.0, constrained)))
        self._physics_level = constrained
        return constrained, violated

    # ══════════════════════════════════════════════════════════════════
    #  Temporal Plausibility Check
    # ══════════════════════════════════════════════════════════════════
    def _temporal_check(self, instant_level: float):
        """Check for implausible temporal jumps and stale readings."""
        self._instant_levels.append(instant_level)
        temporal_ok = True
        stale = False
        jump_reason = None

        hist = list(self._instant_levels)
        if len(hist) >= 2:
            step = hist[-1] - hist[-2]
            if step < -self.temporal_jump_pct:
                win = hist[-(self.temporal_jump_window + 1):-1]
                if len(win) >= self.temporal_jump_window - 1:
                    deltas = np.diff(win)
                    drift = float(np.mean(deltas)) if len(deltas) else 0.0
                    if drift > -1.2:
                        temporal_ok = False
                        jump_reason = "sharp_drop_no_drift"

        if len(hist) >= self.stale_min_frames and self._using_yolo:
            tail = hist[-self.stale_min_frames:]
            if float(np.std(tail)) < self.stale_std_max:
                stale = True

        return temporal_ok, stale, jump_reason

    # ══════════════════════════════════════════════════════════════════
    #  ROI Resolution (Robust Fallback Chain)
    # ══════════════════════════════════════════════════════════════════
    def _resolve_roi(self):
        """
        Priority: YOLO-tracked → EMA smoothed → manual/calibrated → last seen.
        Never returns None if any prior ROI exists.
        """
        if self._using_yolo and self._yolo_roi is not None:
            return self._yolo_roi
        if self._ema_bbox is not None:
            return self._ema_bbox
        if self._yolo_roi is not None:
            return self._yolo_roi
        if self._manual_roi is not None:
            return self._manual_roi
        if self._last_bbox is not None:
            return self._last_bbox
        return None

    # ══════════════════════════════════════════════════════════════════
    #  MAIN DETECT — Orchestrates Full Pipeline
    # ══════════════════════════════════════════════════════════════════
    def detect(self, frame: np.ndarray, context: dict) -> DetectionResult:
        """
        Full saline detection pipeline:
          1. Check if ROI is set (manual draw required when no fine-tuned model)
          2. If fine-tuned IV model: use YOLO bottle detections
          3. If no model: require user to draw ROI via 'Set Saline'
          4. Run 4-method fluid level estimation on the ROI
          5. Weighted fusion with outlier rejection
          6. Physics-constrained temporal smoothing
          7. Alert logic with confirmation streaks

        IMPORTANT: When using generic COCO 'bottle' class (no fine-tuned model),
        YOLO detections are IGNORED to prevent false positives. The user MUST
        manually set the ROI via the 'Set Saline' dashboard button.
        """
        yolo_conf = 0.0

        # ── Step 1: Handle YOLO detections ──
        # Only trust YOLO bottles if we have a FINE-TUNED IV model.
        # Generic COCO class 39 'bottle' produces too many false positives
        # (detects faces, hands, background objects as bottles).
        if self._has_iv_model:
            bottles = context.get("bottles", [])
            if bottles:
                chosen = self._pick_bottle(bottles)
                if chosen is not None:
                    raw_box = tuple(chosen["bbox"])
                    self._yolo_roi = self._smooth_bbox(raw_box)
                    self._last_bbox = self._yolo_roi
                    self._yolo_miss_count = 0
                    self._using_yolo = True
                    self._last_bottle_time = time.time()
                    self._roi_set = True
                    self._tracking_id = chosen.get("track_id",
                                                    self._tracking_id)
                    yolo_conf = float(chosen.get("confidence", 0))
                else:
                    self._yolo_miss_count += 1
                    self._using_yolo = False
            else:
                self._yolo_miss_count += 1
                self._using_yolo = False
        else:
            # No fine-tuned model → YOLO detections disabled
            self._using_yolo = False

        # ── Step 2: Check if ROI is set ──
        # If no ROI has been set (neither manual nor YOLO), return NOT SET
        if not self._roi_set:
            self._prev_frame_gray = None
            return DetectionResult(
                status="NOT SET",
                confidence=0.0,
                detail="Click 'Set Saline' to draw IV bottle ROI",
                metadata={"level": -1, "using_yolo": False,
                          "roi_set": False}
            )

        # ── Step 3: Resolve ROI ──
        roi_base = self._resolve_roi()
        if roi_base is None:
            self._prev_frame_gray = None
            return DetectionResult(
                status="NOT SET",
                confidence=0.0,
                detail="Click 'Set Saline' to draw IV bottle ROI",
                metadata={"level": -1, "using_yolo": False,
                          "roi_set": False}
            )

        self._last_bbox = roi_base

        # ── Step 3: ORB self-recalibration ──
        roi = self._orb_recalibrate(frame, roi_base)

        # ── Step 4: Optical flow lock (fallback) ──
        roi = self._apply_flow_lock(frame, roi)

        # ── Step 4.5: Histogram validation (Is the bottle still there?) ──
        if self._reference_hist is not None:
            c_roi, _ = self._inner_roi(frame, roi)
            if c_roi.size > 0:
                hsv = cv2.cvtColor(c_roi, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                hist_flat = hist.flatten().astype(np.float32)
                
                # Correlation: 1.0 = perfect match, < 0.35 = different object
                similarity = cv2.compareHist(self._reference_hist, hist_flat, cv2.HISTCMP_CORREL)
                
                if similarity < 0.35:
                    self._empty_streak = 0
                    self._critical_streak = 0
                    self._low_streak = 0
                    self._prev_frame_gray = None
                    return DetectionResult(
                        status="NO SALINE DETECTED",
                        confidence=0.0,
                        detail=f"Histogram mismatch (sim={similarity:.2f})",
                        severity="LOW",
                        should_alert=False,
                        metadata={
                            "level": -1,  # -1 causes `annotate` to hide the box
                            "using_yolo": False,
                            "roi_set": True,
                            "bbox": roi
                        }
                    )

        # ── Step 5: Run all 4 detection methods ──
        canny_lv = self._estimate_canny(frame, roi)
        hough_lv = self._estimate_hough(frame, roi)
        grad_lv  = self._estimate_gradient(frame, roi)
        gds_lv   = self._estimate_gds(frame, roi)

        # Store per-method values for dashboard
        self._method_levels = {
            "canny": canny_lv, "hough": hough_lv,
            "gradient": grad_lv, "gds": gds_lv
        }

        # ── Step 6: Weighted fusion ──
        fused_instant, integrity_ok, spread = self._fuse_methods(
            canny_lv, hough_lv, grad_lv, gds_lv)

        # ── Step 7: Physics constraint ──
        fused_constrained, physics_violated = self._apply_physics(fused_instant)
        if physics_violated:
            integrity_ok = False

        # ── Step 8: Temporal plausibility ──
        temporal_ok, stale, jump_reason = self._temporal_check(fused_constrained)
        if not temporal_ok:
            integrity_ok = False
        if stale:
            integrity_ok = False

        self._integrity_ok = integrity_ok
        self._last_raw_level = fused_constrained

        # Moving average smoothing
        self._level_history.append(fused_constrained)
        avg_level = float(np.median(self._level_history))
        self._level = avg_level

        # ── Step 9: Streak counters ──
        if avg_level <= self.empty_threshold:
            self._empty_streak += 1
            self._critical_streak += 1
            self._low_streak += 1
        elif avg_level <= self.critical_threshold:
            self._empty_streak = 0
            self._critical_streak += 1
            self._low_streak += 1
        elif avg_level <= self.low_threshold:
            self._empty_streak = 0
            self._critical_streak = 0
            self._low_streak += 1
        else:
            self._empty_streak = 0
            self._critical_streak = 0
            self._low_streak = 0

        # ── Step 10: Status logic ──
        status = "OK"
        severity = "LOW"
        should_alert = False
        confirm = self.alert_confirm_frames

        if stale and avg_level <= self.low_threshold:
            status = "UNCERTAIN"; severity = "MEDIUM"
        elif (self._empty_streak >= confirm and integrity_ok
              and avg_level <= self.empty_threshold):
            status = "EMPTY"; severity = "CRITICAL"; should_alert = True
        elif (self._critical_streak >= confirm and integrity_ok
              and avg_level <= self.critical_threshold):
            status = "CRITICAL LOW"; severity = "CRITICAL"; should_alert = True
        elif (self._low_streak >= confirm and integrity_ok
              and avg_level <= self.low_threshold):
            status = "LOW"; severity = "HIGH"; should_alert = True
        elif not integrity_ok and avg_level <= self.low_threshold:
            status = "UNCERTAIN"; severity = "MEDIUM"

        self._status = status

        # ── Step 11: Confidence score ──
        method_conf = 0.88 if self._using_yolo else 0.55
        n_methods = sum(1 for v in [canny_lv, hough_lv, grad_lv, gds_lv]
                        if v is not None)
        method_bonus = 0.05 * n_methods
        level_certainty = max(0.0, 1.0 - min(spread / 50.0, 1.0))
        orb_bonus = 0.04 if not self._camera_shifted else 0.0

        confidence = (
            0.28 * method_conf +
            0.22 * yolo_conf +
            0.20 * level_certainty +
            method_bonus + orb_bonus
        )
        if not integrity_ok:
            confidence = min(confidence, self.integrity_cap)
        if stale:
            confidence = min(confidence, self.integrity_cap * 0.9)
        confidence = float(max(0.0, min(1.0, confidence)))

        roi_used = tuple(self._last_bbox) if self._last_bbox else roi

        # ── Build detail string ──
        details = [f"AMF~{avg_level:.0f}% ({'YOLO' if self._using_yolo else 'ROI'})"]
        if canny_lv is not None:
            details.append(f"Canny={canny_lv:.0f}")
        if hough_lv is not None:
            details.append(f"Hough={hough_lv:.0f}")
        if grad_lv is not None:
            details.append(f"Grad={grad_lv:.0f}")
        if gds_lv is not None:
            details.append(f"GDS={gds_lv:.0f}")
        details.append(f"spread={spread:.0f}%")
        if self._camera_shifted:
            details.append("[ORB-shift]")
        if physics_violated:
            details.append("[physics!]")
        if jump_reason:
            details.append(f"[{jump_reason}]")
        if stale:
            details.append("stale?")

        return DetectionResult(
            status=status,
            confidence=confidence,
            detail=" ".join(details),
            severity=severity,
            should_alert=should_alert,
            metadata={
                "level":            avg_level,
                "using_yolo":       self._using_yolo,
                "yolo_miss_count":  self._yolo_miss_count,
                "bbox":             roi_used,
                "tracking_id":      self._tracking_id,
                "integrity_ok":     integrity_ok,
                "temporal_ok":      temporal_ok,
                "stale_reading":    stale,
                "jump_reason":      jump_reason,
                "physics_violated": physics_violated,
                "camera_shifted":   self._camera_shifted,
                "cue_spread":       spread,
                "n_methods":        n_methods,
                "canny_level":      canny_lv,
                "hough_level":      hough_lv,
                "gradient_level":   grad_lv,
                "gds_level":        gds_lv,
                "roi_set":          True,
            }
        )

    # ══════════════════════════════════════════════════════════════════
    #  Annotate
    # ══════════════════════════════════════════════════════════════════
    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw saline indicator overlay on frame."""
        from utils.drawing import draw_saline_indicator
        meta = result.metadata or {}
        bbox = meta.get("bbox")
        level = meta.get("level", -1)
        if bbox and level >= 0:
            draw_saline_indicator(frame, bbox, level, result.status,
                                  tracking_id=meta.get("tracking_id"),
                                  method_levels=self._method_levels,
                                  camera_shifted=meta.get("camera_shifted", False))
        return frame

    # ══════════════════════════════════════════════════════════════════
    #  Public Helpers
    # ══════════════════════════════════════════════════════════════════
    def set_manual_roi(self, roi: tuple):
        """Set ROI from user drawing (x1, y1, x2, y2 in frame pixel coords)."""
        x1, y1, x2, y2 = [int(v) for v in roi]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            log.warning("Saline ROI too small (%dx%d), ignored", x2-x1, y2-y1)
            return
        self._manual_roi = (x1, y1, x2, y2)
        self._ema_bbox = (x1, y1, x2, y2)
        self._last_bbox = (x1, y1, x2, y2)
        self._roi_set = True
        self._prev_frame_gray = None
        self._prev_orb_kp = None
        self._prev_orb_desc = None
        self._level_history.clear()
        self._instant_levels.clear()
        self._physics_level = 100.0
        self._low_streak = 0
        self._empty_streak = 0
        self._critical_streak = 0
        self._save_calibration()
        log.info("Manual saline ROI set: %s", (x1, y1, x2, y2))

    def clear_saline_roi(self):
        """Clear the saline ROI and reset to dormant state."""
        self._manual_roi = None
        self._ema_bbox = None
        self._last_bbox = None
        self._roi_set = False
        self._prev_frame_gray = None
        self._prev_orb_kp = None
        self._prev_orb_desc = None
        self._level_history.clear()
        self._instant_levels.clear()
        self._physics_level = 100.0
        self._low_streak = 0
        self._empty_streak = 0
        self._critical_streak = 0
        
        # Overwrite calibration file with empty dict to erase it
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, "w") as f:
            json.dump({}, f)
        log.info("Saline ROI cleared and deleted from calibration file.")

    def enter_draw_mode(self):
        """Signal that the user is about to draw a new saline ROI."""
        self._draw_mode = True
        log.info("Saline draw mode entered")

    def confirm_draw_roi(self, x1: int, y1: int, x2: int, y2: int):
        """Called by dashboard when user finishes drawing the ROI rectangle."""
        self._draw_mode = False
        self.set_manual_roi((x1, y1, x2, y2))
        log.info("Saline ROI confirmed from draw: (%d,%d)-(%d,%d)", x1, y1, x2, y2)

    def cancel_draw_mode(self):
        self._draw_mode = False

    def calibrate(self, frame: np.ndarray) -> bool:
        """Lock histogram of current bottle appearance as calibration reference."""
        roi = self._yolo_roi or self._manual_roi
        if roi is None:
            log.warning("No ROI available for calibration")
            return False
        crop, _ = self._inner_roi(frame, roi)
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        self._reference_hist = hist.flatten().astype(np.float32)
        self._level_history.clear()
        self._instant_levels.clear()
        self._level = 100.0
        self._physics_level = 100.0
        self._low_streak = 0
        self._empty_streak = 0
        self._critical_streak = 0
        self._save_calibration()
        log.info("Saline reference histogram locked from current bottle ROI")
        return True

    @property
    def level(self) -> float:
        return self._level

    @property
    def status(self) -> str:
        return self._status

    @property
    def method_levels(self) -> dict:
        """Per-method fluid level readings for dashboard display."""
        return self._method_levels.copy()

    @property
    def tracking_id(self):
        return self._tracking_id

    @property
    def camera_shifted(self) -> bool:
        return self._camera_shifted

    @property
    def roi_set(self) -> bool:
        """Whether the user has explicitly configured a saline ROI."""
        return self._roi_set
