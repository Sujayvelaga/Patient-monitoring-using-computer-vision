"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DRAWING UTILITIES — OpenCV overlay helpers for video annotation       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.geometry import SKELETON_CONNECTIONS

# ── Color Palette ──
COLORS = {
    "accent":     (255, 180,  60),   # Amber
    "green":      (  0, 210,  80),
    "red":        (  0,  50, 255),
    "orange":     (  0, 140, 255),
    "cyan":       (255, 220,   0),
    "white":      (240, 240, 240),
    "dim":        (110, 110, 130),
    "dark_bg":    ( 14,  14,  18),
    "panel_bg":   ( 20,  20,  26),
    "purple":     (200,  50, 200),
    "blue":       (255, 150,  50),
    "yellow":     (  0, 255, 255),
}


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                  confidence_threshold: float = 0.3,
                  color: Tuple[int, int, int] = (245, 117, 66),
                  thickness: int = 2) -> np.ndarray:
    """
    Draw pose skeleton on frame.

    Args:
        keypoints: (17, 3) array — [x, y, confidence] for each keypoint
    """
    if keypoints is None or len(keypoints) < 17:
        return frame

    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if (keypoints[i][2] > confidence_threshold and
                keypoints[j][2] > confidence_threshold):
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints
    for kp in keypoints:
        if kp[2] > confidence_threshold:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 3,
                       (245, 66, 230), -1)

    return frame


def draw_tracking_box(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                      track_id: int, is_locked: bool = False,
                      label: str = "", confidence: float = 0.0,
                      color: Optional[Tuple[int, int, int]] = None):
    """Draw tracked person bounding box with ID."""
    x1, y1, x2, y2 = [int(v) for v in bbox]

    if color is None:
        if is_locked:
            color = COLORS["cyan"]
        else:
            # Generate consistent color from track ID
            hue = (track_id * 67) % 180
            color_bgr = cv2.cvtColor(
                np.array([[[hue, 200, 200]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)

    thickness = 3 if is_locked else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # ID badge
    id_text = f"ID:{track_id}"
    if is_locked:
        id_text = f"★ PATIENT #{track_id}"
    if label:
        id_text += f" | {label}"
    if confidence > 0:
        id_text += f" {confidence:.0%}"

    (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    badge_color = (0, 0, 0) if is_locked else (30, 30, 30)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), badge_color, -1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, 1)
    cv2.putText(frame, id_text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_zone_polygon(frame: np.ndarray,
                      zone_pts: List[Tuple[int, int]],
                      is_inside: bool = True,
                      alpha: float = 0.15):
    """Draw bed zone polygon with translucent fill."""
    if not zone_pts or len(zone_pts) < 3:
        return frame

    pts = np.array(zone_pts, dtype=np.int32)
    color = (0, 180, 80) if is_inside else (0, 50, 255)
    fill_color = (0, 60, 20) if is_inside else (60, 10, 10)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], fill_color)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(frame, [pts], True, color, 2)

    return frame


def draw_confidence_bar(frame: np.ndarray, x: int, y: int,
                        width: int, height: int,
                        value: float, label: str = "",
                        high_color: Tuple[int, int, int] = (0, 210, 80),
                        low_color: Tuple[int, int, int] = (0, 50, 255)):
    """Draw a colored confidence/progress bar."""
    value = max(0.0, min(1.0, value))

    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (35, 35, 35), -1)

    # Filled portion — interpolate color
    filled_w = int(width * value)
    r = int(low_color[0] + (high_color[0] - low_color[0]) * value)
    g = int(low_color[1] + (high_color[1] - low_color[1]) * value)
    b = int(low_color[2] + (high_color[2] - low_color[2]) * value)
    cv2.rectangle(frame, (x, y), (x + filled_w, y + height), (r, g, b), -1)

    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)

    # Label
    if label:
        cv2.putText(frame, f"{label}: {value:.0%}",
                    (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    COLORS["white"], 1)


def draw_spine_angle_indicator(frame: np.ndarray,
                                mid_shoulder: Tuple[float, float],
                                mid_hip: Tuple[float, float],
                                angle: float,
                                is_fallen: bool = False):
    """Draw spine vector and angle indicator."""
    if mid_shoulder is None or mid_hip is None:
        return frame

    pt1 = (int(mid_hip[0]), int(mid_hip[1]))
    pt2 = (int(mid_shoulder[0]), int(mid_shoulder[1]))

    color = COLORS["red"] if is_fallen else COLORS["green"]
    cv2.line(frame, pt1, pt2, color, 3)

    # Angle text
    mid_x = (pt1[0] + pt2[0]) // 2
    mid_y = (pt1[1] + pt2[1]) // 2
    cv2.putText(frame, f"{angle:.0f}°",
                (mid_x + 10, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_alert_overlay(frame: np.ndarray, message: str,
                       severity: str = "CRITICAL", y_offset: int = 30):
    """Draw full-frame alert overlay."""
    h, w = frame.shape[:2]
    color = {
        "LOW": (0, 140, 255),
        "MEDIUM": (0, 140, 255),
        "HIGH": (0, 50, 255),
        "CRITICAL": (0, 0, 255)
    }.get(severity, (0, 50, 255))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    # Alert text
    cv2.putText(frame, f"⚠ {message}",
                (10, h - y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    return frame


def draw_saline_indicator(frame: np.ndarray,
                           bbox: Tuple[int, int, int, int],
                           level: float, status: str,
                           tracking_id=None,
                           method_levels: Optional[dict] = None,
                           camera_shifted: bool = False):
    """
    Draw enhanced saline bottle indicator with:
    - Gradient-filled level bar
    - Per-method breakdown mini-bars
    - Tracking ID badge
    - Camera shift indicator
    - Pulsing border for alerts
    """
    import time

    x1, y1, x2, y2 = [int(v) for v in bbox]

    if status == "EMPTY":
        color = COLORS["red"]
    elif status in ("LOW", "CRITICAL LOW"):
        color = COLORS["orange"]
    elif status == "UNCERTAIN":
        color = COLORS["accent"]
    else:
        color = COLORS["green"]

    # Pulsing border for alert states
    is_alert = status in ("EMPTY", "LOW", "CRITICAL LOW")
    thickness = 3 if is_alert and int(time.time() * 3) % 2 == 0 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # ── Main side level bar with gradient fill ──
    bar_h = y2 - y1
    filled = int(bar_h * level / 100)
    bar_x1 = x1 - 18
    bar_x2 = x1 - 4

    # Background
    cv2.rectangle(frame, (bar_x1, y1), (bar_x2, y2), (35, 35, 35), -1)

    # Gradient fill (green at top → red at bottom as level drops)
    if filled > 0:
        for row in range(filled):
            y_pos = y2 - row
            frac = row / max(1, bar_h)  # 0=bottom, 1=top
            r = int(0 + (0 - 0) * frac)
            g = int(50 + (210 - 50) * frac)
            b = int(255 + (80 - 255) * frac)
            cv2.line(frame, (bar_x1, y_pos), (bar_x2, y_pos), (b, g, r), 1)

    # Border
    cv2.rectangle(frame, (bar_x1, y1), (bar_x2, y2), color, 1)

    # ── Label with tracking ID ──
    label = f"IV: {status} {level:.0f}%"
    if tracking_id is not None:
        label = f"IV#{tracking_id}: {status} {level:.0f}%"
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── Camera shift indicator ──
    if camera_shifted:
        cv2.putText(frame, "[CAM SHIFT]", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["accent"], 1)

    # ── Per-method mini-bars (right side of bottle) ──
    if method_levels:
        bar_start_x = x2 + 6
        bar_w = 40
        bar_mini_h = 8
        gap = 3
        method_colors = {
            "canny": (255, 180, 50),    # Amber
            "hough": (50, 220, 100),    # Green
            "gradient": (255, 100, 50), # Coral
            "gds": (50, 150, 255),      # Blue
        }
        method_names = ["canny", "hough", "gradient", "gds"]
        for i, name in enumerate(method_names):
            val = method_levels.get(name)
            by = y1 + i * (bar_mini_h + gap)
            if by + bar_mini_h > y2:
                break
            mc = method_colors.get(name, (150, 150, 150))
            # Background
            cv2.rectangle(frame, (bar_start_x, by),
                         (bar_start_x + bar_w, by + bar_mini_h),
                         (30, 30, 30), -1)
            if val is not None and val >= 0:
                fill_w = int(bar_w * val / 100)
                cv2.rectangle(frame, (bar_start_x, by),
                             (bar_start_x + fill_w, by + bar_mini_h),
                             mc, -1)
            # Border
            cv2.rectangle(frame, (bar_start_x, by),
                         (bar_start_x + bar_w, by + bar_mini_h),
                         (80, 80, 80), 1)
            # Label
            lbl = name[0].upper()
            lbl_val = f"{val:.0f}" if val is not None else "—"
            cv2.putText(frame, f"{lbl}:{lbl_val}",
                       (bar_start_x + bar_w + 4, by + bar_mini_h - 1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
