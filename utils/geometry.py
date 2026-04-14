"""
╔══════════════════════════════════════════════════════════════════════════╗
║  GEOMETRY UTILITIES — Angles, point-in-polygon, distances              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
import cv2
from typing import Tuple, List, Optional


def spine_angle(mid_shoulder: Tuple[float, float],
                mid_hip: Tuple[float, float]) -> float:
    """
    Calculate spine angle in degrees.
    Angle measured from horizontal:
      - ~90° = standing (vertical spine)
      - ~0°  = lying down (horizontal spine)
      - ~45° = sitting

    Args:
        mid_shoulder: (x, y) midpoint of left/right shoulder
        mid_hip: (x, y) midpoint of left/right hip

    Returns:
        Angle in degrees [0, 180]
    """
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_hip[1] - mid_shoulder[1]  # Inverted Y (screen coords)
    angle = math.degrees(math.atan2(dy, abs(dx) + 1e-6))
    return max(0.0, min(180.0, abs(angle)))


def midpoint(p1: Tuple[float, float],
             p2: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate midpoint of two points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def point_distance(p1: Tuple[float, float],
                   p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_in_polygon(point: Tuple[float, float],
                     polygon: List[Tuple[int, int]]) -> bool:
    """Check if a point is inside a polygon using OpenCV."""
    if not polygon or len(polygon) < 3:
        return True  # No zone defined = always in zone
    pts = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return result >= 0


def distance_to_polygon(point: Tuple[float, float],
                        polygon: List[Tuple[int, int]]) -> float:
    """Signed distance from point to polygon boundary. Negative = outside."""
    if not polygon or len(polygon) < 3:
        return 0.0
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), True)


def centroid_from_bbox(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    """Calculate centroid from bounding box (x, y, w, h)."""
    return (x + w / 2.0, y + h / 2.0)


def centroid_from_xyxy(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Calculate centroid from bounding box (x1, y1, x2, y2)."""
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-union for two boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def bbox_area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def fraction_of_points_inside_polygon(
        points: List[Tuple[float, float]],
        polygon: List[Tuple[int, int]],
        confidences: Optional[List[float]] = None,
        conf_threshold: float = 0.3) -> float:
    """
    Fraction of given points that lie inside the polygon (weighted if confidences).
    Ignores points with confidence below threshold when confidences provided.
    """
    if not polygon or len(polygon) < 3:
        return 1.0
    if not points:
        return 0.0
    pts = np.array(polygon, dtype=np.int32)
    num = 0.0
    den = 0.0
    for i, p in enumerate(points):
        if p is None:
            continue
        w = 1.0
        if confidences is not None and i < len(confidences):
            if confidences[i] < conf_threshold:
                continue
        d = cv2.pointPolygonTest(pts, (float(p[0]), float(p[1])), False)
        if d >= 0:
            num += w
        den += w
    return (num / den) if den > 1e-6 else 0.0


def angle_change_rate(angles: list, timestamps: list) -> float:
    """
    Calculate rate of angle change (degrees per second).
    Used for rapid transition detection in fall detection.
    """
    if len(angles) < 2:
        return 0.0
    dt = timestamps[-1] - timestamps[-2]
    if dt <= 0:
        return 0.0
    da = abs(angles[-1] - angles[-2])
    return da / dt


def weighted_joint_displacement(prev_joints: np.ndarray,
                                 curr_joints: np.ndarray,
                                 weights: np.ndarray) -> float:
    """
    Compute weighted displacement of joints between two frames.

    Args:
        prev_joints: (N, 2) array of previous joint positions
        curr_joints: (N, 2) array of current joint positions
        weights: (N,) array of joint importance weights

    Returns:
        Weighted average displacement
    """
    if prev_joints is None or curr_joints is None:
        return 0.0
    if prev_joints.shape != curr_joints.shape:
        return 0.0

    displacements = np.linalg.norm(curr_joints - prev_joints, axis=1)
    if weights is not None and len(weights) == len(displacements):
        weighted = displacements * weights
        return float(np.sum(weighted) / np.sum(weights))
    return float(np.mean(displacements))


# ── YOLO-Pose keypoint indices (COCO 17-keypoint format) ──
KEYPOINT_NAMES = [
    "nose",                                          # 0
    "left_eye", "right_eye",                         # 1, 2
    "left_ear", "right_ear",                         # 3, 4
    "left_shoulder", "right_shoulder",               # 5, 6
    "left_elbow", "right_elbow",                     # 7, 8
    "left_wrist", "right_wrist",                     # 9, 10
    "left_hip", "right_hip",                         # 11, 12
    "left_knee", "right_knee",                       # 13, 14
    "left_ankle", "right_ankle",                     # 15, 16
]

KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12),                         # Torso
    (11, 12),                                 # Hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]
