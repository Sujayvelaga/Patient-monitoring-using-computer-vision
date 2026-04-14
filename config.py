"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SMART HEALTHCARE MONITORING PLATFORM — CONFIGURATION                  ║
║  Centralized configuration for all modules                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════
#  BASE DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
LOG_DIR = os.path.join(DATA_DIR, "logs")
CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration")

for _d in [DATA_DIR, SNAPSHOT_DIR, LOG_DIR, CALIBRATION_DIR]:
    os.makedirs(_d, exist_ok=True)


@dataclass
class StreamConfig:
    """ESP32-CAM / Webcam stream settings."""
    esp32_url: str = "http://10.16.253.140:81/stream"
    webcam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    reconnect_attempts: int = 8
    reconnect_delay: float = 2.0
    buffer_size: int = 1


@dataclass
class YOLOConfig:
    """YOLOv8 detection model settings."""
    model_path: str = "yolov8n.pt"            # Auto-downloads from ultralytics
    pose_model_path: str = "yolov8n-pose.pt"  # Auto-downloads from ultralytics
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.5
    device: str = ""  # "" = auto-select (GPU if available, else CPU)
    img_size: int = 640
    person_class_id: int = 0     # COCO class 0 = person
    bottle_class_id: int = 39    # COCO class 39 = bottle
    max_detections: int = 20


@dataclass
class TrackerConfig:
    """ByteTrack multi-person tracking settings."""
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30       # Frames to keep lost tracks
    match_thresh: float = 0.8
    locked_patient_id: Optional[int] = None


@dataclass
class FallDetectionConfig:
    """Spine-angle based fall detection settings."""
    vertical_angle_min: float = 60.0    # Degrees — standing range
    vertical_angle_max: float = 120.0
    fall_angle_threshold: float = 35.0  # Below this = fallen
    fall_confirm_frames: int = 8        # Must persist N frames
    rapid_transition_deg_per_sec: float = 100.0  # Sudden angle change
    fall_confidence_min_keypoints: int = 4  # Min visible keypoints for valid detection
    recovery_frames: int = 15           # Frames of upright before clearing fall


@dataclass
class MovementConfig:
    """Semantic no-movement detection settings."""
    no_movement_seconds: float = 30.0
    motion_threshold: float = 4.0       # Weighted joint displacement threshold
    sliding_window_size: int = 60       # Frames to analyze
    # Joint weights (higher = more important for movement detection)
    joint_weights: Dict[str, float] = field(default_factory=lambda: {
        "nose": 1.0,
        "left_shoulder": 2.0, "right_shoulder": 2.0,
        "left_elbow": 1.5, "right_elbow": 1.5,
        "left_wrist": 1.5, "right_wrist": 1.5,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_knee": 1.5, "right_knee": 1.5,
        "left_ankle": 1.0, "right_ankle": 1.0,
    })


@dataclass
class SalineConfig:
    """Adaptive saline monitoring settings — v3 Multi-Method Fusion."""
    # ── Thresholds ──
    # Bhattacharyya-distance thresholds (0..1). Legacy percent values are auto-converted.
    low_threshold: float = 0.30
    critical_threshold: float = 20.0    # Kept for compatibility with older modules
    empty_threshold: float = 0.55
    no_bottle_threshold: float = 0.80
    confirm_frames: int = 4
    calibration_frames: int = 7
    auto_tune_enabled: bool = True
    auto_tune_frames: int = 60
    # ── Detection model ──
    iv_model_path: str = ""             # Fine-tuned IV model (empty = COCO bottle)
    bottle_class_id: int = 39           # COCO class 39 = bottle
    detect_transparent: bool = True     # Also detect transparent water bottles
    # ── Edge detection ──
    canny_low: int = 50
    canny_high: int = 150
    use_adaptive_canny: bool = True
    # ── ROI ──
    roi_padding: int = 10
    meniscus_center_width_frac: float = 0.55
    # ── Smoothing ──
    history_size: int = 30
    bbox_ema_alpha: float = 0.4
    # ── Calibration ──
    calibration_file: str = os.path.join(CALIBRATION_DIR, "saline_calibration.json")
    yolo_fallback_frames: int = 30
    # ── Bottle plausibility ──
    min_bottle_area_px: int = 900
    bottle_aspect_ratio_min: float = 1.15
    bottle_aspect_ratio_max: float = 5.0
    max_bbox_center_jump_px: float = 100.0
    min_bbox_iou_with_ema: float = 0.08
    # ── Fusion & integrity ──
    methods_agreement_max_diff: float = 28.0
    low_integrity_confidence_cap: float = 0.45
    alert_confirm_frames: int = 5
    # ── Preprocessing (robustness) ──
    use_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8
    use_median_filter: bool = True
    median_kernel_size: int = 5
    # ── ORB self-recalibration ──
    orb_features: int = 500
    orb_match_threshold: float = 30.0
    camera_shift_threshold_px: float = 15.0
    # ── Optical-flow fallback ──
    use_optical_flow_lock: bool = True
    flow_max_shift_px: float = 18.0
    # ── Hough meniscus ──
    hough_meniscus: bool = True
    hough_min_line_length_frac: float = 0.35
    # ── 4-method fusion weights (must sum to 1.0) ──
    canny_weight: float = 0.25
    hough_weight: float = 0.25
    gradient_weight: float = 0.25
    gds_weight: float = 0.25
    # ── Temporal constraints ──
    temporal_jump_level_pct: float = 22.0
    temporal_jump_window: int = 5
    stale_level_std_max: float = 0.35
    stale_level_min_frames: int = 90
    # ── Alert timing ──
    no_bottle_timeout_sec: float = 10.0
    # ── Threading ──
    enable_threading: bool = True


@dataclass
class BedZoneConfig:
    """Context-aware bed monitoring settings."""
    zone_file: str = os.path.join(CALIBRATION_DIR, "bed_zone.json")
    # Posture thresholds (spine angle based)
    lying_angle_max: float = 35.0
    sitting_angle_range: tuple = (35.0, 65.0)
    standing_angle_min: float = 65.0
    edge_distance_threshold: float = 50.0  # Pixels from zone boundary = "on edge"
    # Multi-anchor + hysteresis (centroid-only spoof / occlusion resistance)
    out_of_bed_confirm_frames: int = 7
    in_bed_confirm_frames: int = 3
    min_core_keypoints_for_zone: int = 2
    torso_anchor_weight: float = 0.52
    bbox_centroid_weight: float = 0.28
    feet_anchor_weight: float = 0.20
    in_zone_vote_threshold: float = 0.45   # weighted fraction inside → "inside"
    # CSI: semantic bands, CoM, bed contact, micro-stability (image proxy)
    feet_outside_penalty: float = 0.28
    bed_contact_band_px: float = 85.0       # hip near bottom of bed polygon
    contact_confirm_frames: int = 55        # ~2s at 25–30 FPS
    com_keypoint_conf_min: float = 0.25
    hip_stability_window: int = 18
    hip_micro_motion_min_std: float = 0.35  # px std; below = very static hips


@dataclass
class AlertConfig:
    """Confidence-based alert engine settings."""
    cooldown_seconds: float = 60.0
    # Confidence thresholds for severity classification
    low_threshold: float = 0.3
    medium_threshold: float = 0.6
    critical_threshold: float = 0.8
    # Which actions per severity
    low_actions: List[str] = field(default_factory=lambda: ["log"])
    medium_actions: List[str] = field(default_factory=lambda: ["log", "email", "dashboard"])
    critical_actions: List[str] = field(default_factory=lambda: [
        "log", "email", "sms", "audio", "dashboard"])


@dataclass
class EmailConfig:
    """Email alert settings."""
    enabled: bool = True
    sender: str = "sujayvelaga@gmail.com"
    password: str = "czol ipwi omeb umuz"
    receiver: str = "goldfish81106@gmail.com"
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587


@dataclass
class TwilioConfig:
    """Twilio SMS alert settings."""
    enabled: bool = True
    account_sid: str = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token: str = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number: str = os.environ.get("TWILIO_FROM_NUMBER", "")
    to_number: str = os.environ.get("TWILIO_TO_NUMBER", "")


@dataclass
class FirebaseConfig:
    """Firebase Firestore cloud sync settings."""
    enabled: bool = True
    credentials_file: str = os.environ.get(
        "FIREBASE_CREDENTIALS", os.path.join(BASE_DIR, "firebase_credentials.json"))
    project_id: str = os.environ.get("FIREBASE_PROJECT_ID", "")
    collection_prefix: str = "hospital_monitor"
    batch_size: int = 10
    flush_interval: float = 5.0  # Seconds


@dataclass
class RiskScoringConfig:
    """Predictive health risk scoring settings."""
    # Risk component weights (must sum to 1.0)
    fall_weight: float = 0.35
    inactivity_weight: float = 0.25
    posture_weight: float = 0.20
    zone_weight: float = 0.20
    # Score thresholds
    safe_max: float = 30.0
    moderate_max: float = 60.0
    high_max: float = 80.0
    # Time windows for analysis (seconds)
    short_window: float = 3600.0     # 1 hour
    medium_window: float = 21600.0   # 6 hours
    long_window: float = 86400.0     # 24 hours
    # Decay — recent events weighted more
    decay_factor: float = 0.95


@dataclass
class HybridIntelligenceConfig:
    """Dynamic classical/DL switching settings."""
    brightness_min: float = 40.0      # Min mean brightness for good quality
    brightness_max: float = 220.0     # Max mean brightness (overexposed)
    blur_threshold: float = 100.0     # Laplacian variance below this = blurry
    # Quality score thresholds
    dl_only_threshold: float = 0.7
    hybrid_threshold: float = 0.4     # Below this = classical fallback


@dataclass
class DashboardConfig:
    """Flask web dashboard settings."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = "smart-hospital-monitor-2026"
    video_quality: int = 70  # JPEG quality for video stream (0-100)
    frame_rate: int = 15     # Target dashboard FPS


@dataclass
class Config:
    """Master configuration aggregating all modules."""
    stream: StreamConfig = field(default_factory=StreamConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    fall: FallDetectionConfig = field(default_factory=FallDetectionConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    saline: SalineConfig = field(default_factory=SalineConfig)
    bed_zone: BedZoneConfig = field(default_factory=BedZoneConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    risk: RiskScoringConfig = field(default_factory=RiskScoringConfig)
    hybrid: HybridIntelligenceConfig = field(default_factory=HybridIntelligenceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def validate(self):
        """Validate configuration on startup."""
        w = self.risk
        total = w.fall_weight + w.inactivity_weight + w.posture_weight + w.zone_weight
        assert abs(total - 1.0) < 0.01, f"Risk weights must sum to 1.0, got {total}"

        if self.email.enabled:
            assert self.email.sender, "Email sender address required"
            assert self.email.password, "Email password (app password) required"
            assert self.email.receiver, "Email receiver address required"

        if self.twilio.enabled:
            assert self.twilio.account_sid, "TWILIO_ACCOUNT_SID required"
            assert self.twilio.auth_token, "TWILIO_AUTH_TOKEN required"
            assert self.twilio.from_number, "TWILIO_FROM_NUMBER required"
            assert self.twilio.to_number, "TWILIO_TO_NUMBER required"

        if self.firebase.enabled:
            assert os.path.exists(self.firebase.credentials_file), \
                f"Firebase credentials not found: {self.firebase.credentials_file}"


# Singleton global config
CONFIG = Config()
