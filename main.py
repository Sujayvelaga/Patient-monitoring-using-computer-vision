"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SMART HEALTHCARE MONITORING PLATFORM  v3.0                            ║
║  AI-Powered · Real-Time · Predictive                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Novel Contributions:                                                   ║
║   1. Hybrid Vision Intelligence (Classical + DL)                       ║
║   2. Patient-Specific Tracking with Identity Lock                      ║
║   3. Pose-Based Fall Detection using Spine Angle                       ║
║   4. Semantic No-Movement Detection (17-joint weighted)                ║
║   5. Adaptive Saline Monitoring (YOLO + Canny)                         ║
║   6. Context-Aware Bed Monitoring                                      ║
║   7. Confidence-Based Alert System                                     ║
║   8. Edge-AI + Cloud Hybrid Architecture                               ║
║   9. Predictive Health Risk Scoring                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Usage:                                                                 ║
║   python main.py --cam 0                                               ║
║   python main.py --url http://10.16.253.140:81/stream                  ║
║   python main.py --cam 0 --no-cloud                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import cv2
import time
import argparse
import logging
import threading
import numpy as np
from datetime import datetime

# ── Setup logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("main")

# ── Configuration ──
from config import CONFIG

# ── Core ──
from core.stream import VideoStream
from core.frame_processor import FrameProcessor
from core.confidence import ConfidenceAggregator

# ── Detectors ──
from detectors.pose_estimator import PoseEstimator
from detectors.tracker import IdentityTracker
from detectors.fall_detector import FallDetector
from detectors.movement_detector import MovementDetector
from detectors.saline_detector import SalineDetector
from detectors.saline_pipeline import SalinePipeline
from detectors.bed_monitor import BedMonitor

# ── Intelligence ──
from intelligence.hybrid_intelligence import HybridIntelligence
from intelligence.alert_engine import AlertEngine
from intelligence.risk_scorer import RiskScorer

# ── Alerts ──
from alerts.email_alert import EmailAlert
from alerts.sms_alert import SMSAlert
from alerts.audio_alert import AudioAlert

# ── Cloud ──
from cloud.firebase_sync import FirebaseSync

# ── Utils ──
from utils.logger import LOGGER
from utils.drawing import (
    draw_skeleton, draw_tracking_box, draw_zone_polygon,
    draw_confidence_bar, draw_spine_angle_indicator, COLORS
)

# ── Dashboard ──
from dashboard.app import create_app, socketio
from dashboard.routes import set_system_state


# ══════════════════════════════════════════════════════════════════════════
#  SHARED STATE DICT — passed to Flask routes for API access
# ══════════════════════════════════════════════════════════════════════════
system_state = {
    "stream": None,
    "tracker": None,
    "fall_detector": None,
    "movement_detector": None,
    "saline_detector": None,
    "saline_pipeline": None,
    "bed_monitor": None,
    "risk_scorer": None,
    "hybrid_intelligence": None,
    "confidence_aggregator": None,
    "alert_engine": None,
    "annotated_frame": None,
    "video_quality": CONFIG.dashboard.video_quality,
}


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗                         ║
║   ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝                         ║
║   ███████╗██╔████╔██║███████║██████╔╝   ██║                            ║
║   ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║                            ║
║   ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║                            ║
║   ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                            ║
║                                                                        ║
║   HEALTHCARE MONITORING PLATFORM  v3.0                                 ║
║   AI-Powered · Real-Time · Predictive                                  ║
║                                                                        ║
║   Novel: Hybrid Vision · Identity Lock · Spine-Angle Fall Detection    ║
║          Semantic Movement · Adaptive Saline · Context-Aware Bed       ║
║          Confidence Alerts · Edge-Cloud · Predictive Risk Scoring      ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def run(source, args):
    """Main processing loop."""

    print_banner()

    # ── Initialize Modules ──
    log.info("Initializing modules...")

    # Stream
    stream = VideoStream(source)
    stream.connect()
    system_state["stream"] = stream

    # Frame processor
    processor = FrameProcessor(
        target_width=CONFIG.stream.frame_width,
        target_height=CONFIG.stream.frame_height
    )

    # Pose estimator (YOLOv8-Pose + ByteTrack)
    log.info("Loading AI models...")
    pose_estimator = PoseEstimator(
        model_path=CONFIG.yolo.pose_model_path,
        confidence_threshold=CONFIG.yolo.confidence_threshold,
        device=CONFIG.yolo.device
    )

    # Identity Tracker
    tracker = IdentityTracker(track_buffer=5.0)
    system_state["tracker"] = tracker

    # Detectors
    fall_detector = FallDetector(vars(CONFIG.fall))
    movement_detector = MovementDetector(vars(CONFIG.movement))
    saline_detector = SalineDetector(vars(CONFIG.saline))
    bed_monitor = BedMonitor(vars(CONFIG.bed_zone))

    system_state["fall_detector"] = fall_detector
    system_state["movement_detector"] = movement_detector
    system_state["saline_detector"] = saline_detector
    system_state["bed_monitor"] = bed_monitor

    # ── Saline threaded pipeline (alert thread + WebSocket emit) ──
    saline_pipeline = SalinePipeline(
        saline_detector=saline_detector,
        person_detector=None,  # Bottles come from context
        socketio=socketio,
        enabled=vars(CONFIG.saline).get("enable_threading", True),
    )
    system_state["saline_pipeline"] = saline_pipeline

    # Intelligence
    hybrid = HybridIntelligence(vars(CONFIG.hybrid))
    confidence_agg = ConfidenceAggregator()
    risk_scorer = RiskScorer(vars(CONFIG.risk))
    alert_engine = AlertEngine(vars(CONFIG.alert))

    system_state["hybrid_intelligence"] = hybrid
    system_state["confidence_aggregator"] = confidence_agg
    system_state["risk_scorer"] = risk_scorer
    system_state["alert_engine"] = alert_engine

    # Alerts
    email_alert = EmailAlert(vars(CONFIG.email))
    sms_alert = SMSAlert(vars(CONFIG.twilio))
    audio_alert = AudioAlert(enabled=True)
    firebase = FirebaseSync(vars(CONFIG.firebase) if not args.no_cloud else {"enabled": False})

    # Register alert handlers
    alert_engine.register_handler("email", email_alert.send)
    alert_engine.register_handler("sms", sms_alert.send)
    alert_engine.register_handler("audio", audio_alert.send)
    alert_engine.register_handler("log", lambda a, f: LOGGER.log_event(
        a["event_type"], a["severity"], a["detail"],
        a.get("confidence", 0), 0, a.get("patient_id")))
    alert_engine.register_handler("cloud", firebase.push_event)
    alert_engine.register_handler("dashboard", lambda a, f: socketio.emit(
        "alert", a, namespace="/"))

    # YOLO object detector (for bottle detection)
    # We use pose model for person tracking, but also need detection for bottles
    from detectors.person_detector import PersonDetector
    person_detector = PersonDetector(
        model_path=CONFIG.yolo.model_path,
        confidence_threshold=CONFIG.yolo.confidence_threshold,
        device=CONFIG.yolo.device
    )

    # ── Start Dashboard ──
    set_system_state(system_state)
    app = create_app(vars(CONFIG.dashboard))
    dash_thread = threading.Thread(
        target=lambda: socketio.run(
            app,
            host=CONFIG.dashboard.host,
            port=CONFIG.dashboard.port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True
        ),
        daemon=True
    )
    dash_thread.start()
    saline_pipeline.start()

    log.info(f"Dashboard running at http://localhost:{CONFIG.dashboard.port}")

    LOGGER.log_event("SYSTEM_START", "INFO",
                     "Smart Healthcare Monitor v3.0 started")

    # ── FPS tracking ──
    fps_counter = 0
    fps_timer = time.time()
    display_fps = 0.0

    # ── Auto-lock flag ──
    auto_locked = False

    print(f"\n  ✓ System running — Dashboard: http://localhost:{CONFIG.dashboard.port}")
    print(f"  ✓ Video source: {source}")
    print(f"  ✓ Press Ctrl+C to stop\n")

    # ══════════════════════════════════════════════════════════════════
    #  MAIN PROCESSING LOOP
    # ══════════════════════════════════════════════════════════════════
    try:
        while True:
            t_start = time.time()

            # ── 1. Grab Frame ──
            frame = stream.read_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # ── 2. Preprocess ──
            frame = processor.preprocess(frame)
            if frame is None:
                continue
                
            # Store unannotated frame for UI operations (Calibration / ROI Set)
            system_state["last_raw_frame"] = frame.copy()

            # ── 3. Environmental Quality Assessment ──
            quality = processor.assess_quality(frame)
            hybrid.update(quality)
            confidence_agg.update("environment", quality)

            # ── 4. YOLO Object Detection (bottles) ──
            yolo_results = person_detector.detect(frame)
            bottles = yolo_results.get("bottles", [])

            # ── 5. Pose Estimation + ByteTrack Tracking ──
            tracked_poses = pose_estimator.estimate_with_tracking(frame)

            # ── 6. Update Identity Tracker ──
            tracker.update(tracked_poses)

            # Auto-lock first person if not locked yet
            if not auto_locked and tracker.person_count > 0 and tracker.locked_patient_id is None:
                tracker.auto_lock_first()
                auto_locked = True
                log.info(f"Auto-locked patient: #{tracker.locked_patient_id}")

            # ── 7. Build Context for Detectors ──
            context = tracker.get_context()
            context["quality_score"] = quality
            context["bottles"] = bottles

            # ── 8. Draw All Tracked Persons ──
            for person in tracker.active_persons:
                if person.bbox is not None:
                    is_locked = (person.track_id == tracker.locked_patient_id)
                    draw_tracking_box(
                        frame, person.bbox, person.track_id,
                        is_locked=is_locked,
                        confidence=person.bbox_confidence
                    )
                if person.keypoints is not None:
                    color = (255, 220, 0) if person.is_patient else (100, 100, 100)
                    draw_skeleton(frame, person.keypoints, color=color,
                                  thickness=2 if person.is_patient else 1)

            # ── 9. Run Detectors (on locked patient only) ──
            fall_result = fall_detector.process(frame, context)
            movement_result = movement_detector.process(frame, context)

            # Add posture info to context for bed monitor
            context["posture"] = fall_detector.posture
            context["spine_angle"] = fall_detector.current_angle

            saline_result = saline_pipeline.process_frame(frame, context)
            bed_result = bed_monitor.process(frame, context)

            # ── Overrides / Conflict Resolution ──
            # If the bed zone is defined and the patient is safely inside it,
            # they are not 'falling', they are just lying/sitting on the bed.
            # We also suppress during the brief "INITIALIZING" period if raw_inside is True.
            
            is_secure = bed_result.status in ("IN BED LYING", "IN BED SITTING")
            is_init_secure = bed_result.status == "INITIALIZING" and bed_result.metadata and bed_result.metadata.get("raw_inside", False)
            
            if bed_monitor.zone_pts is not None and (is_secure or is_init_secure):
                if fall_result.status == "FALL DETECTED":
                    fall_result.status = "SECURE IN BED"
                    fall_result.should_alert = False
                    fall_result.severity = "LOW"
                    fall_result.confidence = 0.0

            # ── 10. Annotate Frame ──
            frame = fall_detector.annotate(frame, fall_result)
            frame = movement_detector.annotate(frame, movement_result)
            frame = saline_detector.annotate(frame, saline_result)
            frame = bed_monitor.annotate(frame, bed_result)

            # ── 11. Update Confidence Aggregator ──
            confidence_agg.update("fall", fall_result.confidence)
            confidence_agg.update("movement", movement_result.confidence)
            confidence_agg.update("saline", saline_result.confidence)
            confidence_agg.update("bed_zone", bed_result.confidence)

            # ── 12. Update Risk Scorer ──
            risk_scorer.update_continuous(
                posture=fall_detector.posture,
                is_inactive=(movement_result.status == "NO MOVEMENT"),
                inactivity_duration=movement_detector.elapsed_still
            )

            # ── 13. Trigger Alerts ──
            for result, event_type in [
                (fall_result,     "PATIENT_FALL"),
                (movement_result, "NO_MOVEMENT"),
                (saline_result,   "SALINE_" + saline_result.status),
                (bed_result,      "BED_" + bed_result.status.replace(" ", "_")),
            ]:
                # Never alert on INITIALIZING state — not enough data yet
                if "INITIALIZING" in result.status.upper():
                    continue
                if result.should_alert:
                    alert_engine.trigger(
                        event_type=event_type,
                        confidence=result.confidence,
                        detail=result.detail,
                        severity=result.severity,
                        patient_id=tracker.locked_patient_id,
                        frame=frame,
                    )
                    # Record in risk scorer
                    risk_scorer.record_event(
                        event_type, result.severity, result.confidence)

            # ── 14. Draw HUD Overlay ──
            h, w = frame.shape[:2]

            # Top status bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

            # Status indicators
            statuses = [
                ("Fall",  fall_result.status,     fall_result.confidence),
                ("Move",  movement_result.status,  movement_result.confidence),
                ("IV",    saline_result.status,    saline_result.confidence),
                ("Bed",   bed_result.status,       bed_result.confidence),
            ]
            bx = 10
            for name, status, conf in statuses:
                is_alert = any(w in status.upper() for w in
                               ["FALL", "NO MOV", "EMPTY", "LOW", "OUT"])
                is_warn  = (not is_alert) and any(w in status.upper() for w in
                                                   ["UNCERTAIN", "EDGE", "NEAR"])
                is_init  = "INIT" in status.upper() or status in ("NO PATIENT", "NO ZONE",
                                                                    "NOT DETECTED", "LOW VISIBILITY")
                if is_alert:
                    color = COLORS["red"];    icon = "▲"
                elif is_warn:
                    color = COLORS["orange"]; icon = "◆"
                elif is_init:
                    color = COLORS["dim"];    icon = "○"
                else:
                    color = COLORS["green"];  icon = "●"
                text = f"{icon} {name}: {status} ({conf:.0%})"
                cv2.putText(frame, text, (bx, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                (tw, _), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                bx += tw + 16

            # FPS counter
            fps_counter += 1
            if fps_counter >= 20:
                display_fps = fps_counter / (time.time() - fps_timer)
                fps_timer = time.time()
                fps_counter = 0

            fps_col = COLORS["green"] if display_fps > 10 else (
                COLORS["orange"] if display_fps > 5 else COLORS["red"])
            cv2.putText(frame, f"{display_fps:.1f} FPS",
                        (w - 90, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, fps_col, 1)

            # Risk score badge
            risk_score = risk_scorer.score
            risk_level = risk_scorer.level
            risk_color = {
                "SAFE": COLORS["green"],
                "MODERATE": COLORS["yellow"],
                "HIGH": COLORS["orange"],
                "CRITICAL": COLORS["red"]
            }.get(risk_level, COLORS["green"])
            cv2.putText(frame, f"Risk: {risk_score:.0f} [{risk_level}]",
                        (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, risk_color, 2)

            # Vision mode badge
            cv2.putText(frame, f"Mode: {hybrid.mode}",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, COLORS["dim"], 1)

            # ── 15. Update shared state for dashboard ──
            system_state["annotated_frame"] = frame.copy()

            # ── 16. Periodic cloud sync ──
            # (Firebase sync happens in background via alert handler)

            # ── Frame timing ──
            elapsed = time.time() - t_start
            # Small sleep to prevent 100% CPU
            if elapsed < 0.020:
                time.sleep(0.020 - elapsed)

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        LOGGER.log_event("SYSTEM_STOP", "INFO", "Monitor stopped")
        try:
            saline_pipeline.stop()
        except Exception as e:
            log.debug("Saline pipeline stop warning: %s", e)
        stream.stop()
        log.info("System stopped.")


# ══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Smart Healthcare Monitoring Platform v3.0"
    )
    parser.add_argument("--url",
                        default=CONFIG.stream.esp32_url,
                        help="ESP32-CAM stream URL")
    parser.add_argument("--cam",
                        type=int, default=None,
                        help="Webcam index (e.g., 0)")
    parser.add_argument("--video",
                        type=str, default=None,
                        help="Video file path for testing")
    parser.add_argument("--port",
                        type=int, default=CONFIG.dashboard.port,
                        help="Dashboard port (default 5000)")
    parser.add_argument("--no-cloud",
                        action="store_true",
                        help="Disable Firebase cloud sync")
    parser.add_argument("--no-email",
                        action="store_true",
                        help="Disable email alerts")
    parser.add_argument("--device",
                        type=str, default="",
                        help="YOLO device ('' = auto, 'cpu', '0' for GPU)")

    args = parser.parse_args()

    # Apply CLI overrides
    CONFIG.dashboard.port = args.port
    if args.no_email:
        CONFIG.email.enabled = False
    if args.device:
        CONFIG.yolo.device = args.device

    # Determine source
    if args.video:
        source = args.video
    elif args.cam is not None:
        source = args.cam
    else:
        source = args.url

    try:
        CONFIG.validate()
    except AssertionError as e:
        log.error(f"Configuration error: {e}")
        sys.exit(1)

    run(source, args)


if __name__ == "__main__":
    main()
