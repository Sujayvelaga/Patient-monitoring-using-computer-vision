"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FLASK ROUTES — REST API endpoints for dashboard                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import time
import logging
from flask import Blueprint, render_template, jsonify, request, Response

log = logging.getLogger(__name__)
bp = Blueprint("dashboard", __name__)

# These will be set by main.py when initializing
_system_state = {}


def set_system_state(state: dict):
    """Set reference to shared system state from main.py."""
    global _system_state
    _system_state = state


@bp.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@bp.route("/api/status")
def api_status():
    """Current system status for all modules."""
    tracker = _system_state.get("tracker")
    fall = _system_state.get("fall_detector")
    movement = _system_state.get("movement_detector")
    saline = _system_state.get("saline_detector")
    bed = _system_state.get("bed_monitor")
    risk = _system_state.get("risk_scorer")
    hybrid = _system_state.get("hybrid_intelligence")
    confidence = _system_state.get("confidence_aggregator")
    stream = _system_state.get("stream")

    status = {
        "timestamp": time.time(),
        "stream": {
            "fps": stream.fps if stream else 0,
            "latency": stream.latency if stream else 0,
        },
        "tracker": tracker.get_status() if tracker else {},
        "fall": {
            "status": fall.last_result.status if fall else "N/A",
            "confidence": fall.last_result.confidence if fall else 0,
            "posture": fall.posture if fall else "UNKNOWN",
            "angle": fall.current_angle if fall else 0,
        } if fall else {},
        "movement": {
            "status": movement.last_result.status if movement else "N/A",
            "confidence": movement.last_result.confidence if movement else 0,
            "score": movement.current_score if movement else 0,
            "elapsed": movement.elapsed_still if movement else 0,
        } if movement else {},
        "saline": {
            "status": saline.last_result.status if saline else "N/A",
            "confidence": saline.last_result.confidence if saline else 0,
            "level": saline.level if saline else 0,
            "using_yolo": False,
            "roi_set": saline.roi_set if saline else False,
            "physics_violated": (saline.last_result.metadata or {}).get("physics_violated", False) if saline else False,
            "camera_shifted": (saline.last_result.metadata or {}).get("camera_shifted", False) if saline else False,
            "tracking_id": (saline.last_result.metadata or {}).get("tracking_id") if saline else None,
            "n_methods": (saline.last_result.metadata or {}).get("n_methods", 0) if saline else 0,
            "canny_level": (saline.last_result.metadata or {}).get("canny_level") if saline else None,
            "hough_level": (saline.last_result.metadata or {}).get("hough_level") if saline else None,
            "gradient_level": (saline.last_result.metadata or {}).get("gradient_level") if saline else None,
            "gds_level": (saline.last_result.metadata or {}).get("gds_level") if saline else None,
            "integrity_ok": (saline.last_result.metadata or {}).get("integrity_ok", True) if saline else True,
            "spread": (saline.last_result.metadata or {}).get("cue_spread", 0) if saline else 0,
        } if saline else {},
        "bed_zone": {
            "status": bed.last_result.status if bed else "N/A",
            "confidence": bed.last_result.confidence if bed else 0,
            "state": bed.state if bed else "N/A",
            "confirmed": (bed.last_result.metadata or {}).get("confirmed", False) if bed else False,
            "inside_fraction": (bed.last_result.metadata or {}).get("inside_fraction", 0) if bed else 0,
        } if bed else {},
        "risk": risk.get_status() if risk else {},
        "hybrid": hybrid.get_status() if hybrid else {},
        "confidence": confidence.get_status() if confidence else {},
    }
    return jsonify(status)


@bp.route("/api/events")
def api_events():
    """Recent events."""
    from utils.logger import LOGGER
    count = request.args.get("count", 50, type=int)
    events = LOGGER.recent(count)
    return jsonify({"events": events})


@bp.route("/api/risk")
def api_risk():
    """Risk score data."""
    risk = _system_state.get("risk_scorer")
    if not risk:
        return jsonify({"error": "Risk scorer not initialized"})

    return jsonify({
        "current": risk.get_status(),
        "history": risk.get_score_history(100),
    })


@bp.route("/api/alerts")
def api_alerts():
    """Recent alerts."""
    alert_engine = _system_state.get("alert_engine")
    if not alert_engine:
        return jsonify({"alerts": []})
    return jsonify({
        "alerts": alert_engine.recent_alerts(50),
        "counts": alert_engine.alert_count_by_severity(),
    })


@bp.route("/api/lock_patient", methods=["POST"])
def api_lock_patient():
    """Lock onto a patient ID."""
    tracker = _system_state.get("tracker")
    if not tracker:
        return jsonify({"error": "Tracker not initialized"}), 500

    data = request.json or {}
    if "track_id" in data:
        track_id = int(data["track_id"])
        success = tracker.lock_patient(track_id)
    elif "x" in data and "y" in data:
        track_id = tracker.lock_nearest_to(float(data["x"]), float(data["y"]))
        success = track_id is not None
    else:
        return jsonify({"error": "Provide track_id or x,y coordinates"}), 400

    return jsonify({
        "success": success,
        "locked_id": tracker.locked_patient_id
    })


@bp.route("/api/unlock_patient", methods=["POST"])
def api_unlock_patient():
    """Unlock patient."""
    tracker = _system_state.get("tracker")
    if tracker:
        tracker.unlock_patient()
    return jsonify({"success": True})


@bp.route("/api/set_zone", methods=["POST"])
def api_set_zone():
    """Set bed zone polygon."""
    bed = _system_state.get("bed_monitor")
    if not bed:
        return jsonify({"error": "Bed monitor not initialized"}), 500

    data = request.json or {}
    points = data.get("points", [])
    if len(points) < 3:
        return jsonify({"error": "Need at least 3 points"}), 400

    success = bed.set_zone(points)
    return jsonify({"success": success})


@bp.route("/api/reset_fall", methods=["POST"])
def api_reset_fall():
    """Reset fall detector."""
    fall = _system_state.get("fall_detector")
    if fall:
        fall.reset()
    return jsonify({"success": True})


@bp.route("/api/reset_movement", methods=["POST"])
def api_reset_movement():
    """Reset movement detector."""
    movement = _system_state.get("movement_detector")
    if movement:
        movement.reset()
    return jsonify({"success": True})


# ── Saline ROI Draw Mode ───────────────────────────────────────────────────

@bp.route("/api/set_saline_roi", methods=["POST"])
def api_set_saline_roi():
    """
    Set saline ROI from user-drawn rectangle on the video canvas.
    Body: { x1, y1, x2, y2 }  (canvas-pixel coordinates)
    Optionally: { natural_width, natural_height } for scaling from display to frame pixels.
    """
    saline = _system_state.get("saline_detector")
    if not saline:
        return jsonify({"error": "Saline detector not initialized"}), 500

    data = request.json or {}
    try:
        x1 = float(data["x1"]); y1 = float(data["y1"])
        x2 = float(data["x2"]); y2 = float(data["y2"])
    except KeyError:
        return jsonify({"error": "Provide x1,y1,x2,y2"}), 400

    # Scale from display pixels → frame pixels if natural size provided
    nw = data.get("natural_width")
    nh = data.get("natural_height")
    dw = data.get("display_width")
    dh = data.get("display_height")
    if nw and nh and dw and dh:
        sx = float(nw) / float(dw)
        sy = float(nh) / float(dh)
        x1 *= sx; x2 *= sx
        y1 *= sy; y2 *= sy

    saline.set_manual_roi((int(x1), int(y1), int(x2), int(y2)))
    
    # Auto-calibrate the new ROI so we can detect if the bottle is removed
    frame = _system_state.get("last_raw_frame")
    if frame is not None:
        saline.calibrate(frame)

    return jsonify({
        "success": True,
        "roi": [int(x1), int(y1), int(x2), int(y2)]
    })


@bp.route("/api/saline_info")
def api_saline_info():
    """Return current saline ROI and calibration state."""
    saline = _system_state.get("saline_detector")
    if not saline:
        return jsonify({})
    roi = saline._manual_roi
    return jsonify({
        "roi": list(roi) if roi else None,
        "roi_set": saline.roi_set,
        "has_calibration": saline._reference_hist is not None,
        "using_yolo": False,
        "level": saline.level,
        "status": saline.status,
    })


@bp.route("/api/calibrate_saline", methods=["POST"])
def api_calibrate_saline():
    """Lock histogram of current bottle appearance as calibration reference."""
    saline = _system_state.get("saline_detector")
    frame  = _system_state.get("last_raw_frame")
    if not saline:
        return jsonify({"error": "Saline detector not initialized"}), 500
    if frame is None:
        return jsonify({"error": "No frame available"}), 503
    ok = saline.calibrate(frame)
    return jsonify({"success": ok})


@bp.route("/api/clear_saline", methods=["POST"])
def api_clear_saline():
    """Clear the manual saline ROI."""
    saline = _system_state.get("saline_detector")
    if not saline:
        return jsonify({"error": "Saline detector not initialized"}), 500
    saline.clear_saline_roi()
    return jsonify({"success": True})


# ── Bed Zone Management ────────────────────────────────────────────────────

@bp.route("/api/clear_zone", methods=["POST"])
def api_clear_zone():
    """Remove the current bed zone. User must re-draw."""
    bed = _system_state.get("bed_monitor")
    if not bed:
        return jsonify({"error": "Bed monitor not initialized"}), 500
    bed.clear_zone()
    return jsonify({"success": True})


@bp.route("/api/zone_info")
def api_zone_info():
    """Return current zone polygon and state."""
    bed = _system_state.get("bed_monitor")
    if not bed:
        return jsonify({})
    return jsonify({
        "zone_set": bed.has_zone,
        "zone_pts": bed.zone_pts if bed.has_zone else [],
        "state": bed.state,
        "confirmed": bed._confirmed,
        "fell_from_bed": bed._fell_from_bed if hasattr(bed, "_fell_from_bed") else False,
    })





@bp.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint for browser."""
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def _generate_frames():
    """Generator for MJPEG stream."""
    while True:
        frame = _system_state.get("annotated_frame")
        if frame is not None:
            quality = _system_state.get("video_quality", 70)
            _, jpeg = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg.tobytes() + b"\r\n")
        time.sleep(0.033)  # ~30fps max

