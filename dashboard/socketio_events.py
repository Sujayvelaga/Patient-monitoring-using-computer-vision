"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SOCKETIO EVENTS — Real-time WebSocket communication                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import logging

log = logging.getLogger(__name__)


def register_events(socketio):
    """Register SocketIO event handlers."""

    @socketio.on("connect")
    def handle_connect():
        log.info("Dashboard client connected")
        socketio.emit("server_message", {"msg": "Connected to Smart Healthcare Monitor"})

    @socketio.on("disconnect")
    def handle_disconnect():
        log.info("Dashboard client disconnected")

    @socketio.on("request_status")
    def handle_status_request():
        """Client requesting immediate status update."""
        from dashboard.routes import _system_state

        tracker = _system_state.get("tracker")
        risk = _system_state.get("risk_scorer")

        socketio.emit("status_update", {
            "tracker": tracker.get_status() if tracker else {},
            "risk": risk.get_status() if risk else {},
        })

    @socketio.on("lock_patient")
    def handle_lock_patient(data):
        """Lock patient from dashboard click."""
        from dashboard.routes import _system_state

        tracker = _system_state.get("tracker")
        if not tracker:
            return

        if "track_id" in data:
            tracker.lock_patient(int(data["track_id"]))
        elif "x" in data and "y" in data:
            tracker.lock_nearest_to(float(data["x"]), float(data["y"]))

        socketio.emit("patient_locked", {
            "locked_id": tracker.locked_patient_id
        })

    @socketio.on("set_zone_points")
    def handle_set_zone(data):
        """Set bed zone from dashboard drawing."""
        from dashboard.routes import _system_state

        bed = _system_state.get("bed_monitor")
        if bed and "points" in data:
            success = bed.set_zone(data["points"])
            socketio.emit("zone_set", {"success": success})

    @socketio.on("request_saline_info")
    def handle_saline_info():
        """Client requesting current saline detector state."""
        from dashboard.routes import _system_state

        saline = _system_state.get("saline_detector")
        if not saline:
            socketio.emit("saline_info", {})
            return

        meta = saline.last_result.metadata or {} if saline.last_result else {}
        socketio.emit("saline_info", {
            "level": saline.level,
            "status": saline.status,
            "tracking_id": saline.tracking_id,
            "camera_shifted": saline.camera_shifted,
            "method_levels": saline.method_levels,
            "using_yolo": meta.get("using_yolo", False),
            "integrity_ok": meta.get("integrity_ok", True),
        })
