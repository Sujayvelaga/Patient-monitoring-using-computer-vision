"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ALERT ENGINE — Confidence-Based Smart Alert System (Novel 7)          ║
║  Severity classification + multi-channel dispatch                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Optional, Callable
from collections import deque

log = logging.getLogger(__name__)


class AlertEngine:
    """
    NOVEL CONTRIBUTION #7: Confidence-Based Alert System.

    Key improvements:
    1. Each detection module outputs confidence score [0.0-1.0]
    2. Alerts triggered ONLY when confidence exceeds threshold
    3. Severity classification: LOW → MEDIUM → HIGH → CRITICAL
    4. Multi-channel dispatch: log, email, SMS, audio, dashboard
    5. Per-event-type cooldown to prevent alert flooding
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.cooldown_sec = cfg.get("cooldown_seconds", 60.0)
        self.low_threshold = cfg.get("low_threshold", 0.3)
        self.medium_threshold = cfg.get("medium_threshold", 0.6)
        self.high_threshold = cfg.get("critical_threshold", 0.8)

        self._last_alert_time: Dict[str, float] = {}
        self._alert_history = deque(maxlen=500)
        self._handlers: Dict[str, Callable] = {}
        self._alert_callbacks = []

    def register_handler(self, channel: str, handler: Callable):
        """
        Register alert handler for a channel.
        Channels: 'email', 'sms', 'audio', 'dashboard', 'log', 'cloud'
        """
        self._handlers[channel] = handler
        log.info(f"Alert handler registered: {channel}")

    def register_callback(self, callback: Callable):
        """Register a callback that fires on every alert (for dashboard)."""
        self._alert_callbacks.append(callback)

    def classify_severity(self, confidence: float) -> str:
        """Classify alert severity based on confidence score."""
        if confidence >= self.high_threshold:
            return "CRITICAL"
        elif confidence >= self.medium_threshold:
            return "HIGH"
        elif confidence >= self.low_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def get_channels_for_severity(self, severity: str) -> list:
        """Determine which alert channels to use for a given severity."""
        channels = {
            "LOW": ["log"],
            "MEDIUM": ["log", "email", "dashboard"],
            "HIGH": ["log", "email", "dashboard", "audio"],
            "CRITICAL": ["log", "email", "sms", "dashboard", "audio", "cloud"],
        }
        return channels.get(severity, ["log"])

    def should_alert(self, event_type: str) -> bool:
        """Check if alert cooldown has passed for this event type."""
        now = time.time()
        last = self._last_alert_time.get(event_type, 0)
        return (now - last) >= self.cooldown_sec

    def trigger(self, event_type: str, confidence: float,
                detail: str, severity: Optional[str] = None,
                patient_id: Optional[int] = None,
                frame=None, metadata: dict = None):
        """
        Process an alert event.

        Args:
            event_type: e.g., "PATIENT_FALL", "NO_MOVEMENT", "SALINE_LOW"
            confidence: [0.0-1.0] from detector
            detail: Human-readable description
            severity: Override auto-classification if provided
            patient_id: Tracked patient ID
            frame: Video frame for snapshot
            metadata: Additional data
        """
        if confidence < self.low_threshold:
            return  # Below minimum threshold, ignore

        if not self.should_alert(event_type):
            return  # Cooldown not expired

        # Auto-classify severity if not provided
        if severity is None:
            severity = self.classify_severity(confidence)

        self._last_alert_time[event_type] = time.time()

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            "timestamp": ts,
            "event_type": event_type,
            "severity": severity,
            "confidence": round(confidence, 3),
            "detail": detail,
            "patient_id": patient_id,
            "metadata": metadata or {},
        }
        self._alert_history.appendleft(alert)

        log.warning(f"ALERT [{severity}] {event_type}: {detail} "
                    f"(conf={confidence:.2f})")

        # Dispatch to channels
        channels = self.get_channels_for_severity(severity)
        for channel in channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    threading.Thread(
                        target=handler,
                        args=(alert, frame),
                        daemon=True
                    ).start()
                except Exception as e:
                    log.error(f"Alert handler '{channel}' failed: {e}")

        # Fire callbacks
        for cb in self._alert_callbacks:
            try:
                cb(alert)
            except Exception as e:
                log.error(f"Alert callback error: {e}")

    def recent_alerts(self, n: int = 20) -> list:
        """Get N most recent alerts."""
        return list(self._alert_history)[:n]

    def alert_count_by_severity(self, window_seconds: float = 3600) -> dict:
        """Count alerts by severity within time window."""
        cutoff = time.time() - window_seconds
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for alert in self._alert_history:
            try:
                at = datetime.strptime(
                    alert["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
                if at >= cutoff:
                    sev = alert.get("severity", "LOW")
                    counts[sev] = counts.get(sev, 0) + 1
            except Exception:
                continue
        return counts

    def get_status(self) -> dict:
        return {
            "total_alerts": len(self._alert_history),
            "recent": self.recent_alerts(5),
            "severity_counts": self.alert_count_by_severity(),
            "handlers_registered": list(self._handlers.keys()),
        }
