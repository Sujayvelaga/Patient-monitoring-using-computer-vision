"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EVENT LOGGER — Enhanced CSV + structured logging                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import csv
import time
import logging
import threading
from datetime import datetime
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)


class EventLogger:
    """
    Thread-safe structured event logger.
    Logs to CSV + Python logging + in-memory buffer for dashboard.
    """

    CSV_HEADERS = [
        "timestamp", "event_type", "severity", "detail",
        "confidence", "risk_score", "patient_id"
    ]

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        self.csv_path = os.path.join(log_dir, f"events_{date_str}.csv")
        self.events = deque(maxlen=500)
        self._lock = threading.Lock()
        self._alert_callbacks = []

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.CSV_HEADERS)

    def log_event(self, event_type: str, severity: str, detail: str,
                  confidence: float = 0.0, risk_score: float = 0.0,
                  patient_id: Optional[int] = None):
        """Log an event to CSV, memory, and Python logging."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": ts,
            "event_type": event_type,
            "severity": severity,
            "detail": detail,
            "confidence": round(confidence, 3),
            "risk_score": round(risk_score, 3),
            "patient_id": patient_id
        }

        with self._lock:
            self.events.appendleft(row)
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                writer.writerow(row)

        # Python logging
        level = {
            "LOW": logging.INFO,
            "MEDIUM": logging.WARNING,
            "HIGH": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }.get(severity, logging.INFO)
        log.log(level, f"[{severity}] {event_type}: {detail} "
                       f"(conf={confidence:.2f})")

        # Notify callbacks (for dashboard real-time updates)
        for cb in self._alert_callbacks:
            try:
                cb(row)
            except Exception as e:
                log.error(f"Alert callback error: {e}")

    def register_callback(self, callback):
        """Register a callback for new events (used by dashboard)."""
        self._alert_callbacks.append(callback)

    def recent(self, n: int = 10) -> list:
        """Get N most recent events."""
        with self._lock:
            return list(self.events)[:n]

    def get_events_since(self, timestamp: float, limit: int = 100) -> list:
        """Get events since a given UNIX timestamp."""
        with self._lock:
            result = []
            for event in self.events:
                try:
                    et = datetime.strptime(
                        event["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
                    if et >= timestamp:
                        result.append(event)
                except Exception:
                    continue
                if len(result) >= limit:
                    break
            return result

    def get_event_counts(self, window_seconds: float = 3600) -> dict:
        """Count events by type within a time window."""
        cutoff = time.time() - window_seconds
        counts = {}
        with self._lock:
            for event in self.events:
                try:
                    et = datetime.strptime(
                        event["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
                    if et >= cutoff:
                        etype = event["event_type"]
                        counts[etype] = counts.get(etype, 0) + 1
                except Exception:
                    continue
        return counts


# Singleton logger instance
LOGGER = EventLogger()
