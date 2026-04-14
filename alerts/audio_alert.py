"""
╔══════════════════════════════════════════════════════════════════════════╗
║  AUDIO ALERT — Local audible alarms with severity-based tones          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import platform

log = logging.getLogger(__name__)


class AudioAlert:
    """Local audio alarm with severity-based tone patterns."""

    # Severity → (frequency_hz, duration_ms, repeat)
    TONES = {
        "LOW":      (800, 200, 1),
        "MEDIUM":   (1000, 300, 2),
        "HIGH":     (1200, 400, 3),
        "CRITICAL": (1500, 500, 5),
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and platform.system() == "Windows"
        if self.enabled:
            log.info("Audio alerts enabled (Windows)")
        else:
            log.info("Audio alerts disabled (non-Windows or disabled)")

    def send(self, alert: dict, frame=None):
        """Play audio alert. Called by alert engine."""
        if not self.enabled:
            return

        severity = alert.get("severity", "MEDIUM")

        try:
            threading.Thread(
                target=self._play_tone,
                args=(severity,),
                daemon=True
            ).start()
        except Exception as e:
            log.error(f"Audio alert failed: {e}")

    def _play_tone(self, severity: str):
        """Play tone pattern based on severity."""
        try:
            import winsound
            freq, duration, repeat = self.TONES.get(severity, (1000, 300, 2))
            for _ in range(repeat):
                winsound.Beep(freq, duration)
        except Exception as e:
            log.error(f"Audio tone failed: {e}")
