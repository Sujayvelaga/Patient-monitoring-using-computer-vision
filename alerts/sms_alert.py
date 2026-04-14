"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SMS ALERT — Twilio SMS for critical alerts                            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import logging

log = logging.getLogger(__name__)


class SMSAlert:
    """Twilio SMS alerts for CRITICAL events."""

    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = cfg.get("enabled", False)
        self.account_sid = cfg.get("account_sid", "")
        self.auth_token = cfg.get("auth_token", "")
        self.from_number = cfg.get("from_number", "")
        self.to_number = cfg.get("to_number", "")
        self._client = None

        if self.enabled and self.account_sid:
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
                log.info("Twilio SMS client initialized")
            except ImportError:
                log.warning("twilio package not installed. SMS alerts disabled.")
                self.enabled = False
            except Exception as e:
                log.error(f"Twilio init failed: {e}")
                self.enabled = False

    def send(self, alert: dict, frame=None):
        """Send SMS alert. Called by alert engine for CRITICAL events."""
        if not self.enabled or not self._client:
            return

        try:
            severity = alert.get("severity", "ALERT")
            event_type = alert.get("event_type", "UNKNOWN")
            detail = alert.get("detail", "")
            ts = alert.get("timestamp", "")
            confidence = alert.get("confidence", 0.0)

            body = (
                f"🏥 HOSPITAL ALERT [{severity}]\n"
                f"Event: {event_type}\n"
                f"Time: {ts}\n"
                f"Confidence: {confidence:.0%}\n"
                f"Details: {detail[:100]}\n"
                f"— Smart Healthcare Monitor"
            )

            message = self._client.messages.create(
                body=body,
                from_=self.from_number,
                to=self.to_number
            )
            log.info(f"SMS sent: {message.sid}")

        except Exception as e:
            log.error(f"SMS FAILED: {e}")
