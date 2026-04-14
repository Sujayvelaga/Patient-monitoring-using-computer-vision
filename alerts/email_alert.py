"""
╔══════════════════════════════════════════════════════════════════════════╗
║  EMAIL ALERT — Enhanced HTML email alerts                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import smtplib
import logging
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

log = logging.getLogger(__name__)


class EmailAlert:
    """Enhanced email alerts with HTML formatting and snapshot attachments."""

    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.sender = cfg.get("sender", "")
        self.password = cfg.get("password", "")
        self.receiver = cfg.get("receiver", "")
        self.smtp_server = cfg.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = cfg.get("smtp_port", 587)

    def send(self, alert: dict, frame=None):
        """Send alert email. Called by alert engine."""
        if not self.enabled or not self.sender:
            return

        try:
            severity = alert.get("severity", "INFO")
            event_type = alert.get("event_type", "UNKNOWN")
            detail = alert.get("detail", "")
            ts = alert.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            confidence = alert.get("confidence", 0.0)
            patient_id = alert.get("patient_id", "N/A")

            # Color coding
            severity_colors = {
                "LOW": "#3498db",
                "MEDIUM": "#f39c12",
                "HIGH": "#e74c3c",
                "CRITICAL": "#c0392b",
            }
            color = severity_colors.get(severity, "#3498db")

            msg = MIMEMultipart("related")
            msg["Subject"] = f"[{severity}] Hospital Alert: {event_type} — {ts}"
            msg["From"] = self.sender
            msg["To"] = self.receiver

            html = f"""
            <html>
            <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: #16213e; border-radius: 12px; overflow: hidden; border: 1px solid {color};">
                    <div style="background: {color}; padding: 16px 24px;">
                        <h1 style="margin: 0; font-size: 20px; color: white;">
                            ⚠ {severity} ALERT: {event_type}
                        </h1>
                    </div>
                    <div style="padding: 24px;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px 0; color: #888;">Time</td>
                                <td style="padding: 8px 0; color: #fff;">{ts}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #888;">Event</td>
                                <td style="padding: 8px 0; color: #fff;">{event_type}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #888;">Severity</td>
                                <td style="padding: 8px 0; color: {color}; font-weight: bold;">{severity}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #888;">Confidence</td>
                                <td style="padding: 8px 0; color: #fff;">{confidence:.0%}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #888;">Patient ID</td>
                                <td style="padding: 8px 0; color: #fff;">#{patient_id}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #888;">Details</td>
                                <td style="padding: 8px 0; color: #fff;">{detail}</td>
                            </tr>
                        </table>
                        {"<br><p style='color: #888;'>📷 Snapshot attached below:</p><img src='cid:snapshot' style='width: 100%; border-radius: 8px;'>" if frame is not None else ""}
                    </div>
                    <div style="padding: 12px 24px; background: #0f3460; color: #666; font-size: 12px; text-align: center;">
                        Smart Healthcare Monitoring Platform v3.0 — Auto-generated alert
                    </div>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(html, "html"))

            # Attach snapshot if available
            if frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img = MIMEImage(jpeg.tobytes(), _subtype="jpeg")
                img.add_header("Content-ID", "<snapshot>")
                img.add_header("Content-Disposition", "inline", filename="alert_snapshot.jpg")
                msg.attach(img)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as s:
                s.ehlo()
                s.starttls()
                s.ehlo()
                s.login(self.sender, self.password)
                s.sendmail(self.sender, self.receiver, msg.as_string())

            log.info(f"Email alert sent: {event_type}")

        except smtplib.SMTPAuthenticationError:
            log.error("Email FAILED — use Gmail App Password")
        except Exception as e:
            log.error(f"Email FAILED: {e}")
