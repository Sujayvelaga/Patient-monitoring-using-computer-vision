"""
╔══════════════════════════════════════════════════════════════════════════╗
║  RISK SCORER — Predictive Health Risk Scoring (Novel 9)                ║
║  Analyzes temporal patterns to compute composite risk score            ║
║  Enables early warning for critical patient conditions                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import math
import logging
from collections import deque
from typing import Dict, Optional

log = logging.getLogger(__name__)


class RiskEvent:
    """A single risk-relevant event."""
    def __init__(self, event_type: str, severity: str,
                 confidence: float, timestamp: float = None):
        self.event_type = event_type
        self.severity = severity
        self.confidence = confidence
        self.timestamp = timestamp or time.time()


class RiskScorer:
    """
    NOVEL CONTRIBUTION #9: Predictive Health Risk Scoring.

    Analyzes temporal patterns across detection events to compute a
    composite risk score [0-100] enabling early warning:

    Components:
      1. Fall risk: frequency + severity of fall events
      2. Inactivity risk: total inactivity duration normalized
      3. Posture risk: time in concerning postures (lying when shouldn't be)
      4. Zone risk: frequency of bed zone breaches

    Features:
      - Sliding time windows (1hr, 6hr, 24hr)
      - Exponential decay: recent events weighted more
      - Trend detection: rising risk triggers early warning
      - Score levels: Safe (0-30), Moderate (30-60), High (60-80), Critical (80-100)
    """

    def __init__(self, config=None):
        cfg = config or {}

        # Component weights
        self.weights = {
            "fall": cfg.get("fall_weight", 0.35),
            "inactivity": cfg.get("inactivity_weight", 0.25),
            "posture": cfg.get("posture_weight", 0.20),
            "zone": cfg.get("zone_weight", 0.20),
        }

        # Score thresholds
        self.safe_max = cfg.get("safe_max", 30.0)
        self.moderate_max = cfg.get("moderate_max", 60.0)
        self.high_max = cfg.get("high_max", 80.0)

        # Decay
        self.decay_factor = cfg.get("decay_factor", 0.95)

        # Time windows (seconds)
        self.short_window = cfg.get("short_window", 3600.0)
        self.medium_window = cfg.get("medium_window", 21600.0)

        # State
        self._events = deque(maxlen=5000)
        self._risk_score = 0.0
        self._component_scores: Dict[str, float] = {
            "fall": 0.0,
            "inactivity": 0.0,
            "posture": 0.0,
            "zone": 0.0,
        }
        self._score_history = deque(maxlen=500)
        self._trend = "STABLE"
        self._level = "SAFE"

        # Continuous state tracking
        self._inactivity_total = 0.0
        self._current_posture = "UNKNOWN"
        self._posture_time = {"STANDING": 0, "SITTING": 0, "LYING": 0}
        self._last_update = time.time()

    def record_event(self, event_type: str, severity: str,
                     confidence: float):
        """Record a new risk-relevant event."""
        event = RiskEvent(event_type, severity, confidence)
        self._events.append(event)
        self._recompute()

    def update_continuous(self, posture: str = "UNKNOWN",
                          is_inactive: bool = False,
                          inactivity_duration: float = 0.0):
        """Update continuous state (called every frame or periodically)."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # Track posture duration
        if posture in self._posture_time:
            self._posture_time[posture] += dt

        self._current_posture = posture

        # Track inactivity
        if is_inactive:
            self._inactivity_total += dt

        # Recompute periodically (not every frame)
        if dt > 1.0:
            self._recompute()

    def _get_events_in_window(self, window_seconds: float) -> list:
        """Get events within a time window."""
        cutoff = time.time() - window_seconds
        return [e for e in self._events if e.timestamp >= cutoff]

    def _compute_fall_risk(self) -> float:
        """Fall risk: frequency + severity + recency."""
        events = self._get_events_in_window(self.short_window)
        fall_events = [e for e in events if "FALL" in e.event_type]

        if not fall_events:
            return 0.0

        # Base risk: number of falls
        count_risk = min(len(fall_events) * 25.0, 60.0)

        # Severity boost
        high_severity = sum(1 for e in fall_events
                            if e.severity in ("HIGH", "CRITICAL"))
        severity_risk = high_severity * 15.0

        # Recency boost: most recent fall
        most_recent = max(fall_events, key=lambda e: e.timestamp)
        recency = time.time() - most_recent.timestamp
        recency_risk = max(0, 30.0 - (recency / 60.0))  # Decays over 30 min

        # Confidence-weighted
        avg_conf = sum(e.confidence for e in fall_events) / len(fall_events)

        return min(100.0, (count_risk + severity_risk + recency_risk) * avg_conf)

    def _compute_inactivity_risk(self) -> float:
        """Inactivity risk: total still time normalized."""
        events = self._get_events_in_window(self.short_window)
        inactivity_events = [e for e in events if "MOVEMENT" in e.event_type
                             or "INACTIV" in e.event_type]

        # Base risk from total inactivity time
        # Assume 30+ minutes of continuous inactivity = concerning
        time_risk = min(self._inactivity_total / 1800.0 * 50.0, 50.0)

        # Event-based risk
        event_risk = min(len(inactivity_events) * 10.0, 40.0)

        # Confidence
        if inactivity_events:
            avg_conf = sum(e.confidence for e in inactivity_events) / \
                       len(inactivity_events)
        else:
            avg_conf = 0.5

        return min(100.0, (time_risk + event_risk) * avg_conf)

    def _compute_posture_risk(self) -> float:
        """Posture risk: time in concerning postures."""
        total_time = sum(self._posture_time.values()) or 1

        # Lying for extended periods without being in bed = concerning
        lying_ratio = self._posture_time.get("LYING", 0) / total_time

        # Sitting for very long = moderate concern
        sitting_ratio = self._posture_time.get("SITTING", 0) / total_time

        posture_risk = lying_ratio * 60.0 + sitting_ratio * 20.0
        return min(100.0, posture_risk)

    def _compute_zone_risk(self) -> float:
        """Zone risk: frequency of bed zone breaches."""
        events = self._get_events_in_window(self.short_window)
        zone_events = [e for e in events if "ZONE" in e.event_type
                       or "BED" in e.event_type]

        if not zone_events:
            return 0.0

        breach_events = [e for e in zone_events
                         if any(w in e.event_type for w in
                                ["BREACH", "OUT", "LEFT"])]

        count_risk = min(len(breach_events) * 20.0, 60.0)

        if breach_events:
            most_recent = max(breach_events, key=lambda e: e.timestamp)
            recency = time.time() - most_recent.timestamp
            recency_risk = max(0, 25.0 - (recency / 60.0))
            avg_conf = sum(e.confidence for e in breach_events) / \
                       len(breach_events)
        else:
            recency_risk = 0
            avg_conf = 0.5

        return min(100.0, (count_risk + recency_risk) * avg_conf)

    def _recompute(self):
        """Recompute composite risk score."""
        self._component_scores["fall"] = self._compute_fall_risk()
        self._component_scores["inactivity"] = self._compute_inactivity_risk()
        self._component_scores["posture"] = self._compute_posture_risk()
        self._component_scores["zone"] = self._compute_zone_risk()

        # Weighted composite
        self._risk_score = sum(
            self.weights[k] * self._component_scores[k]
            for k in self.weights
        )
        self._risk_score = max(0.0, min(100.0, self._risk_score))

        # Record history
        self._score_history.append({
            "time": time.time(),
            "score": self._risk_score,
            "components": dict(self._component_scores)
        })

        # Classify level
        if self._risk_score <= self.safe_max:
            self._level = "SAFE"
        elif self._risk_score <= self.moderate_max:
            self._level = "MODERATE"
        elif self._risk_score <= self.high_max:
            self._level = "HIGH"
        else:
            self._level = "CRITICAL"

        # Detect trend
        self._detect_trend()

    def _detect_trend(self):
        """Detect if risk score is rising, falling, or stable."""
        history = list(self._score_history)
        if len(history) < 5:
            self._trend = "STABLE"
            return

        recent = [h["score"] for h in history[-10:]]
        older = [h["score"] for h in history[-20:-10]] if len(history) >= 20 \
            else [h["score"] for h in history[:len(history) // 2]]

        if not older:
            self._trend = "STABLE"
            return

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        diff = avg_recent - avg_older
        if diff > 5.0:
            self._trend = "RISING"
        elif diff < -5.0:
            self._trend = "FALLING"
        else:
            self._trend = "STABLE"

    @property
    def score(self) -> float:
        return self._risk_score

    @property
    def level(self) -> str:
        return self._level

    @property
    def trend(self) -> str:
        return self._trend

    @property
    def components(self) -> Dict[str, float]:
        return dict(self._component_scores)

    def get_score_history(self, count: int = 60) -> list:
        """Get risk score history for charting."""
        return list(self._score_history)[-count:]

    def get_status(self) -> dict:
        return {
            "score": round(self._risk_score, 1),
            "level": self._level,
            "trend": self._trend,
            "components": {k: round(v, 1)
                           for k, v in self._component_scores.items()},
            "total_events": len(self._events),
        }
