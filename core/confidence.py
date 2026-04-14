"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CONFIDENCE AGGREGATION ENGINE                                         ║
║  Collects per-module confidence scores and computes composite          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import logging
from collections import deque
from typing import Dict, Optional

log = logging.getLogger(__name__)


class ConfidenceAggregator:
    """
    Aggregates confidence scores from all detection modules.
    Provides composite system confidence and per-module history.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = weights or {
            "fall": 0.30,
            "movement": 0.20,
            "saline": 0.20,
            "bed_zone": 0.15,
            "environment": 0.15,
        }
        self._scores: Dict[str, float] = {}
        self._history: Dict[str, deque] = {
            k: deque(maxlen=100) for k in self._weights
        }
        self._composite = 0.0
        self._last_update = time.time()

    def update(self, module: str, confidence: float):
        """Update confidence for a specific module."""
        confidence = max(0.0, min(1.0, confidence))
        self._scores[module] = confidence
        if module in self._history:
            self._history[module].append({
                "time": time.time(),
                "score": confidence
            })
        self._compute_composite()
        self._last_update = time.time()

    def _compute_composite(self):
        """Weighted average of all module confidences."""
        total_weight = 0.0
        weighted_sum = 0.0
        for module, weight in self._weights.items():
            if module in self._scores:
                weighted_sum += weight * self._scores[module]
                total_weight += weight
        self._composite = weighted_sum / total_weight if total_weight > 0 else 0.0

    @property
    def composite(self) -> float:
        return self._composite

    @property
    def scores(self) -> Dict[str, float]:
        return dict(self._scores)

    def get_module_confidence(self, module: str) -> float:
        return self._scores.get(module, 0.0)

    def get_history(self, module: str, count: int = 50) -> list:
        if module in self._history:
            return list(self._history[module])[-count:]
        return []

    def get_status(self) -> dict:
        """Full status for dashboard."""
        return {
            "composite": round(self._composite, 3),
            "modules": {k: round(v, 3) for k, v in self._scores.items()},
            "last_update": self._last_update
        }
