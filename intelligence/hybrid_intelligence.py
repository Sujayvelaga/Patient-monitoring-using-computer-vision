"""
╔══════════════════════════════════════════════════════════════════════════╗
║  HYBRID INTELLIGENCE — Dynamic Classical/DL Switching (Novel 1)        ║
║  Combines classical CV + Deep Learning with environmental confidence   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import logging

log = logging.getLogger(__name__)


class HybridIntelligence:
    """
    NOVEL CONTRIBUTION #1: Hybrid Vision Intelligence.

    Dynamically switches between classical CV and deep learning methods
    based on environmental quality assessment:
      - Quality > 0.7  → DL only (high confidence)
      - Quality 0.4-0.7 → Hybrid DL + classical ensemble
      - Quality < 0.4  → Classical fallback with degradation warning

    This ensures robust operation under varying lighting, blur, and noise.
    """

    MODE_DL_ONLY = "DL_ONLY"
    MODE_HYBRID = "HYBRID"
    MODE_CLASSICAL = "CLASSICAL"

    def __init__(self, config=None):
        cfg = config or {}
        self.dl_threshold = cfg.get("dl_only_threshold", 0.7)
        self.hybrid_threshold = cfg.get("hybrid_threshold", 0.4)
        self._current_mode = self.MODE_DL_ONLY
        self._quality_score = 1.0
        self._mode_history = []

    def update(self, quality_score: float):
        """Update mode based on environmental quality score."""
        self._quality_score = quality_score
        prev_mode = self._current_mode

        if quality_score >= self.dl_threshold:
            self._current_mode = self.MODE_DL_ONLY
        elif quality_score >= self.hybrid_threshold:
            self._current_mode = self.MODE_HYBRID
        else:
            self._current_mode = self.MODE_CLASSICAL

        if self._current_mode != prev_mode:
            log.info(f"Vision mode changed: {prev_mode} → {self._current_mode} "
                     f"(quality={quality_score:.2f})")
            self._mode_history.append({
                "from": prev_mode,
                "to": self._current_mode,
                "quality": quality_score
            })

    @property
    def mode(self) -> str:
        return self._current_mode

    @property
    def quality(self) -> float:
        return self._quality_score

    @property
    def use_dl(self) -> bool:
        return self._current_mode in (self.MODE_DL_ONLY, self.MODE_HYBRID)

    @property
    def use_classical(self) -> bool:
        return self._current_mode in (self.MODE_CLASSICAL, self.MODE_HYBRID)

    def get_confidence_weight(self) -> dict:
        """
        Returns weight distribution for combining DL and classical results.
        """
        if self._current_mode == self.MODE_DL_ONLY:
            return {"dl": 1.0, "classical": 0.0}
        elif self._current_mode == self.MODE_HYBRID:
            # Linearly interpolate
            dl_w = (self._quality_score - self.hybrid_threshold) / \
                   (self.dl_threshold - self.hybrid_threshold)
            dl_w = max(0.3, min(0.7, dl_w))
            return {"dl": dl_w, "classical": 1.0 - dl_w}
        else:
            return {"dl": 0.0, "classical": 1.0}

    def get_status(self) -> dict:
        return {
            "mode": self._current_mode,
            "quality_score": round(self._quality_score, 3),
            "use_dl": self.use_dl,
            "use_classical": self.use_classical,
            "weights": self.get_confidence_weight(),
        }
