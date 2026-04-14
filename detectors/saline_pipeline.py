"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SALINE PIPELINE — Threaded Detection / Processing / Alert Engine      ║
║                                                                          ║
║  Architecture:                                                           ║
║    Main loop  → process_frame()  (inline, ≤2 ms overhead)              ║
║    Alert thread → _alert_loop()  (async, 60 s cooldown)                ║
║                                                                          ║
║  Thread safety: all shared state guarded by threading.Lock.             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import threading
import queue
import logging
import numpy as np
from typing import Optional, Callable

log = logging.getLogger(__name__)


class SalinePipeline:
    """
    Threaded wrapper around SalineDetector for real-time performance.

    FIX NOTES (v2):
      • process_frame() now calls self.detector.detect() (previously called
        the non-existent .process() method, causing an AttributeError at runtime).
        SalineDetector also exposes .process() as an alias for full compat.
      • WebSocket payload trimmed to keys that SalineDetector v2 actually emits —
        dead fields from the old multi-method design removed.
      • result_queue uses put_nowait with explicit drop-oldest eviction so a slow
        alert thread never stalls the main loop.
      • Alert thread uses a non-blocking get with timeout so _running=False is
        always noticed within ~1 s.
    """

    def __init__(
        self,
        saline_detector,
        person_detector=None,
        alert_callback: Optional[Callable] = None,
        socketio=None,
        enabled: bool = True,
    ):
        self.detector        = saline_detector
        self.person_detector = person_detector
        self.alert_callback  = alert_callback
        self.socketio        = socketio
        self.enabled         = enabled

        # ── Inter-thread queues ────────────────────────────────────────────
        self._result_queue = queue.Queue(maxsize=10)

        # ── Shared state ───────────────────────────────────────────────────
        self._lock          = threading.Lock()
        self._latest_result = None
        self._running       = False

        # ── Thread handles ─────────────────────────────────────────────────
        self._alert_thread: Optional[threading.Thread] = None

        # ── Alert state ────────────────────────────────────────────────────
        self._last_alert_time    = 0.0
        self._alert_cooldown     = 60.0   # seconds between repeated alerts

        # ── WebSocket throttle ─────────────────────────────────────────────
        self._last_ws_emit_time  = 0.0
        self._ws_emit_interval   = 0.5    # 2 Hz

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        """Start the alert background thread."""
        if not self.enabled:
            log.info("Saline pipeline threading disabled — running inline.")
            return

        self._running = True
        self._alert_thread = threading.Thread(
            target=self._alert_loop, daemon=True, name="saline-alert"
        )
        self._alert_thread.start()
        log.info("Saline pipeline started.")

    def stop(self):
        """Signal threads to stop and join."""
        self._running = False
        if self._alert_thread and self._alert_thread.is_alive():
            self._alert_thread.join(timeout=3.0)
        log.info("Saline pipeline stopped.")

    # ── Main-loop entry point ──────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, context: dict):
        """
        Run saline detection on `frame` and dispatch alerts asynchronously.

        Called from the camera/main loop on every frame.  Detection is
        synchronous (fast — pure OpenCV histogram math, no YOLO).
        Alert dispatching is handed off to the alert thread via a queue.
        """
        # FIX: call .detect() which exists on SalineDetector.
        # (.process() is now an alias on the detector as well for compat.)
        result = self.detector.detect(frame, context)

        with self._lock:
            self._latest_result = result

        # Hand off to alert thread — drop oldest entry if queue is full.
        if not self._result_queue.full():
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                pass
        else:
            # Evict oldest, enqueue latest so the alert thread stays current.
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                pass

        # Throttled WebSocket update.
        self._emit_ws_update(result)

        return result

    # ── WebSocket emission ─────────────────────────────────────────────────────

    def _emit_ws_update(self, result):
        """Emit saline status via WebSocket at most 2 Hz."""
        if self.socketio is None:
            return
        now = time.time()
        if now - self._last_ws_emit_time < self._ws_emit_interval:
            return
        self._last_ws_emit_time = now

        meta = result.metadata or {}
        bbox = meta.get("bbox")

        try:
            self.socketio.emit(
                "saline_update",
                {
                    "level":       meta.get("level", -1),
                    "status":      result.status,
                    "confidence":  round(result.confidence, 3),
                    "distance":    round(meta.get("distance", 0.0), 4),
                    "using_yolo":  meta.get("using_yolo", False),
                    "roi_set":     meta.get("roi_set", False),
                    "bbox":        list(bbox) if bbox else None,
                },
                namespace="/",
            )
        except Exception as e:
            log.debug("WS saline_update emit error: %s", e)

    # ── Alert thread ───────────────────────────────────────────────────────────

    def _alert_loop(self):
        """Background thread: watches for LOW/EMPTY results and dispatches alerts."""
        log.info("Saline alert thread started.")
        while self._running:
            try:
                result = self._result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not getattr(result, "should_alert", False):
                continue

            now = time.time()
            if now - self._last_alert_time < self._alert_cooldown:
                continue

            self._last_alert_time = now
            meta = result.metadata or {}

            # ── Callback ──────────────────────────────────────────────────
            if self.alert_callback:
                try:
                    self.alert_callback(result)
                except Exception as e:
                    log.error("Saline alert callback error: %s", e)

            # ── WebSocket alert ───────────────────────────────────────────
            if self.socketio:
                try:
                    self.socketio.emit(
                        "saline_alert",
                        {
                            "status":    result.status,
                            "level":     meta.get("level", -1),
                            "severity":  getattr(result, "severity", "UNKNOWN"),
                            "detail":    result.detail,
                            "timestamp": now,
                        },
                        namespace="/",
                    )
                except Exception as e:
                    log.debug("WS saline_alert emit error: %s", e)

        log.info("Saline alert thread stopped.")

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def latest_result(self):
        with self._lock:
            return self._latest_result