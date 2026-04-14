"""
╔══════════════════════════════════════════════════════════════════════════╗
║  STREAM INGESTION — ESP32-CAM / Webcam / Video File                    ║
║  Migrated from monitor2.py + enhanced with reconnection & fallback     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import socket
import threading
import requests
import urllib.parse
import logging

log = logging.getLogger(__name__)


class ESP32Stream:
    """Raw TCP socket MJPEG stream reader for ESP32-CAM."""

    def __init__(self, url: str, reconnect_attempts: int = 8,
                 reconnect_delay: float = 2.0):
        self.url = url
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._frame = None
        self._lock = threading.Lock()
        self._alive = False
        self._ready = threading.Event()
        self.fps = 0.0
        self.latency = 0.0

        parsed = urllib.parse.urlparse(url)
        self._hostname = parsed.hostname
        self._port = parsed.port or 81
        self._path = parsed.path or "/stream"

    def connect(self):
        base_url = f"http://{self._hostname}"
        log.info(f"Connecting to ESP32-CAM at {self.url}")

        try:
            r = requests.get(base_url, timeout=8)
            r.close()
            log.info(f"ESP32 online (HTTP {r.status_code})")
        except Exception as ex:
            raise RuntimeError(
                f"Cannot reach ESP32-CAM at {base_url}: {ex}\n"
                f"  → Are PC and ESP32 on the SAME WiFi?\n"
                f"  → IP changed? Check Arduino Serial Monitor.")

        for attempt in range(self._reconnect_attempts):
            try:
                s = socket.create_connection(
                    (self._hostname, self._port), timeout=4)
                s.close()
                log.info(f"Port {self._port} open")
                break
            except OSError:
                if attempt < self._reconnect_attempts - 1:
                    log.warning(f"Waiting for port {self._port}... "
                                f"({attempt+1}/{self._reconnect_attempts})")
                    time.sleep(self._reconnect_delay)
                else:
                    raise RuntimeError(
                        f"Port {self._port} not responding after "
                        f"{self._reconnect_attempts} attempts")

        self._alive = True
        threading.Thread(target=self._reader, daemon=True).start()
        log.info("Waiting for first frame (up to 25s)...")
        if not self._ready.wait(timeout=25):
            self._alive = False
            raise RuntimeError("No frames received in 25 seconds")
        log.info("ESP32 Stream LIVE")

    def _reader(self):
        _t0 = time.time()
        _fc = 0
        while self._alive:
            sock = None
            try:
                sock = socket.create_connection(
                    (self._hostname, self._port), timeout=10)
                sock.settimeout(None)
                req = (f"GET {self._path} HTTP/1.1\r\n"
                       f"Host: {self._hostname}:{self._port}\r\n"
                       f"Connection: keep-alive\r\n"
                       f"Cache-Control: no-cache\r\n\r\n")
                sock.sendall(req.encode())

                hbuf = b""
                while b"\r\n\r\n" not in hbuf:
                    c = sock.recv(1)
                    if not c:
                        break
                    hbuf += c

                buf = b""
                while self._alive:
                    data = sock.recv(8192)
                    if not data:
                        log.warning("Stream closed by ESP32, reconnecting...")
                        break
                    buf += data
                    if len(buf) > 300_000:
                        idx = buf.rfind(b'\xff\xd8')
                        buf = buf[idx:] if idx != -1 else b""
                        continue

                    while True:
                        s = buf.find(b'\xff\xd8')
                        if s == -1:
                            break
                        e = buf.find(b'\xff\xd9', s + 2)
                        if e == -1:
                            break
                        t_recv = time.time()
                        jpg = buf[s:e + 2]
                        buf = buf[e + 2:]
                        img = cv2.imdecode(
                            np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            with self._lock:
                                self._frame = img
                            self._ready.set()
                            self.latency = (time.time() - t_recv) * 1000
                            _fc += 1
                            if _fc >= 30:
                                self.fps = _fc / (time.time() - _t0)
                                _t0 = time.time()
                                _fc = 0

            except OSError as ex:
                if self._alive:
                    log.warning(f"Socket error: {ex}, retrying in "
                                f"{self._reconnect_delay}s")
                    time.sleep(self._reconnect_delay)
            except Exception as ex:
                if self._alive:
                    log.error(f"Reader error: {ex}, retrying...")
                    time.sleep(self._reconnect_delay)
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

    def read_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._alive = False


class VideoStream:
    """Unified stream interface — ESP32, webcam, or video file."""

    def __init__(self, source, config=None):
        """
        source: str (ESP32 URL or file path) or int (webcam index)
        """
        self.source = source
        self._esp = None
        self._cap = None
        self.fps = 0.0
        self.latency = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._is_esp = isinstance(source, str) and source.startswith("http")

    def connect(self):
        if self._is_esp:
            self._esp = ESP32Stream(self.source)
            self._esp.connect()
        else:
            self._cap = cv2.VideoCapture(self.source)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Cannot open video source: {self.source}")
            log.info(f"Video source opened: {self.source}")

    def read_frame(self):
        t0 = time.time()

        if self._is_esp:
            frame = self._esp.read_frame()
            if frame is not None:
                self.fps = self._esp.fps
                self.latency = self._esp.latency
            return frame
        else:
            self._cap.grab()
            ret, frame = self._cap.retrieve()
            if not ret:
                return None
            self.latency = (time.time() - t0) * 1000
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()
            return frame

    def stop(self):
        if self._esp:
            self._esp.stop()
        if self._cap:
            self._cap.release()

    @property
    def is_live(self):
        return self._is_esp or isinstance(self.source, int)
