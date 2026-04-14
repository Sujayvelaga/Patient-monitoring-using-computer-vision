"""
╔══════════════════════════════════════════════════════════════════════════╗
║     SMART HOSPITAL PATIENT MONITORING SYSTEM  v2.0                      ║
║     ESP32-CAM  ·  Python  ·  OpenCV  ·  Real-Time AI Detection          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Detections:                                                             ║
║   1. Saline Level        — calibrated histogram fingerprinting           ║
║   2. Patient Fall        — MOG2 + aspect-ratio + temporal confirmation   ║
║   3. No Movement         — frame-diff progressive timer                  ║
║   4. Posture Analysis    — standing / sitting / lying classification     ║
║   5. Zone Intrusion      — alert when patient leaves bed zone            ║
║   6. Session Logging     — CSV event log with timestamps                 ║
║   7. FPS & Latency HUD   — real-time performance overlay                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Keys:                                                                   ║
║   S  = Saline ROI setup       Z = Set patient bed zone                   ║
║   C  = Calibrate saline       L = Toggle event log panel                 ║
║   R  = Reset fall background  P = Save snapshot                          ║
║   F  = Toggle fullscreen      Q = Quit                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
Usage:
  python monitor.py --url http://10.16.253.140:81/stream
  python monitor.py --cam 0
"""

import cv2
import numpy as np
import time, json, os, csv, argparse, threading, smtplib, socket
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from collections import deque

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
CONFIG = {
    "esp32_stream_url":     "http://10.16.253.140:81/stream",
    "fullscreen":           False,
    "window_width":         1200,
    "window_height":        720,

    # Email alerts
    "SEND_EMAIL":           False,
    "email_sender":         "your_email@gmail.com",
    "email_password":       "xxxx xxxx xxxx xxxx",
    "email_receiver":       "nurse_station@hospital.com",

    # Saline
    "saline_low_threshold":    25,
    "saline_empty_threshold":  50,
    "calib_file":              "saline_calibration.json",

    # Fall
    "fall_aspect_ratio":    1.75,
    "fall_min_area":        8000,
    "fall_confirm_frames":  6,

    # No-movement
    "no_movement_sec":      30,
    "motion_threshold":     5.5,

    # Zone intrusion
    "zone_file":            "bed_zone.json",

    # Logging
    "log_file":             "patient_events.csv",
    "snapshot_dir":         "snapshots",

    # Alert cooldown
    "alert_cooldown_sec":   60,
}

PANEL_W    = 300
LOG_H      = 160   # height of bottom log strip when visible
ACCENT     = (255, 180,  60)   # amber
GREEN      = (  0, 210,  80)
RED        = (  0,  50, 255)
ORANGE     = (  0, 140, 255)
CYAN       = (  0, 220, 220)
WHITE      = (240, 240, 240)
DIM        = (110, 110, 130)
DARK_BG    = ( 14,  14,  18)
PANEL_BG   = ( 20,  20,  26)

os.makedirs(CONFIG["snapshot_dir"], exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  CSV EVENT LOGGER
# ══════════════════════════════════════════════════════════════════════════
class EventLogger:
    def __init__(self):
        self.path    = CONFIG["log_file"]
        self.events  = deque(maxlen=200)
        self._lock   = threading.Lock()
        # Write CSV header if new file
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "event_type", "severity", "detail"])

    def log(self, event_type: str, severity: str, detail: str):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [ts, event_type, severity, detail]
        with self._lock:
            self.events.appendleft(row)
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    def recent(self, n=8):
        with self._lock:
            return list(self.events)[:n]

LOGGER = EventLogger()


# ══════════════════════════════════════════════════════════════════════════
#  ALERT ENGINE
# ══════════════════════════════════════════════════════════════════════════
_last_alert: dict = {}

def send_alert(kind: str, msg: str, severity="HIGH", frame=None):
    now = time.time()
    if now - _last_alert.get(kind, 0) < CONFIG["alert_cooldown_sec"]:
        return
    _last_alert[kind] = now
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    LOGGER.log(kind, severity, msg)

    print(f"\n{'='*60}")
    print(f"  *** ALERT [{severity}] [{kind}]  {ts}")
    print(f"  {msg}")
    print(f"{'='*60}\n")

    # Auto-save snapshot on alert
    if frame is not None:
        snap_name = os.path.join(
            CONFIG["snapshot_dir"],
            f"{kind}_{ts.replace(':','-').replace(' ','_')}.jpg")
        cv2.imwrite(snap_name, frame)
        print(f"  Snapshot saved: {snap_name}")

    if CONFIG["SEND_EMAIL"]:
        threading.Thread(
            target=_email, args=(kind, msg, ts, severity), daemon=True).start()

def _email(kind, msg, ts, severity):
    try:
        m = MIMEMultipart()
        m["Subject"] = f"[{severity}] HOSPITAL ALERT: {kind} — {ts}"
        m["From"]    = CONFIG["email_sender"]
        m["To"]      = CONFIG["email_receiver"]
        body = (f"SMART HOSPITAL MONITORING SYSTEM\n{'='*44}\n"
                f"Alert Type : {kind}\n"
                f"Severity   : {severity}\n"
                f"Time       : {ts}\n\n"
                f"Details:\n{msg}\n\n"
                f"--- Auto-generated by Hospital Monitor v2.0 ---")
        m.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo(); s.starttls(); s.ehlo()
            s.login(CONFIG["email_sender"], CONFIG["email_password"])
            s.sendmail(CONFIG["email_sender"], CONFIG["email_receiver"], m.as_string())
        print(f"  ✓ Email sent: {kind}")
    except smtplib.SMTPAuthenticationError:
        print("  Email FAILED — use Gmail App Password")
    except Exception as e:
        print(f"  Email FAILED: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  SALINE DETECTOR  — Calibration-based Bhattacharyya histogram distance
# ══════════════════════════════════════════════════════════════════════════
class SalineDetector:
    def __init__(self):
        self.roi_px       = None
        self.calib        = None
        self.history      = deque(maxlen=20)
        self.status       = "NOT SET"
        self.level        = 100
        self.change_pct   = 0.0
        self._setup_mode  = False
        self._drawing     = False
        self._pt1 = self._pt2 = (0, 0)
        self._load_calibration()

    def _save_calibration(self):
        if self.roi_px is None or self.calib is None: return
        data = {"roi": list(self.roi_px),
                "hist": self.calib.flatten().tolist()}
        with open(CONFIG["calib_file"], "w") as f:
            json.dump(data, f)
        print("  ✓ Saline calibration saved.")

    def _load_calibration(self):
        p = CONFIG["calib_file"]
        if not os.path.exists(p): return
        try:
            with open(p) as f: data = json.load(f)
            self.roi_px = tuple(data["roi"])
            loaded_hist = np.array(data["hist"], dtype=np.float32).reshape(-1, 1)
            # Check length to prevent OpenCV assertions if the JSON is from an older version
            if loaded_hist.size == 4352:
                self.calib = loaded_hist
                self.status = "OK"
                print(f"  Saline calibration loaded: ROI={self.roi_px}")
            else:
                self.calib = None
                self.status = "NEED CALIB"
                print(f"  Old calibration discarded. Please recalibrate your saline ROI.")
        except Exception as e:
            print(f"  Could not load calibration: {e}")

    def _roi_hist(self, frame):
        x1,y1,x2,y2 = [int(v) for v in self.roi_px]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([hsv],[0,1,2],None,[16,16,16],[0,180,0,256,0,256])
        cv2.normalize(hist_hsv, hist_hsv, 0, 1, cv2.NORM_MINMAX)

        # To detect transparent objects against a background, we MUST include structural edges.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # FIX: Do not use MINMAX normalization per frame, it stretches background noise to 255!
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        ang = np.clip(ang / 2, 0, 179).astype(np.uint8)
        
        merged = cv2.merge([ang, mag])
        # 16 bins for angle, 16 bins for magnitude
        hist_grad = cv2.calcHist([merged], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist_grad, hist_grad, 0, 1, cv2.NORM_MINMAX)
        
        # Combine color and structural fingerprints
        # Do NOT apply a second NORM_MINMAX after concatenation, as it destroys the relative
        # weighting between HSV and gradient parts.
        hist = np.concatenate([hist_hsv.flatten(), hist_grad.flatten()]).astype(np.float32)
        return hist

    def calibrate(self, frame):
        if self.roi_px is None:
            print("  Draw ROI first (S → drag → S)"); return False
        hist = self._roi_hist(frame)
        if hist is None: return False
        self.calib = hist
        self.history.clear()
        self.level  = 100
        self.status = "OK"
        self._save_calibration()
        LOGGER.log("SALINE_CALIBRATED","INFO",f"ROI={self.roi_px}")
        print("  ✓ Saline calibrated! Bottle fingerprint stored.")
        return True

    def enter_setup(self):
        self._setup_mode = True
        self._drawing    = False
        print("  SALINE SETUP: Click and drag around the bottle. Press S to confirm.")

    def exit_setup(self, frame):
        self._setup_mode = False
        if self._pt1 != self._pt2:
            x1 = min(self._pt1[0], self._pt2[0])
            y1 = min(self._pt1[1], self._pt2[1])
            x2 = max(self._pt1[0], self._pt2[0])
            y2 = max(self._pt1[1], self._pt2[1])
            if x2-x1 > 10 and y2-y1 > 10:
                self.roi_px = (x1,y1,x2,y2)
                print(f"  ROI set: {self.roi_px}  →  Press C to calibrate.")
                if self.calib is None: self.status = "NEED CALIB"
        self._drawing = False

    def mouse_event(self, event, x, y, flags, param):
        if not self._setup_mode: return
        if   event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True; self._pt1 = self._pt2 = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._pt2 = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False; self._pt2 = (x,y)

    def detect(self, frame):
        if self._setup_mode:
            ov = frame.copy()
            cv2.rectangle(ov, self._pt1, self._pt2, CYAN, 2)
            if self.roi_px:
                cv2.rectangle(ov, self.roi_px[:2], self.roi_px[2:], ACCENT, 1)
            cv2.putText(ov,
                "SALINE SETUP  —  Draw box around bottle, then press S",
                (10, frame.shape[0]-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2)
            return cv2.addWeighted(ov,0.9,frame,0.1,0), "SETUP MODE"

        if self.roi_px is None:
            cv2.putText(frame,"Saline: Press S to set ROI",
                        (10,60),cv2.FONT_HERSHEY_SIMPLEX,0.52,CYAN,2)
            return frame, "NOT SET"

        if self.calib is None:
            x1,y1,x2,y2 = [int(v) for v in self.roi_px]
            cv2.rectangle(frame,(x1,y1),(x2,y2),CYAN,2)
            cv2.putText(frame,"Press C to calibrate",
                        (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.52,CYAN,2)
            return frame, "NEED CALIB"

        hist = self._roi_hist(frame)
        if hist is None: return frame, self.status

        score = float(cv2.compareHist(self.calib, hist, cv2.HISTCMP_BHATTACHARYYA))
        
        self.history.append(score)
        avg = float(np.mean(self.history))
        
        # Power-curve distance mapping
        empty_t = CONFIG["saline_empty_threshold"] / 100.0  # normalize if it's 50
        gamma = 0.6
        if avg >= empty_t:
            self.level = 0
            self.change_pct = 100.0
        else:
            fraction = 1.0 - (avg / empty_t)
            self.level = int(min(100.0, max(0.0, (fraction ** gamma) * 100.0)))
            self.change_pct = float(100.0 - self.level)

        low_t   = CONFIG["saline_low_threshold"]
        empty_t = CONFIG["saline_empty_threshold"]

        if avg >= empty_t:
            self.status = "EMPTY!"
            col = RED
            send_alert("SALINE_EMPTY","HIGH",
                "IV Saline bottle is EMPTY or missing! Replace immediately.",
                frame)
        elif avg >= low_t:
            self.status = "LOW"
            col = ORANGE
            send_alert("SALINE_LOW","MEDIUM",
                "IV Saline level LOW. Prepare a replacement.", frame)
        else:
            self.status = "OK"
            col = GREEN

        x1,y1,x2,y2 = [int(v) for v in self.roi_px]

        # Animated pulsing border on alert
        if self.status in ("EMPTY!","LOW"):
            thickness = 3 if int(time.time()*3)%2==0 else 1
        else:
            thickness = 2
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,thickness)

        # Side liquid-level bar
        bar_h  = y2-y1
        filled = int(bar_h * self.level/100)
        cv2.rectangle(frame,(x1-16,y1),(x1-4,y2),(35,35,35),-1)
        cv2.rectangle(frame,(x1-16,y2-filled),(x1-4,y2),col,-1)
        cv2.rectangle(frame,(x1-16,y1),(x1-4,y2),col,1)

        # Label with percentage
        cv2.putText(frame,f"Saline:{self.status} {self.level}%",
                    (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)
        return frame, self.status


# ══════════════════════════════════════════════════════════════════════════
#  FALL DETECTOR  — MOG2 + bounding box aspect ratio + posture classifier
# ══════════════════════════════════════════════════════════════════════════
class FallDetector:
    POSTURES = {
        "STANDING":  (0.0, 0.9),   # aspect < 0.9
        "SITTING":   (0.9, 1.4),   # 0.9 ≤ aspect < 1.4
        "LYING":     (1.4, 99.9),  # aspect ≥ 1.4
    }

    def __init__(self):
        self.bgsub      = cv2.createBackgroundSubtractorMOG2(
                              history=300, varThreshold=40, detectShadows=False)
        self.kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        self.counter    = 0
        self.status     = "NORMAL"
        self.posture    = "UNKNOWN"
        self._last_box  = None
        self._posture_hist = deque(maxlen=30)

    def reset_bg(self):
        self.bgsub   = cv2.createBackgroundSubtractorMOG2(
                           history=300, varThreshold=40, detectShadows=False)
        self.counter = 0
        self.status  = "NORMAL"
        self.posture = "UNKNOWN"
        self._last_box = None
        print("  Background model reset.")

    def _classify_posture(self, aspect):
        for name,(lo,hi) in self.POSTURES.items():
            if lo <= aspect < hi:
                return name
        return "UNKNOWN"

    def detect(self, frame):
        mask = self.bgsub.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        person_cnts = [c for c in cnts
                       if cv2.contourArea(c) >= CONFIG["fall_min_area"]]

        if not person_cnts:
            self.counter = max(0, self.counter-1)
            if self.counter == 0:
                self.status  = "NORMAL"
                self.posture = "UNKNOWN"
                self._last_box = None
            return frame, self.status

        largest   = max(person_cnts, key=cv2.contourArea)
        x,y,bw,bh = cv2.boundingRect(largest)
        aspect    = bw / max(bh, 1)
        posture   = self._classify_posture(aspect)
        self._posture_hist.append(posture)
        # Majority vote for stable posture
        if self._posture_hist:
            self.posture = max(set(self._posture_hist),
                               key=self._posture_hist.count)

        fell = aspect >= CONFIG["fall_aspect_ratio"]
        if fell: self.counter += 1
        else:    self.counter = max(0, self.counter-1)

        confirmed   = self.counter >= CONFIG["fall_confirm_frames"]
        prev_status = self.status
        self.status = "FALL DETECTED!" if confirmed else "NORMAL"
        self._last_box = (x,y,bw,bh,aspect)

        if confirmed and prev_status != "FALL DETECTED!":
            LOGGER.log("PATIENT_FALL","HIGH",
                       f"Aspect={aspect:.2f} Counter={self.counter}")

        col = RED if confirmed else (ORANGE if aspect>1.2 else GREEN)
        cv2.rectangle(frame,(x,y),(x+bw,y+bh),col,2)

        # Posture badge
        badge = f"{self.posture}  W/H:{aspect:.2f}"
        bx,by = x, max(y-10,20)
        (tw,th),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.52,2)
        cv2.rectangle(frame,(bx-2,by-th-4),(bx+tw+2,by+2),DARK_BG,-1)
        cv2.putText(frame, badge,(bx,by),cv2.FONT_HERSHEY_SIMPLEX,0.52,col,2)

        if confirmed:
            # Full-frame warning overlay
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),RED,-1)
            frame = cv2.addWeighted(frame,0.85,ov,0.15,0)
            cv2.putText(frame,"⚠ FALL DETECTED",
                        (10,frame.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,RED,3)
            send_alert("PATIENT_FALL","HIGH",
                "Patient fall detected! Immediate assistance required.",frame)

        return frame, self.status


# ══════════════════════════════════════════════════════════════════════════
#  NO-MOVEMENT DETECTOR
# ══════════════════════════════════════════════════════════════════════════
class NoMovementDetector:
    def __init__(self):
        self.prev_gray        = None
        self.last_motion_time = time.time()
        self.score            = 0.0
        self.status           = "ACTIVE"
        self._elapsed         = 0.0
        self._score_hist      = deque(maxlen=30)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(15,15),0)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy(); return frame, self.status

        diff       = cv2.absdiff(self.prev_gray, gray)
        self.score = float(diff.mean())
        self._score_hist.append(self.score)
        self.prev_gray = gray.copy()

        if self.score > CONFIG["motion_threshold"]:
            self.last_motion_time = time.time()

        self._elapsed = time.time() - self.last_motion_time
        limit         = CONFIG["no_movement_sec"]
        no_move       = self._elapsed >= limit

        self.status = "NO MOVEMENT!" if no_move else "ACTIVE"

        # Motion score sparkline on video
        h,w = frame.shape[:2]
        scores = list(self._score_hist)
        if len(scores) > 1:
            max_s = max(scores) or 1
            bar_w = min(w - PANEL_W, 200)
            for i,s in enumerate(scores[-bar_w:]):
                bh_px = int(12 * s / max_s)
                bx    = 10 + i
                col   = RED if no_move else (GREEN if s>CONFIG["motion_threshold"] else DIM)
                cv2.line(frame,(bx,h-12),(bx,h-12-bh_px),col,1)

        # Progress bar (bottom strip)
        prog  = min(self._elapsed/limit, 1.0)
        bar_x = int((w-PANEL_W)*prog)
        r     = int(255*prog); g = int(200*(1-prog))
        cv2.rectangle(frame,(0,h-8),(bar_x,h),(0,g,r),-1)
        if no_move:
            cv2.rectangle(frame,(0,h-8),(w-PANEL_W,h),RED,1)
            send_alert("NO_MOVEMENT","HIGH",
                f"No patient movement for {self._elapsed:.0f}s. Check patient!",
                frame)

        return frame, self.status


# ══════════════════════════════════════════════════════════════════════════
#  BED ZONE INTRUSION DETECTOR  (NEW)
# ══════════════════════════════════════════════════════════════════════════
class ZoneDetector:
    """
    User draws a polygon around the patient bed area.
    If the patient's bounding box centroid moves outside the zone → alert.
    """
    def __init__(self):
        self.zone_pts   = None   # list of (x,y) polygon vertices
        self._setup     = False
        self._temp_pts  = []
        self.status     = "NO ZONE"
        self._load()

    def _save(self):
        if self.zone_pts is None: return
        with open(CONFIG["zone_file"],"w") as f:
            json.dump({"pts": self.zone_pts}, f)

    def _load(self):
        p = CONFIG["zone_file"]
        if not os.path.exists(p): return
        try:
            with open(p) as f: data = json.load(f)
            self.zone_pts = data["pts"]
            self.status   = "OK"
            print(f"  Bed zone loaded: {len(self.zone_pts)} points")
        except: pass

    def enter_setup(self):
        self._setup    = True
        self._temp_pts = []
        print("  ZONE SETUP: Left-click to add polygon points.")
        print("  Right-click to finish and save.")

    def mouse_event(self, event, x, y, flags, param):
        if not self._setup: return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._temp_pts.append((x,y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(self._temp_pts) >= 3:
            self.zone_pts = self._temp_pts[:]
            self._setup   = False
            self.status   = "OK"
            self._save()
            LOGGER.log("ZONE_SET","INFO",f"{len(self.zone_pts)} points")
            print(f"  ✓ Bed zone saved: {len(self.zone_pts)} points")

    def point_in_zone(self, cx, cy):
        if self.zone_pts is None: return True
        pts = np.array(self.zone_pts, dtype=np.int32)
        return cv2.pointPolygonTest(pts,(float(cx),float(cy)),False) >= 0

    def detect(self, frame, fall_box):
        """fall_box = (x,y,w,h,aspect) or None"""
        if self._setup:
            # Draw in-progress polygon
            for pt in self._temp_pts:
                cv2.circle(frame, pt, 4, CYAN, -1)
            if len(self._temp_pts) > 1:
                cv2.polylines(frame,
                    [np.array(self._temp_pts,dtype=np.int32)],False,CYAN,1)
            cv2.putText(frame,
                "ZONE SETUP: Left-click=add point  Right-click=done",
                (10,frame.shape[0]-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, CYAN, 2)
            return frame, "SETUP"

        if self.zone_pts:
            pts = np.array(self.zone_pts, dtype=np.int32)
            cv2.polylines(frame,[pts],True,(0,180,80),2)
            # Translucent fill
            ov = frame.copy()
            cv2.fillPoly(ov,[pts],(0,60,20))
            frame = cv2.addWeighted(frame,0.85,ov,0.15,0)

        if fall_box and self.zone_pts:
            x,y,bw,bh,_ = fall_box
            cx,cy = x+bw//2, y+bh//2
            inside = self.point_in_zone(cx,cy)
            cv2.circle(frame,(cx,cy),6,
                       GREEN if inside else RED,-1)
            if not inside:
                self.status = "OUT OF ZONE!"
                send_alert("ZONE_BREACH","HIGH",
                    "Patient has moved outside the bed zone!", frame)
            else:
                self.status = "IN ZONE"
        elif self.zone_pts:
            self.status = "OK"

        return frame, self.status


# ══════════════════════════════════════════════════════════════════════════
#  ESP32-CAM STREAM  (raw socket — bypasses all urllib3 timeout bugs)
# ══════════════════════════════════════════════════════════════════════════
class ESP32Stream:
    def __init__(self, url):
        self.url       = url
        self._frame    = None
        self._lock     = threading.Lock()
        self._alive    = False
        self._ready    = threading.Event()
        self._fps      = 0.0
        self._latency  = 0.0

    def connect(self):
        import urllib.parse
        parsed         = urllib.parse.urlparse(self.url)
        self._hostname = parsed.hostname
        self._port     = parsed.port or 81
        self._path     = parsed.path or "/stream"
        base_url       = f"{parsed.scheme}://{self._hostname}"

        print(f"  Connecting → {self.url}")
        print(f"  Checking ESP32 at {base_url} …")
        try:
            r = requests.get(base_url, timeout=8); r.close()
            print(f"  ✓ ESP32 online (HTTP {r.status_code})")
        except Exception as ex:
            raise RuntimeError(
                f"\n  Cannot reach ESP32-CAM at {base_url}\n  {ex}\n\n"
                f"  → Are PC and ESP32 on the SAME WiFi?\n"
                f"  → College WiFi? Use mobile hotspot.\n"
                f"  → IP changed? Check Arduino Serial Monitor.")

        print(f"  Checking port {self._port} …")
        for attempt in range(8):
            try:
                s = socket.create_connection((self._hostname,self._port),timeout=4)
                s.close()
                print(f"  ✓ Port {self._port} open")
                break
            except OSError:
                if attempt < 7:
                    print(f"  Waiting for port {self._port}… ({attempt+1}/8)")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Port {self._port} not responding.\n"
                        f"  Open {base_url} in browser → keep tab open\n"
                        f"  Then re-run this script.")

        self._alive = True
        threading.Thread(target=self._reader, daemon=True).start()
        print("  Waiting for first frame (up to 25s) …")
        if not self._ready.wait(timeout=25):
            self._alive = False
            raise RuntimeError(
                "No frames in 25s.\n"
                f"  Open {base_url} in browser and keep that tab open.")
        print("  ✓ Stream LIVE\n")

    def _reader(self):
        _t0 = time.time(); _fc = 0
        while self._alive:
            sock = None
            try:
                sock = socket.create_connection(
                    (self._hostname, self._port), timeout=10)
                sock.settimeout(None)
                req  = (f"GET {self._path} HTTP/1.1\r\n"
                        f"Host: {self._hostname}:{self._port}\r\n"
                        f"Connection: keep-alive\r\n"
                        f"Cache-Control: no-cache\r\n\r\n")
                sock.sendall(req.encode())

                hbuf = b""
                while b"\r\n\r\n" not in hbuf:
                    c = sock.recv(1)
                    if not c: break
                    hbuf += c

                print("  Raw socket connected — reading MJPEG…")
                buf = b""
                while self._alive:
                    data = sock.recv(8192)
                    if not data:
                        print("  Stream closed by ESP32 — reconnecting…"); break
                    buf += data
                    if len(buf) > 300_000:
                        idx = buf.rfind(b'\xff\xd8')
                        buf = buf[idx:] if idx!=-1 else b""; continue

                    while True:
                        s = buf.find(b'\xff\xd8')
                        if s==-1: break
                        e = buf.find(b'\xff\xd9',s+2)
                        if e==-1: break
                        t_recv = time.time()
                        jpg = buf[s:e+2]; buf = buf[e+2:]
                        img = cv2.imdecode(np.frombuffer(jpg,np.uint8),
                                           cv2.IMREAD_COLOR)
                        if img is not None:
                            with self._lock:
                                self._frame = img
                            self._ready.set()
                            self._latency = (time.time()-t_recv)*1000
                            _fc += 1
                            if _fc >= 30:
                                self._fps = _fc/(time.time()-_t0)
                                _t0=time.time(); _fc=0

            except OSError as ex:
                if self._alive:
                    print(f"  Socket: {ex} — retry 2s…"); time.sleep(2)
            except Exception as ex:
                if self._alive:
                    print(f"  Reader: {ex} — retry 2s…"); time.sleep(2)
            finally:
                if sock:
                    try: sock.close()
                    except: pass

    def read_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self): self._alive = False


# ══════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════
def draw_panel(canvas, saline, fall, nomove, zone,
               fps, latency, frame_w, show_log):
    h  = canvas.shape[0]
    px = frame_w
    pw = PANEL_W
    canvas[:,px:px+pw] = PANEL_BG

    def txt(text, row, col=WHITE, scale=0.46, bold=1):
        cv2.putText(canvas, str(text),(px+10,row),
                    cv2.FONT_HERSHEY_SIMPLEX,scale,col,bold)

    def hrule(y, col=(45,45,55)):
        cv2.line(canvas,(px+6,y),(px+pw-6,y),col,1)

    def status_badge(label, status, y, ok_states=("NORMAL","OK","ACTIVE","IN ZONE")):
        is_ok  = status in ok_states
        col    = GREEN if is_ok else RED
        badge_col = (20,60,20) if is_ok else (60,10,10)
        cv2.rectangle(canvas,(px+8,y-14),(px+pw-8,y+6),badge_col,-1)
        cv2.rectangle(canvas,(px+8,y-14),(px+pw-8,y+6),col,1)
        cv2.putText(canvas,f"{label}: {status}",(px+12,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.44,col,1)
        return y+22

    y = 28
    # Header
    cv2.rectangle(canvas,(px,0),(px+pw,50),(25,25,35),-1)
    txt("HOSPITAL MONITOR v2.0", y,(255,220,80),0.50,2); y+=22
    txt(datetime.now().strftime("%d %b %Y   %H:%M:%S"),y,DIM); y+=4
    hrule(y); y+=14

    # Performance
    fps_col = GREEN if fps>10 else (ORANGE if fps>5 else RED)
    txt(f"Stream  {fps:.1f} FPS   Latency {latency:.0f}ms",y,fps_col); y+=20
    hrule(y); y+=12

    # ── Saline ──
    txt("IV SALINE", y, ACCENT, 0.48,2); y+=18
    y = status_badge("Level", saline.status, y,
                     ok_states=("OK","NEED CALIB","NOT SET","SETUP MODE"))

    if saline.roi_px and saline.calib is not None:
        bx1,bx2 = px+10, px+pw-10
        by1,by2 = y, y+14
        cv2.rectangle(canvas,(bx1,by1),(bx2,by2),(35,35,35),-1)
        lv = saline.level
        col = GREEN if lv>60 else (ORANGE if lv>25 else RED)
        fx = bx1+int((bx2-bx1)*lv/100)
        cv2.rectangle(canvas,(bx1,by1),(fx,by2),col,-1)
        cv2.rectangle(canvas,(bx1,by1),(bx2,by2),(80,80,80),1)
        txt(f"Level: {lv}%  Δ:{saline.change_pct:.0f}%",by2+14,col)
        y += 38
    txt("S=ROI  C=Calibrate",y,DIM,0.40); y+=18
    hrule(y); y+=12

    # ── Fall / Posture ──
    txt("FALL + POSTURE", y, ACCENT, 0.48,2); y+=18
    y = status_badge("Fall", fall.status, y)
    p_col = {"STANDING":GREEN,"SITTING":ORANGE,"LYING":RED}.get(fall.posture,DIM)
    txt(f"Posture: {fall.posture}", y, p_col); y+=18
    if fall._last_box:
        _,_,_,_,ar = fall._last_box
        txt(f"W/H ratio: {ar:.2f}  (fall>{CONFIG['fall_aspect_ratio']})",
            y, DIM, 0.40); y+=16
    txt("R=Reset BG", y, DIM, 0.40); y+=18
    hrule(y); y+=12

    # ── Movement ──
    txt("MOVEMENT", y, ACCENT, 0.48,2); y+=18
    y = status_badge("Motion", nomove.status,y,
                     ok_states=("ACTIVE",))
    txt(f"Score: {nomove.score:.2f}  (min {CONFIG['motion_threshold']})",
        y, DIM, 0.40); y+=16
    elapsed = nomove._elapsed
    limit   = CONFIG["no_movement_sec"]
    prog    = min(elapsed/limit,1.0)
    bx1,bx2 = px+10,px+pw-10
    cv2.rectangle(canvas,(bx1,y),(bx2,y+10),(35,35,35),-1)
    pcol = RED if prog>0.8 else (ORANGE if prog>0.5 else GREEN)
    cv2.rectangle(canvas,(bx1,y),(bx1+int((bx2-bx1)*prog),y+10),pcol,-1)
    txt(f"Still: {elapsed:.0f}s / {limit}s",y+22,DIM,0.40); y+=36
    hrule(y); y+=12

    # ── Zone ──
    txt("BED ZONE", y, ACCENT, 0.48,2); y+=18
    z_ok = zone.status in ("OK","IN ZONE","NO ZONE")
    y = status_badge("Zone", zone.status, y,
                     ok_states=("OK","IN ZONE","NO ZONE"))
    txt("Z=Set zone",y,DIM,0.40); y+=18
    hrule(y); y+=12

    # ── Log ──
    txt(f"EVENTS  (L=toggle log)", y, ACCENT, 0.48,2); y+=18
    recent = LOGGER.recent(5)
    for row in recent:
        ts_short = row[0][11:]   # HH:MM:SS
        sev_col  = RED if row[2]=="HIGH" else (ORANGE if row[2]=="MEDIUM" else DIM)
        txt(f"{ts_short} {row[1]}",y,sev_col,0.38,1); y+=14
        if y > h-30: break

    # Bottom hint
    cv2.rectangle(canvas,(px,h-24),(px+pw,h),(18,18,24),-1)
    txt("P=Snapshot  Q=Quit  F=Full",h-8,DIM,0.38)


def draw_topbar(canvas, statuses, vid_w):
    bar_h = 40
    roi   = canvas[0:bar_h, 0:vid_w]
    canvas[0:bar_h, 0:vid_w] = cv2.addWeighted(
        roi,0.2,np.zeros_like(roi),0.8,0)
    bx = 8
    for name,st,col in statuses:
        alarm = any(w in st for w in
                    ["EMPTY","FALL","NO MOV","LOW","BREACH","OUT"])
        icon  = "▲" if alarm else "●"
        lbl   = f"{icon} {name}: {st}"
        cv2.putText(canvas,lbl,(bx,26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,col,2)
        (tw,_),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.52,2)
        bx += tw+18


def draw_log_strip(canvas, vid_w, total_h, log_h):
    """Bottom strip showing recent events in a table."""
    y0 = total_h - log_h
    cv2.rectangle(canvas,(0,y0),(vid_w,total_h),(12,12,16),-1)
    cv2.line(canvas,(0,y0),(vid_w,y0),(60,60,80),1)
    cv2.putText(canvas,"EVENT LOG",(8,y0+16),
                cv2.FONT_HERSHEY_SIMPLEX,0.46,ACCENT,1)
    rows = LOGGER.recent(6)
    for i,row in enumerate(rows):
        ry   = y0+34+i*20
        sev_col = RED if row[2]=="HIGH" else (ORANGE if row[2]=="MEDIUM" else GREEN)
        line = f"  {row[0][11:]}  [{row[2]}]  {row[1]:<20}  {row[3][:50]}"
        cv2.putText(canvas,line,(8,ry),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,sev_col,1)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
def run(source):
    saline = SalineDetector()
    fall   = FallDetector()
    nomove = NoMovementDetector()
    zone   = ZoneDetector()

    use_esp = isinstance(source, str)
    esp = cap = None
    if use_esp:
        esp = ESP32Stream(source)
        esp.connect()
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    WIN = "Smart Hospital Patient Monitoring System  v2.0"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CONFIG["window_width"], CONFIG["window_height"])

    def mouse_cb(event, x, y, flags, param):
        saline.mouse_event(event,x,y,flags,param)
        zone.mouse_event(event,x,y,flags,param)
    cv2.setMouseCallback(WIN, mouse_cb)

    if CONFIG["fullscreen"]:
        cv2.setWindowProperty(WIN,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    print("\n  ╔═══════════════════════════════════════════╗")
    print("  ║  Smart Hospital Monitor v2.0  — RUNNING   ║")
    print("  ╠═══════════════════════════════════════════╣")
    print("  ║  S=Saline ROI    C=Calibrate              ║")
    print("  ║  Z=Set bed zone  R=Reset fall BG          ║")
    print("  ║  P=Snapshot      L=Log strip              ║")
    print("  ║  F=Fullscreen    Q=Quit                   ║")
    print("  ╚═══════════════════════════════════════════╝\n")

    if saline.calib is not None:
        print("  ✓ Saline calibration auto-loaded.")
    else:
        print("  *** First run: Press S → draw box → S → C to calibrate ***\n")

    fps_disp = 0.0; fps_n = 0; fps_t = time.time()
    latency  = 0.0
    show_log = False
    target_vid_w = CONFIG["window_width"] - PANEL_W

    LOGGER.log("SYSTEM_START","INFO","Monitor v2.0 started")

    while True:
        # ── Grab frame ──────────────────────────────────────────────
        t_frame = time.time()
        if use_esp:
            frame = esp.read_frame()
            if frame is None: time.sleep(0.005); continue
            latency = esp._latency
        else:
            cap.grab()
            ret,frame = cap.retrieve()
            if not ret: break

        # ── Scale to panel area ─────────────────────────────────────
        fh,fw = frame.shape[:2]
        scale  = target_vid_w/fw
        new_h  = int(fh*scale)
        frame  = cv2.resize(frame,(target_vid_w,new_h),
                            interpolation=cv2.INTER_LINEAR)

        # ── Run detectors ───────────────────────────────────────────
        frame, ss = saline.detect(frame)
        frame, fs = fall.detect(frame)
        frame, ms = nomove.detect(frame)
        frame, zs = zone.detect(frame, fall._last_box)

        # ── Build canvas ────────────────────────────────────────────
        log_h    = LOG_H if show_log else 0
        canvas_h = max(new_h + log_h, 600)
        canvas   = np.zeros((canvas_h, target_vid_w+PANEL_W, 3), np.uint8)
        canvas[:new_h, :target_vid_w] = frame

        # ── FPS ─────────────────────────────────────────────────────
        fps_n += 1
        if fps_n >= 20:
            fps_disp = fps_n/(time.time()-fps_t)
            fps_t=time.time(); fps_n=0

        # ── Top status bar ──────────────────────────────────────────
        statuses = [
            ("Saline", ss,
             {"OK":GREEN,"LOW":ORANGE,"EMPTY!":RED,"NOT SET":DIM,
              "NEED CALIB":CYAN,"SETUP MODE":CYAN}.get(ss,WHITE)),
            ("Fall",  fs, RED if fs=="FALL DETECTED!" else GREEN),
            ("Motion",ms, RED if ms=="NO MOVEMENT!"   else GREEN),
            ("Zone",  zs, RED if "OUT" in zs          else GREEN),
        ]
        draw_topbar(canvas, statuses, target_vid_w)

        # ── Right panel ──────────────────────────────────────────────
        stream_fps = esp._fps if use_esp else fps_disp
        draw_panel(canvas, saline, fall, nomove, zone,
                   stream_fps, latency, target_vid_w, show_log)

        # ── Log strip ────────────────────────────────────────────────
        if show_log:
            draw_log_strip(canvas, target_vid_w, canvas_h, log_h)

        # ── Divider ──────────────────────────────────────────────────
        cv2.line(canvas,(target_vid_w,0),(target_vid_w,canvas_h),(50,50,65),1)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF

        if   key == ord('q'): break
        elif key == ord('s'):
            if not saline._setup_mode: saline.enter_setup()
            else: saline.exit_setup(frame)
        elif key == ord('c'): saline.calibrate(frame)
        elif key == ord('r'): fall.reset_bg()
        elif key == ord('f'):
            cur = cv2.getWindowProperty(WIN,cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WIN,cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if cur==0 else cv2.WINDOW_NORMAL)
        elif key == ord('z'):
            zone.enter_setup()
        elif key == ord('l'):
            show_log = not show_log
        elif key == ord('p'):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(CONFIG["snapshot_dir"],f"snapshot_{ts}.jpg")
            cv2.imwrite(path, canvas)
            LOGGER.log("SNAPSHOT","INFO",path)
            print(f"  ✓ Snapshot saved: {path}")

    LOGGER.log("SYSTEM_STOP","INFO","Monitor stopped by user")
    if esp: esp.stop()
    if cap: cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Monitor stopped. Events saved to {CONFIG['log_file']}")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smart Hospital Patient Monitor v2.0")
    ap.add_argument("--url",        default=CONFIG["esp32_stream_url"])
    ap.add_argument("--cam",        type=int, default=None)
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()
    if args.fullscreen: CONFIG["fullscreen"] = True
    run(args.cam if args.cam is not None else args.url)
