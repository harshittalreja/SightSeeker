"""
SightSeeker - Vision Brain: real YOLO perception -> autonomous navigation decisions
==================================================================================

The webcam is the robot's eye. Every frame goes through the same Sense -> Think -> Act loop
the robot uses:

  SENSE  A YOLO detector (YOLO26 by default, 80 COCO classes; or YOLOE for any object) finds boxes + confidence.
         On a webcam it runs on a background thread, so the video stays smooth even on a CPU laptop.
  THINK  A goal given in plain English ("find the bottle") is grounded to a COCO class.
         A finite-state machine decides: SEARCHING -> NAVIGATING -> AVOIDING -> GOAL_REACHED.
         * target bearing  = horizontal offset of the target box from the image centre
         * target distance = how much of the frame the box fills (bigger = closer)
         * obstacle        = any other large object inside the robot's path corridor
  ACT    Differential-drive motor commands (left/right wheel, -255..255) from a proportional
         steering controller. Optionally streamed over USB serial to an Arduino + L298N rover
         (see hardware/sightseeker_rover/sightseeker_rover.ino).

Run:
  python yolo_detection.py                         # webcam, goal = bottle
  python yolo_detection.py --goal "find my phone"  # natural-language goal
  python yolo_detection.py --video clip.mp4 --save demo_out.mp4
  python yolo_detection.py --image photo.jpg
  python yolo_detection.py --serial COM5           # also drive a real rover
  python yolo_detection.py --max                   # MAXIMUM mode: biggest model + images, Full-HD camera, no caps, all cores
  python yolo_detection.py --model x --imgsz 960   # big model only (keeps the heat limits); default: YOLO26l on GPU, YOLO26n on CPU
  python yolo_detection.py --model s               # pick a size yourself: n / s / m / l / x (bigger = more accurate, slower)
  python yolo_detection.py --device cpu            # force CPU (default: NVIDIA GPU if PyTorch can use it)
  python yolo_detection.py --ai-fps 10             # fewer detections per second = cooler laptop (default 15)
  python yolo_detection.py --open-vocab --goal "find the red toolbox"   # any object, via YOLOE text prompts
  python yolo_detection.py --model runs/detect/sightseeker/weights/best.pt   # your fine-tuned model

Keys (in the video window):
  T  type a new goal      V  say a goal (offline voice)      G  cycle demo goals      C  calibrate distance
  R  start/stop recording to recordings/      N  navigation overlay + top-down view      A  all classes / indoor set
  P  presenter mode (people and desks aren't obstacles)   SPACE pause   S screenshot   Q/ESC quit
"""

import argparse
import math
import os
import re
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_NAME = "SightSeeker - Vision Brain"
DISPLAY_H = 720          # camera image is scaled to this height for display
PANEL_W = 400            # width of the decision panel on the right

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")

# Colours (BGR)
C_GREEN = (80, 200, 60)
C_RED = (60, 60, 230)
C_YELLOW = (0, 200, 240)
C_TEAL = (200, 170, 0)
C_WHITE = (245, 245, 245)
C_DIM = (170, 170, 170)
C_PANEL = (30, 27, 24)

STATE_COLORS = {
    "SEARCHING": C_YELLOW,
    "NAVIGATING": C_GREEN,
    "AVOIDING": C_RED,
    "GOAL_REACHED": C_TEAL,
}

MODEL_LABEL = ["yolo"]    # set at start-up, shown in the panel footer and window title
MAX_MODE = [False]        # --max: shown in the panel footer
DEVICE_LABEL = ["CPU"]    # "GPU: <name>", "CPU" or "CPU (OpenVINO)"
INFER_KW = {}             # extra arguments for every model call, e.g. {"device": 0, "half": True} on a GPU
DEMO_GOALS = ["bottle", "cup", "cell phone", "person", "book", "laptop", "backpack", "chair"]

# ----------------------------------------------------------------------------------------
# Language grounding: free text -> COCO class name
# ----------------------------------------------------------------------------------------
SYNONYMS = {
    "person": ["person", "people", "human", "man", "woman", "boy", "girl", "someone", "me", "guy", "friend"],
    "cell phone": ["phone", "mobile", "smartphone", "cellphone", "iphone", "android"],
    "cup": ["cup", "mug", "coffee", "tea", "glass"],
    "bottle": ["bottle", "water", "flask", "sipper"],
    "laptop": ["laptop", "computer", "notebook", "macbook", "pc"],
    "book": ["book", "novel", "textbook", "diary"],
    "chair": ["chair", "seat", "stool"],
    "backpack": ["backpack", "bag", "rucksack", "schoolbag"],
    "handbag": ["handbag", "purse"],
    "potted plant": ["plant", "flower", "pot"],
    "dining table": ["table", "desk"],
    "tv": ["tv", "television", "monitor", "screen"],
    "couch": ["couch", "sofa"],
    "sports ball": ["ball", "football", "basketball"],
    "teddy bear": ["teddy", "toy", "plush"],
    "bicycle": ["bicycle", "cycle", "bike"],
    "clock": ["clock"],
    "scissors": ["scissors", "scissor"],
    "remote": ["remote"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "umbrella": ["umbrella"],
    "suitcase": ["suitcase", "luggage", "trolley"],
}


def _close_word(a, b):
    """True if a == b, a plural of b, or one edit/transposition apart (for words >= 4 chars)."""
    if a == b or a == b + "s" or a == b + "es":
        return True
    if len(a) < 4 or len(b) < 4 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long_)):
        if long_[:i] + long_[i + 1:] == short:
            return True
    return False


def ground_goal(text, class_names):
    """Map a natural-language goal to one of the detector's class names. Returns (class, matched) or (None, None)."""
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())) + " "
    # 1) a class name appears verbatim. English puts the head noun last ("the orange cup" -> cup),
    #    so prefer the right-most match, then the longest ("cell phone" over "phone").
    hits = []
    for name in class_names:
        key = re.sub(r"[_\-]+", " ", str(name).lower())    # custom datasets often use "red_barrel"
        for form in (f" {key} ", f" {key}s "):
            i = t.rfind(form)
            if i >= 0:
                hits.append((i + len(form), len(key), name))
    if hits:
        name = max(hits)[2]
        return name, name
    # 2) synonyms, then typos of class names - scanning from the right for the same reason
    toks = t.split()[::-1]
    for tok in toks:
        for cls, words in SYNONYMS.items():
            if cls in class_names and any(_close_word(tok, w) for w in words):
                return cls, tok
    for tok in toks:
        for name in class_names:
            key = re.sub(r"[_\-]+", " ", str(name).lower())
            if " " not in key and _close_word(tok, key):
                return name, tok
    return None, None


# ----------------------------------------------------------------------------------------
# Decision making
# ----------------------------------------------------------------------------------------
def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# Path corridor: the strip of floor the robot will drive over in the next metre or two. Seen in
# perspective it is a trapezoid: 28% of the image wide where it starts (45% down), 56% wide at the bottom.
CORRIDOR_TOP = 0.45
CORRIDOR_HALF_TOP, CORRIDOR_HALF_BOTTOM = 0.14, 0.28
# Things that objects stand on. With a webcam on a desk, the desk itself is detected as "dining table"
# across the bottom of the picture and would block forever; a target sitting on one is reachable.
SURFACE_CLASSES = {"dining table", "bed", "couch"}


def corridor_x(y_frac):
    """Left/right edges of the corridor (fractions of the image width) at height y_frac (0 = top, 1 = bottom)."""
    t = min(1.0, max(0.0, (y_frac - CORRIDOR_TOP) / (1.0 - CORRIDOR_TOP)))
    half = CORRIDOR_HALF_TOP + t * (CORRIDOR_HALF_BOTTOM - CORRIDOR_HALF_TOP)
    return 0.5 - half, 0.5 + half


# ----------------------------------------------------------------------------------------
# Distance from one camera (pinhole model): distance = focal length (px) x real size / size in pixels.
# Typical real sizes of the LONG side of each object (cm), so it works whichever way it is held.
# It's an estimate (about +-20-30%); press C once with the target at 50 cm to calibrate the camera.
# ----------------------------------------------------------------------------------------
REAL_SIZE_CM = {
    "person": 170, "bottle": 24, "cup": 10, "wine glass": 18, "cell phone": 15, "book": 24, "laptop": 33,
    "keyboard": 44, "mouse": 11, "remote": 18, "scissors": 18, "chair": 90, "couch": 200, "potted plant": 40,
    "tv": 80, "backpack": 45, "handbag": 30, "suitcase": 65, "umbrella": 90, "teddy bear": 30, "sports ball": 22,
    "clock": 30, "vase": 25, "bowl": 16, "banana": 18, "apple": 8, "orange": 8, "spoon": 15, "fork": 18,
    "knife": 20, "bicycle": 170, "dog": 60, "cat": 45, "dining table": 150, "bed": 200,
}
CAMERA = {"hfov": 70.0, "f_norm": None}     # f_norm = focal length / image width, set by calibration
CALIB_FILE = "camera_calibration.json"


def focal_px(w):
    if CAMERA["f_norm"]:
        return CAMERA["f_norm"] * w
    return (w / 2.0) / math.tan(math.radians(CAMERA["hfov"]) / 2.0)


def long_side_px(box, w, h):
    """Length of the box's long side, or None if the object is cut off by the picture edge along it."""
    x1, y1, x2, y2 = box
    m = 0.01 * max(w, h)
    if (y2 - y1) >= (x2 - x1):
        return None if (y1 <= m or y2 >= h - m) else (y2 - y1)
    return None if (x1 <= m or x2 >= w - m) else (x2 - x1)


def estimate_cm(cls, box, w, h):
    size = REAL_SIZE_CM.get(cls)
    px = long_side_px(box, w, h) if size else None
    if not px or px < 4:
        return None
    return focal_px(w) * size / px


def load_calibration():
    try:
        import json
        with open(os.path.join(SCRIPT_DIR, CALIB_FILE)) as f:
            CAMERA["f_norm"] = float(json.load(f)["f_norm"])
        return True
    except Exception:  # noqa: BLE001
        return False


def save_calibration(f_norm):
    import json
    CAMERA["f_norm"] = f_norm
    with open(os.path.join(SCRIPT_DIR, CALIB_FILE), "w") as f:
        json.dump({"f_norm": f_norm, "saved": time.strftime("%Y-%m-%d %H:%M")}, f)


class NavBrain:
    """Turns a list of detections into a navigation state + motor command."""

    def __init__(self, goal, hfov_deg=70.0, stop_at=0.60, presenter_mode=True):
        self.goal = goal
        self.hfov = hfov_deg
        self.stop_at = stop_at                  # target fills this fraction of the frame -> arrived
        self.presenter_mode = presenter_mode    # demo mode: the presenter and the desk aren't obstacles
        self.state = "SEARCHING"
        self.track = None                       # smoothed target track
        self.last_seen = -1e9
        self.last_side = 1                      # +1 = target last seen on the right
        self.motor = np.zeros(2)                # smoothed (left, right) in [-1, 1]
        self.events = deque(maxlen=7)
        self.goal_time = None
        self.track_since = None
        self.solid_arrival = False
        self.reset_stats()
        self.log(f"Goal set: {goal}")

    def reset_stats(self):
        """Per-mission numbers for the summary card shown when the goal is reached."""
        self.start_t = None
        self.n_avoid = 0
        self.n_turns = 0
        self.last_turn_cmd = None
        self.summary = None
        self.summary_t = -1e9

    def log(self, text):
        self.events.append((time.strftime("%H:%M:%S"), text))

    def set_goal(self, goal):
        self.goal = goal
        self.track = None
        self.last_seen = -1e9
        self.state = "SEARCHING"
        self.goal_time = None
        self.track_since = None
        self.solid_arrival = False
        self.reset_stats()
        self.log(f"Goal set: {goal}")

    def update(self, dets, w, h, now):
        if self.start_t is None:
            self.start_t = now
        # ---------- pick / track the target ----------
        targets = [d for d in dets if d["cls"] == self.goal and d["conf"] >= 0.35]
        tgt = None
        if targets:
            if self.track is not None:
                cand = max(targets, key=lambda d: box_iou(d["box"], self.track["box"]))
                if box_iou(cand["box"], self.track["box"]) > 0.2:
                    tgt = cand
            if tgt is None:
                tgt = max(targets, key=lambda d: d["conf"] * math.sqrt(max(1.0, (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))))
        if tgt is not None:
            x1, y1, x2, y2 = tgt["box"]
            cx = (x1 + x2) / 2.0
            close = max((y2 - y1) / h, (x2 - x1) / w)
            dist = estimate_cm(tgt["cls"], tgt["box"], w, h)
            if self.track is None:
                self.track = {"cx": cx, "close": close, "dist": dist}
                self.track_since = now
                self.log(f"Target acquired: {self.goal} ({tgt['conf']:.2f})"
                         + (f" ~{dist:.0f} cm" if dist else ""))
            else:
                self.track["cx"] += 0.5 * (cx - self.track["cx"])
                self.track["close"] += 0.5 * (close - self.track["close"])
                if dist is None:
                    self.track["dist"] = None
                elif self.track.get("dist") is None:
                    self.track["dist"] = dist
                else:
                    self.track["dist"] += 0.5 * (dist - self.track["dist"])
            self.track.update(box=tgt["box"], conf=tgt["conf"])
            self.last_seen = now
            self.last_side = 1 if self.track["cx"] >= w / 2 else -1
        elif self.track is not None and now - self.last_seen > 0.7:
            self.track = None
            if self.state == "GOAL_REACHED" and self.solid_arrival:
                self.log("Holding at goal (target out of view)")
            else:
                self.log(f"Lost {self.goal} -> searching {'right' if self.last_side > 0 else 'left'}")

        # ---------- obstacles in the path corridor ----------
        obstacles = []
        for d in dets:
            if d is tgt or d["cls"] == self.goal or d["conf"] < 0.40:
                continue
            if self.presenter_mode and d["cls"] == "person":
                continue
            if d["cls"] in SURFACE_CLASSES:
                if self.presenter_mode:
                    continue
                if tgt is not None:                 # the target stands on this surface: it's where we're going
                    tcx, tbottom = (tgt["box"][0] + tgt["box"][2]) / 2.0, tgt["box"][3]
                    if d["box"][0] <= tcx <= d["box"][2] and d["box"][1] - 0.05 * h <= tbottom <= d["box"][3]:
                        continue
            x1, y1, x2, y2 = d["box"]
            if y2 < CORRIDOR_TOP * h:              # entirely above the corridor = still far away
                continue
            lf, rf = corridor_x(min(y2, h) / h)    # corridor width where the object touches the floor
            overlap = max(0.0, min(x2, rf * w) - max(x1, lf * w))
            if overlap < min(0.25 * (rf - lf) * w, 0.5 * (x2 - x1)):   # a narrow object mostly in the path counts too
                continue
            close = max((y2 - y1) / h, (x2 - x1) / w)
            if close < 0.30:
                continue
            obstacles.append({"det": d, "close": close})
        obstacles.sort(key=lambda o: -o["close"])

        # ---------- state machine ----------
        prev = self.state
        tclose = self.track["close"] if self.track else 0.0
        if prev == "GOAL_REACHED" and tgt is None and self.solid_arrival:
            # Arrived. Up close the target overflows the camera and stops being detected: stay stopped
            # until it is seen again further away, or a new goal is given.
            state = "GOAL_REACHED"
        elif self.track is not None and (tclose >= self.stop_at or (prev == "GOAL_REACHED" and tclose >= 0.85 * self.stop_at)):
            state = "GOAL_REACHED"
        elif obstacles and (self.track is None or obstacles[0]["close"] > tclose):
            state = "AVOIDING"
        elif self.track is not None:
            state = "NAVIGATING"
        else:
            state = "SEARCHING"

        # ---------- motor command (differential drive) ----------
        err = (self.track["cx"] - w / 2.0) / (w / 2.0) if self.track is not None else 0.0
        if state == "SEARCHING":
            side = self.last_side
            left, right = 0.35 * side, -0.35 * side
            cmd = f"ROTATE {'RIGHT' if side > 0 else 'LEFT'} (scanning)"
        elif state == "NAVIGATING":
            base = 0.65 * float(np.clip(1.0 - tclose / self.stop_at, 0.25, 1.0))
            turn = 0.55 * err
            left, right = base + turn, base - turn
            cmd = "FORWARD" if abs(err) < 0.15 else ("TURN RIGHT" if err > 0 else "TURN LEFT")
        elif state == "AVOIDING":
            ob = obstacles[0]["det"]
            ocx = (ob["box"][0] + ob["box"][2]) / 2.0
            turn_dir = -1 if ocx >= w / 2.0 else 1          # obstacle on the right -> veer left
            left, right = 0.12 + 0.45 * turn_dir, 0.12 - 0.45 * turn_dir
            cmd = f"AVOID {ob['cls'].upper()} -> {'RIGHT' if turn_dir > 0 else 'LEFT'}"
        else:
            left = right = 0.0
            cmd = "STOP - GOAL REACHED"
        target = np.clip(np.array([left, right]), -1.0, 1.0)
        self.motor = 0.55 * self.motor + 0.45 * target
        if state == "GOAL_REACHED":
            self.motor = np.zeros(2)

        kind = cmd.split()[0] + (cmd.split()[1] if cmd.startswith("TURN") else "")
        if cmd.startswith("TURN") and kind != self.last_turn_cmd:
            self.n_turns += 1                      # a course correction toward the target
        self.last_turn_cmd = kind if cmd.startswith("TURN") else None

        if state != prev:
            if state == "AVOIDING":
                self.n_avoid += 1
            if state == "GOAL_REACHED":
                self.summary = {"goal": self.goal, "time": now - (self.start_t or now), "turns": self.n_turns,
                                "avoided": self.n_avoid,
                                "dist": self.track.get("dist") if self.track else None}
                self.summary_t = now
                self.goal_time = now
                # Hold the stop even if the target later overflows the view - but only for a real arrival
                # (target tracked for at least half a second), not a two-frame false detection.
                self.solid_arrival = self.track_since is not None and now - self.track_since >= 0.5
                self.log(f"GOAL REACHED: {self.goal} -> stop")
            elif state == "AVOIDING":
                self.log(f"Obstacle in path: {obstacles[0]['det']['cls']} -> {cmd.split('-> ')[-1].lower()}")
            elif state == "NAVIGATING" and prev != "SEARCHING":
                self.log(f"Path clear -> heading to {self.goal}")
        self.state = state

        return {
            "state": state,
            "cmd": cmd,
            "err": err,
            "bearing": err * self.hfov / 2.0,
            "close": tclose,
            "dist": self.track.get("dist") if self.track else None,
            "target": tgt,
            "obstacles": obstacles,
            "pwm": (int(round(self.motor[0] * 255)), int(round(self.motor[1] * 255))),
        }


# ----------------------------------------------------------------------------------------
# Optional hardware bridge
# ----------------------------------------------------------------------------------------
class SerialBridge:
    """Streams 'L<int>,R<int>\\n' (-255..255) to an Arduino rover at <= 20 Hz."""

    def __init__(self, port, baud=115200):
        self.ok = False
        self.port = port
        self.last = 0.0
        try:
            import serial  # pyserial
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(2.0)  # most Arduinos reset when the port opens
            self.ok = True
            print(f"[serial] Connected to {port} @ {baud}")
        except Exception as e:  # noqa: BLE001
            print(f"[serial] Could not open {port}: {e}\n[serial] Continuing without hardware.")

    def send(self, left, right, force=False):
        if not self.ok:
            return
        now = time.time()
        if not force and now - self.last < 0.05:
            return
        self.last = now
        try:
            self.ser.write(f"L{left},R{right}\n".encode("ascii"))
        except Exception as e:  # noqa: BLE001
            print(f"[serial] Write failed ({e}); hardware output disabled.")
            self.ok = False

    def close(self):
        if self.ok:
            self.send(0, 0, force=True)
            try:
                self.ser.close()
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------------------
def draw_transparent_rect(img, top_left, bottom_right, color, alpha=0.5):
    x1, y1 = max(0, int(top_left[0])), max(0, int(top_left[1]))
    x2, y2 = min(img.shape[1], int(bottom_right[0])), min(img.shape[0], int(bottom_right[1]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def put_text_with_background(img, text, position, font_scale, font_thickness, text_color, bg_color, alpha=0.75):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = int(position[0]), int(position[1])
    x = max(0, min(x, img.shape[1] - tw - 6))
    y = max(th + 6, y)
    draw_transparent_rect(img, (x, y - th - 5), (x + tw + 6, y + baseline + 3), bg_color, alpha)
    cv2.putText(img, text, (x + 3, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


def draw_detections(disp, dets, decision, scale, nav_overlay):
    target = decision["target"]
    obstacle_ids = {id(o["det"]) for o in decision["obstacles"]}
    for d in dets:
        x1, y1, x2, y2 = [int(v * scale) for v in d["box"]]
        if d is target:
            dist = decision.get("dist")
            color, thick, label = C_GREEN, 3, f"TARGET {d['cls']} {d['conf']:.2f}" + (f"  ~{dist:.0f} cm" if dist else "")
        elif nav_overlay and id(d) in obstacle_ids:
            color, thick, label = C_RED, 3, f"OBSTACLE {d['cls']} {d['conf']:.2f}"
        else:
            color = [int(c) for c in COLORS[d["cls_id"] % len(COLORS)]]
            thick, label = 2, f"{d['cls']} {d['conf']:.2f}"
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, thick)
        put_text_with_background(disp, label, (x1, y1 - 6), 0.55, 1, (255, 255, 255), color)


def draw_corridor(disp, decision):
    h, w = disp.shape[:2]
    blocked = decision["state"] == "AVOIDING"
    col = C_RED if blocked else C_TEAL
    (bl, br), (tl, tr) = corridor_x(1.0), corridor_x(CORRIDOR_TOP)
    pts = np.array([[bl * w, h - 1], [br * w, h - 1], [tr * w, CORRIDOR_TOP * h], [tl * w, CORRIDOR_TOP * h]], np.int32)
    overlay = disp.copy()
    cv2.fillPoly(overlay, [pts], col)
    cv2.addWeighted(overlay, 0.16, disp, 0.84, 0, disp)
    cv2.polylines(disp, [pts], True, col, 1, cv2.LINE_AA)


def draw_steering(disp, decision, scale):
    h, w = disp.shape[:2]
    base = (w // 2, h - 12)
    tgt = decision["target"]
    if tgt is not None:
        tx = int((tgt["box"][0] + tgt["box"][2]) / 2 * scale)
        ty = int((tgt["box"][1] + tgt["box"][3]) / 2 * scale)
        cv2.line(disp, base, (tx, ty), C_GREEN, 2, cv2.LINE_AA)
        cv2.circle(disp, (tx, ty), 6, C_GREEN, -1, cv2.LINE_AA)

    # Steering arrow from the robot's point of view
    st = decision["state"]
    if st == "GOAL_REACHED":
        c = (base[0], base[1] - 46)
        cv2.circle(disp, c, 40, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(disp, c, 40, C_TEAL, 4, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(disp, "STOP", (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_DUPLEX, 0.75, C_TEAL, 2, cv2.LINE_AA)
    else:
        left, right = decision["pwm"]
        ang = math.atan2((left - right) / 255.0, max(0.05, (left + right) / 510.0 + 0.25))
        L = 90
        tip = (int(base[0] + L * math.sin(ang)), int(base[1] - L * math.cos(ang)))
        cv2.arrowedLine(disp, base, tip, (20, 20, 20), 11, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(disp, base, tip, STATE_COLORS.get(st, C_WHITE), 6, cv2.LINE_AA, tipLength=0.3)


def draw_bar(img, x, y, w, h, value, label):
    """Signed bar for a motor value in -255..255."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 66, 62), -1)
    mid = x + w // 2
    end = int(mid + (value / 255.0) * (w // 2))
    col = C_GREEN if value >= 0 else C_RED
    cv2.rectangle(img, (min(mid, end), y), (max(mid, end), y + h), col, -1)
    cv2.line(img, (mid, y - 2), (mid, y + h + 2), C_DIM, 1)
    cv2.putText(img, f"{label} {value:+4d}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)


def build_panel(height, brain, decision, dets, fps, infer_ms, serial, nav_overlay, typing_text, message, ai_fps=None):
    p = np.full((height, PANEL_W, 3), C_PANEL, np.uint8)
    x = 18
    y = 38
    cv2.putText(p, "SightSeeker", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (235, 200, 60), 2, cv2.LINE_AA)
    y += 24
    cv2.putText(p, "Vision Brain - YOLO + state machine", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DIM, 1, cv2.LINE_AA)
    y += 30
    cv2.putText(p, "GOAL", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(p, brain.goal, (x + 60, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, C_WHITE, 1, cv2.LINE_AA)
    y += 34
    st = decision["state"]
    col = STATE_COLORS.get(st, C_WHITE)
    cv2.rectangle(p, (x, y - 24), (PANEL_W - 18, y + 10), col, 2)
    cv2.putText(p, st.replace("_", " "), (x + 10, y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2, cv2.LINE_AA)
    y += 42
    cv2.putText(p, "COMMAND", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
    y += 26
    cv2.putText(p, decision["cmd"][:30], (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)
    y += 40
    left, right = decision["pwm"]
    draw_bar(p, x, y, PANEL_W - 36, 12, left, "LEFT WHEEL ")
    y += 40
    draw_bar(p, x, y, PANEL_W - 36, 12, right, "RIGHT WHEEL")
    y += 34
    if decision["target"] is not None:
        t = decision["target"]
        dist = decision.get("dist")
        dtxt = (f"~{dist:.0f} cm" + (" (calibrated)" if CAMERA["f_norm"] else " (estimate)")) if dist else "very close / cut off"
        lines = [
            f"Target conf   {t['conf']:.2f}",
            f"Bearing       {decision['bearing']:+5.1f} deg",
            f"Distance      {dtxt}",
            f"Fills view    {decision['close'] * 100:4.0f}%  (stop at {brain.stop_at * 100:.0f}%)",
        ]
    else:
        lines = ["Target        not in view", "Bearing       --", "Distance      --", "Fills view    --"]
    for ln in lines:
        cv2.putText(p, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)
        y += 21
    y += 8
    classes = sorted({d["cls"] for d in dets})
    cv2.putText(p, f"OBJECTS IN VIEW: {len(dets)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
    y += 20
    txt = ", ".join(classes) if classes else "-"
    while txt and y < height - 210:
        cut = txt[:40]
        if len(txt) > 40 and "," in cut:
            cut = cut[:cut.rfind(",") + 1]
        cv2.putText(p, cut.strip(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, C_WHITE, 1, cv2.LINE_AA)
        txt = txt[len(cut):].strip()
        y += 19
    y = max(y + 10, height - 215)
    cv2.putText(p, "DECISION LOG", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
    y += 20
    for ts, ev in list(brain.events)[-5:]:
        cv2.putText(p, f"{ts} {ev}"[:44], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        y += 18
    # footer
    cv2.putText(p, f"model: {MODEL_LABEL[0]} | {DEVICE_LABEL[0]}"[:52], (x, height - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    speed = f"video {fps:4.1f} FPS | AI {ai_fps:4.1f} FPS" if ai_fps is not None else f"{fps:4.1f} FPS"
    flags = [speed, f"{infer_ms:4.0f} ms", "presenter ON" if brain.presenter_mode else "people = obstacles"]
    if MAX_MODE[0]:
        flags[2] = "MAX MODE"
    cv2.putText(p, " | ".join(flags), (x, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    ser = "serial: off" if serial is None else (f"serial: {serial.port} OK" if serial.ok else f"serial: {serial.port} FAILED")
    cv2.putText(p, ser + ("" if nav_overlay else " | nav overlay OFF"), (x, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    return p


def draw_radar(disp, frame_w, frame_h, dets, decision):
    """Top-down mini-map of what the camera sees: robot at the bottom, camera field of view as a fan,
    the target (green) and obstacles (red) placed by bearing and estimated distance."""
    W = disp.shape[1]
    cw, ch = 230, 168
    x0, y0 = W - cw - 12, 12
    draw_transparent_rect(disp, (x0, y0), (x0 + cw, y0 + ch), (20, 20, 20), 0.62)
    cv2.putText(disp, "TOP-DOWN VIEW", (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    ax, ay = x0 + cw // 2, y0 + ch - 12               # robot position
    R = ch - 40
    max_cm = 300.0
    half = math.radians(CAMERA["hfov"] / 2.0)
    for frac, lab in ((1 / 3, "1 m"), (2 / 3, "2 m"), (1.0, "3 m")):
        rr = int(R * frac)
        cv2.ellipse(disp, (ax, ay), (rr, rr), 0, -90 - math.degrees(half), -90 + math.degrees(half), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(disp, lab, (ax + int(rr * math.sin(half)) + 3, ay - int(rr * math.cos(half))), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1, cv2.LINE_AA)
    for sgn in (-1, 1):
        cv2.line(disp, (ax, ay), (ax + int(sgn * R * math.sin(half)), ay - int(R * math.cos(half))), (110, 110, 110), 1, cv2.LINE_AA)
    obstacle_ids = {id(o["det"]) for o in decision["obstacles"]}
    target = decision["target"]
    for d in dets:
        bx = (d["box"][0] + d["box"][2]) / 2.0
        ang = ((bx - frame_w / 2.0) / (frame_w / 2.0)) * half
        dist = decision.get("dist") if d is target else estimate_cm(d["cls"], d["box"], frame_w, frame_h)
        if dist is None:               # unknown size or cut off: place it by how much of the view it fills
            close = max((d["box"][3] - d["box"][1]) / frame_h, (d["box"][2] - d["box"][0]) / frame_w)
            dist = max(15.0, (1.0 - min(close, 0.95)) * max_cm)
        rr = R * min(dist, max_cm) / max_cm
        px, py = ax + int(rr * math.sin(ang)), ay - int(rr * math.cos(ang))
        if d is target:
            cv2.line(disp, (ax, ay), (px, py), C_GREEN, 1, cv2.LINE_AA)
            cv2.circle(disp, (px, py), 7, C_GREEN, -1, cv2.LINE_AA)
        elif id(d) in obstacle_ids:
            cv2.circle(disp, (px, py), 6, C_RED, -1, cv2.LINE_AA)
        else:
            cv2.circle(disp, (px, py), 4, (170, 170, 170), -1, cv2.LINE_AA)
    cv2.circle(disp, (ax, ay), 6, C_TEAL, -1, cv2.LINE_AA)


def draw_summary(disp, summary):
    """Mission summary card shown for a few seconds after GOAL REACHED."""
    H, W = disp.shape[:2]
    lines = [f"GOAL REACHED: {summary['goal']}",
             f"Found in {summary['time']:.1f} s  |  {summary['turns']} course correction{'s' if summary['turns'] != 1 else ''}"
             f"  |  {summary['avoided']} obstacle{'s' if summary['avoided'] != 1 else ''} avoided"]
    if summary.get("dist"):
        lines.append(f"Stopped about {summary['dist']:.0f} cm from the target")
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, 0.85 if i == 0 else 0.6, 2 if i == 0 else 1)[0] for i, t in enumerate(lines)]
    bw = max(sz[0] for sz in sizes) + 40
    bh = 30 + sum(sz[1] + 16 for sz in sizes)
    x0, y0 = (W - bw) // 2, int(H * 0.30)
    draw_transparent_rect(disp, (x0, y0), (x0 + bw, y0 + bh), (25, 60, 20), 0.82)
    cv2.rectangle(disp, (x0, y0), (x0 + bw, y0 + bh), C_GREEN, 2)
    y = y0 + 18
    for i, (t, sz) in enumerate(zip(lines, sizes)):
        y += sz[1] + 8
        cv2.putText(disp, t, ((W - sz[0]) // 2, y), cv2.FONT_HERSHEY_DUPLEX, 0.85 if i == 0 else 0.6,
                    (255, 255, 255), 2 if i == 0 else 1, cv2.LINE_AA)
        y += 8


def fit_text(text, max_w, scale, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    w = cv2.getTextSize(text, font, scale, thick)[0][0]
    return scale if w <= max_w else scale * max_w / w


# ----------------------------------------------------------------------------------------
# Digital twin: a virtual rover driven by exactly the wheel commands the Arduino would get
# ----------------------------------------------------------------------------------------
TWIN_W = 420


class DigitalTwin:
    """Differential-drive model of the SightSeeker rover (2WD chassis, ~16 cm between the wheels, ~0.4 m/s
    top speed). It turns the same left/right wheel commands that go to the Arduino into motion: where the
    rover would head next, and the path it would have driven so far."""
    VMAX, TRACK = 0.40, 0.16          # m/s at full power, metres between the wheels

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = self.th = 0.0               # metres, heading (radians, 0 = start direction, + = right)
        self.trail = [(0.0, 0.0)]
        self.driven = 0.0
        self.phase = [0.0, 0.0]                       # wheel tread animation
        self.last_t = None

    def speeds(self, pwm):
        lw, rw = pwm[0] / 255.0, pwm[1] / 255.0
        v = self.VMAX * (lw + rw) / 2.0
        w = self.VMAX * (lw - rw) / self.TRACK         # + = turning right
        return v, w, lw, rw

    def update(self, decision, now):
        if decision is None:
            return
        dt = 0.0 if self.last_t is None else min(0.2, max(0.0, now - self.last_t))
        self.last_t = now
        v, w, lw, rw = self.speeds(decision["pwm"])
        self.th += w * dt
        self.x += v * math.sin(self.th) * dt
        self.y += v * math.cos(self.th) * dt
        self.driven += abs(v) * dt
        self.phase[0] = (self.phase[0] + lw * dt * 60) % 12
        self.phase[1] = (self.phase[1] + rw * dt * 60) % 12
        lx, ly = self.trail[-1]
        if math.hypot(self.x - lx, self.y - ly) > 0.02:
            self.trail.append((self.x, self.y))
            self.trail = self.trail[-600:]

    def predict(self, pwm, seconds=3.0, step=0.05):
        """Where it would go if it kept these commands: up to `seconds`, or until it has turned 90 degrees."""
        v, w, _, _ = self.speeds(pwm)
        x = y = th = 0.0
        pts = [(0.0, 0.0, 0.0)]
        for _ in range(int(seconds / step)):
            if abs(th) > math.pi / 2:
                break
            th += w * step
            x += v * math.sin(th) * step
            y += v * math.cos(th) * step
            pts.append((x, y, th))
        return pts


def draw_rover(img, cx, cy, th, scale_px, color, phase=None, ghost=False):
    """Top view of the 2WD rover (body + two wheels) at (cx, cy), heading th (0 = up, + = right)."""
    c, s_ = math.cos(th), math.sin(th)
    def pt(dx, dy):                                   # rover frame (dx right, dy forward, metres) -> image
        return (int(cx + (dx * c + dy * s_) * scale_px), int(cy - (-dx * s_ + dy * c) * scale_px))
    body = np.array([pt(-0.07, -0.09), pt(0.07, -0.09), pt(0.07, 0.09), pt(-0.07, 0.09)], np.int32)
    if ghost:
        cv2.polylines(img, [body], True, color, 1, cv2.LINE_AA)
        return
    cv2.fillPoly(img, [body], color, cv2.LINE_AA)
    for side in (-1, 1):
        wheel = np.array([pt(side * 0.095 - 0.015, -0.035), pt(side * 0.095 + 0.015, -0.035),
                          pt(side * 0.095 + 0.015, 0.035), pt(side * 0.095 - 0.015, 0.035)], np.int32)
        cv2.fillPoly(img, [wheel], (40, 40, 40), cv2.LINE_AA)
    cv2.circle(img, pt(0, 0.075), max(2, int(0.02 * scale_px)), (30, 30, 30), -1, cv2.LINE_AA)   # the camera


def build_twin_panel(height, twin, frame_w, frame_h, dets, decision, serial, now):
    p = np.full((height, TWIN_W, 3), (24, 22, 20), np.uint8)
    cv2.putText(p, "DIGITAL TWIN", (16, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, C_TEAL, 1, cv2.LINE_AA)
    cv2.putText(p, "a virtual rover driven by the same wheel commands", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
    if decision is None:
        return p
    # ---- what the rover sees and where it heads next (rover frame, 1 m = 100 px) ----
    x0, y0, bw, bh = 12, 70, TWIN_W - 24, 340
    cv2.rectangle(p, (x0 - 1, y0 - 1), (x0 + bw, y0 + bh), (60, 56, 52), 1)
    v = p[y0:y0 + bh, x0:x0 + bw]                         # drawing into this slice clips to the box
    S, R = 100.0, 2.4                                     # 1 m = 100 px; the rover icon drawn 2.4x larger
    ax, ay = bw // 2, bh - 44
    for m in range(1, 4):
        cv2.circle(v, (ax, ay), int(m * S), (52, 48, 45), 1, cv2.LINE_AA)
        cv2.putText(v, f"{m} m", (ax + 6, ay - int(m * S) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
    half = math.radians(CAMERA["hfov"] / 2.0)
    for sgn in (-1, 1):
        cv2.line(v, (ax, ay), (ax + int(sgn * 3.4 * S * math.sin(half)), ay - int(3.4 * S * math.cos(half))), (80, 80, 80), 1, cv2.LINE_AA)
    obstacle_ids = {id(o["det"]) for o in decision["obstacles"]}
    target = decision["target"]
    for d in dets:
        bx = (d["box"][0] + d["box"][2]) / 2.0
        ang = ((bx - frame_w / 2.0) / (frame_w / 2.0)) * half
        dist = decision.get("dist") if d is target else estimate_cm(d["cls"], d["box"], frame_w, frame_h)
        if dist is None:
            close = max((d["box"][3] - d["box"][1]) / frame_h, (d["box"][2] - d["box"][0]) / frame_w)
            dist = max(15.0, (1.0 - min(close, 0.95)) * 300.0)
        rr = S * min(dist, 300.0) / 100.0
        px, py = ax + int(rr * math.sin(ang)), ay - int(rr * math.cos(ang))
        if d is target:
            cv2.circle(v, (px, py), 10, C_GREEN, -1, cv2.LINE_AA)
            cv2.putText(v, d["cls"], (px + 13, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREEN, 1, cv2.LINE_AA)
        elif id(d) in obstacle_ids:
            cv2.rectangle(v, (px - 10, py - 10), (px + 10, py + 10), C_RED, -1, cv2.LINE_AA)
            cv2.putText(v, d["cls"], (px + 13, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_RED, 1, cv2.LINE_AA)
        else:
            cv2.circle(v, (px, py), 5, (150, 150, 150), -1, cv2.LINE_AA)
    pts = twin.predict(decision["pwm"])
    path = [(ax + int(x * S), ay - int(y * S)) for x, y, _ in pts]
    for i in range(0, len(path) - 1, 2):
        cv2.line(v, path[i], path[i + 1], C_YELLOW, 3, cv2.LINE_AA)
    if len(pts) > 1:
        k = int((now % 2.0) / 2.0 * (len(pts) - 1))      # a ghost rover runs along the predicted path
        gx, gy, gth = pts[k]
        draw_rover(v, ax + gx * S, ay - gy * S, gth, S * R, C_YELLOW, ghost=True)
    draw_rover(v, ax, ay, 0.0, S * R, C_TEAL)
    cv2.putText(v, "yellow: where these commands take it next", (8, bh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_YELLOW, 1, cv2.LINE_AA)
    # ---- the wheels ----
    left, right = decision["pwm"]
    y = y0 + bh + 18
    for i, (lab, val) in enumerate((("LEFT", left), ("RIGHT", right))):
        wx = 24 + i * 200
        cv2.rectangle(p, (wx, y), (wx + 34, y + 70), (45, 45, 45), -1)
        off = twin.phase[i]
        for k2 in range(-1, 7):
            yy = int(y + k2 * 12 + off) if val >= 0 else int(y + 70 - k2 * 12 - off)
            if y <= yy <= y + 70:
                cv2.line(p, (wx + 2, yy), (wx + 32, yy), (95, 95, 95), 2)
        col = C_GREEN if val > 0 else (C_RED if val < 0 else C_DIM)
        cv2.putText(p, lab, (wx + 46, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DIM, 1, cv2.LINE_AA)
        cv2.putText(p, f"{val:+d}", (wx + 46, y + 56), cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 1, cv2.LINE_AA)
    y += 96
    cv2.putText(p, "Arduino receives:", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(p, f"L{left},R{right}", (180, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)
    link = "connected" if (serial is not None and getattr(serial, "ok", False)) else "no rover connected - twin only"
    cv2.putText(p, f"serial: {link}", (16, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    # ---- the path it would have driven so far (odometry, start direction = up) ----
    y += 40
    mx0, my0, mw, mh = 12, y, TWIN_W - 24, height - y - 14
    if mh > 60:
        cv2.rectangle(p, (mx0, my0), (mx0 + mw, my0 + mh), (60, 56, 52), 1)
        cv2.putText(p, f"PATH SO FAR  {twin.driven:.1f} m driven", (mx0 + 8, my0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_DIM, 1, cv2.LINE_AA)
        xs = [t[0] for t in twin.trail] + [0.0]
        ys = [t[1] for t in twin.trail] + [0.0]
        span = max(1.0, max(xs) - min(xs), (max(ys) - min(ys)) * mw / max(1, mh - 30))
        sc = (mw - 30) / span
        cx0, cy0 = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
        def mp(x, yv):
            return (int(mx0 + mw / 2 + (x - cx0) * sc), int(my0 + 14 + (mh - 14) / 2 - (yv - cy0) * sc))
        tr = [mp(x, yv) for x, yv in twin.trail]
        for i in range(len(tr) - 1):
            cv2.line(p, tr[i], tr[i + 1], C_TEAL, 2, cv2.LINE_AA)
        cv2.circle(p, mp(0.0, 0.0), 4, C_DIM, -1, cv2.LINE_AA)
        draw_rover(p, *mp(twin.x, twin.y), twin.th, max(40.0, min(140.0, sc)), C_TEAL)
    return p

def compose(frame, dets, decision, brain, fps, infer_ms, serial, nav_overlay, typing_text=None, message=None, ai_fps=None,
            overlay=None):
    fh, fw = frame.shape[:2]
    scale = DISPLAY_H / float(fh)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC     # sharp when shrinking HD, smooth when enlarging
    disp = cv2.resize(frame, (int(round(fw * scale)), DISPLAY_H), interpolation=interp)
    if nav_overlay:
        draw_corridor(disp, decision)
    draw_detections(disp, dets, decision, scale, nav_overlay)
    overlay = overlay or {}
    if nav_overlay:
        draw_steering(disp, decision, scale)
        draw_radar(disp, fw, fh, dets, decision)
    top = f"FPS {fps:.1f}" if ai_fps is None else f"FPS {fps:.1f}  |  AI {ai_fps:.1f}/s"
    put_text_with_background(disp, top, (12, 34), 0.7, 2, (255, 255, 255), (0, 0, 0), 0.6)
    if overlay.get("rec"):
        put_text_with_background(disp, "REC", (12 + cv2.getTextSize(top, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] + 22, 34),
                                 0.7, 2, (255, 255, 255), (30, 30, 200), 0.85)
    hint = "T type goal | V voice | G next goal | W walk | D twin | C calibrate | R record | N overlay | A classes | P presenter | SPACE pause | S shot | Q quit"
    put_text_with_background(disp, hint, (12, DISPLAY_H - 14), fit_text(hint, disp.shape[1] - 30, 0.45), 1,
                             (230, 230, 230), (0, 0, 0), 0.55)
    if overlay.get("walk"):
        g = overlay["walk"]
        put_text_with_background(disp, g, (12, DISPLAY_H - 52), fit_text(g, disp.shape[1] - 40, 0.85), 2,
                                 (20, 20, 20), (200, 225, 80), 0.92)
    elif overlay.get("walk_on"):
        put_text_with_background(disp, "WALK MODE: follow the spoken directions", (12, DISPLAY_H - 52), 0.7, 2,
                                 (20, 20, 20), (200, 225, 80), 0.92)
    if brain.summary and 0 <= overlay.get("now", 0) - brain.summary_t < 8 and brain.state == "GOAL_REACHED":
        draw_summary(disp, brain.summary)
    if overlay.get("voice"):
        put_text_with_background(disp, overlay["voice"], (12, 80), 0.8, 2, (255, 255, 255), (130, 40, 120), 0.88)
        message = None
    if typing_text is not None:
        put_text_with_background(disp, f"Goal > {typing_text}_", (12, 80), 0.8, 2, (255, 255, 255), (120, 80, 0), 0.85)
        put_text_with_background(disp, "type e.g. 'find my phone', ENTER = go, ESC = cancel", (12, 112), 0.5, 1, (255, 255, 255), (0, 0, 0), 0.6)
    elif message:
        put_text_with_background(disp, message, (12, 80) if not overlay.get("voice") else (12, 118), 0.6, 1,
                                 (255, 255, 255), (40, 40, 40), 0.8)
    panel = build_panel(DISPLAY_H, brain, decision, dets, fps, infer_ms, serial, nav_overlay, typing_text, message, ai_fps)
    if overlay.get("twin") is not None:
        twin_p = build_twin_panel(DISPLAY_H, overlay["twin"], fw, fh, dets, decision, serial, overlay.get("now", 0.0))
        return np.hstack([disp, panel, twin_p])
    return np.hstack([disp, panel])


# ----------------------------------------------------------------------------------------
# Perception
# ----------------------------------------------------------------------------------------
# Objects you are likely to meet indoors. Restricting to these removes the odd false alarms
# (e.g. "toilet", "refrigerator") a small COCO model throws on a busy webcam scene. Press A to toggle.
INDOOR_CLASSES = {
    "person", "backpack", "umbrella", "handbag", "suitcase", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "orange", "chair", "couch", "potted plant", "bed", "dining table",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "book", "clock", "vase", "scissors",
    "teddy bear", "sports ball", "bicycle", "dog", "cat",
}

# Pretrained COCO checkpoints (Ultralytics' published COCO mAP50-95 / CPU ONNX ms / T4 GPU TensorRT ms):
#   YOLO26n 40.9 / 39 / 1.7   YOLO26s 48.6 / 87 / 2.5   YOLO26m 53.1 / 220 / 4.7
#   YOLO26l 55.0 / 286 / 6.2  YOLO26x 57.5 / 526 / 11.8      (YOLO26 = newest generation)
#   YOLOv8n 37.3 / 80         YOLOv8s 44.9 / 128             (older generation, offline fallback)
MODEL_ALIASES = {"n": "yolo26n.pt", "s": "yolo26s.pt", "m": "yolo26m.pt", "l": "yolo26l.pt", "x": "yolo26x.pt",
                 "v8n": "yolov8n.pt", "v8s": "yolov8s.pt", "v8m": "yolov8m.pt"}
# "auto" on a CPU = YOLO26n (faster AND more accurate than YOLOv8n); the OpenVINO export of it (made by
# launch.bat option 6) runs ~2-3x faster again on Intel CPUs. "auto" on an NVIDIA GPU = YOLO26l.
OPENVINO_DIR = "yolo26n_openvino_model"
OPENVINO_IMGSZ = 480          # the size download_models.py exports it at (OpenVINO models have a fixed size)
AUTO_ORDER = (OPENVINO_DIR, "yolo26n.pt", "yolo26s.pt")
GPU_ORDER = ("yolo26l.pt", "yolo26m.pt", "yolo26s.pt", "yolo26n.pt", "yolo26x.pt")   # offline fallbacks, x last (hot)
OLD_FALLBACK = ("yolov8s.pt", "yolov8n.pt")   # used only if YOLO26 can't be downloaded

# Open-vocabulary mode (YOLOE): the detector is prompted with words, so the goal can be ANY object.
OPEN_VOCAB_WEIGHTS = "yoloe-26s-seg.pt"
OBSTACLE_PROMPTS = ["person", "chair", "table", "box", "bag", "bottle", "cup", "laptop"]
FILLER_WORDS = {"find", "go", "to", "the", "a", "an", "my", "me", "locate", "search", "for", "get", "bring",
                "please", "fetch", "look", "drive", "move", "towards", "toward", "reach", "where", "is", "of", "some"}


def goal_phrase(text):
    """'please find the red barrel' -> 'red barrel' (used as the open-vocabulary prompt)."""
    words = [w for w in re.sub(r"[^a-z ]", " ", text.lower()).split() if w not in FILLER_WORDS]
    return " ".join(words[:4]) or "bottle"


def set_prompts(model, goal):
    names = [goal] + [c for c in OBSTACLE_PROMPTS if c != goal]
    try:
        model.set_classes(names)
    except TypeError:                       # older Ultralytics releases need the text embeddings passed in
        model.set_classes(names, model.get_text_pe(names))
    return names


def load_open_vocab():
    from ultralytics import YOLOE
    local = os.path.join(SCRIPT_DIR, OPEN_VOCAB_WEIGHTS)
    print(f"Loading open-vocabulary model: {OPEN_VOCAB_WEIGHTS}")
    return YOLOE(local if os.path.exists(local) else OPEN_VOCAB_WEIGHTS)


def pick_device(requested="auto"):
    """Returns (device for Ultralytics, label, use_half). Uses an NVIDIA GPU when PyTorch can see one."""
    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu", "CPU", False
    if requested == "intel:gpu":                 # only meaningful for the OpenVINO model (see main)
        return "intel:gpu", "CPU", False
    if requested not in ("auto", "gpu", "cuda", "0", "cuda:0"):
        return requested, requested, False       # e.g. "mps" or "cuda:1" - passed straight to Ultralytics
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            try:      # RTX 50-series cards need a CUDA 12.8+ PyTorch build; older builds fail on the first op
                (torch.ones(8, device="cuda") * 2).sum().item()
                return 0, f"GPU: {name}", True
            except Exception as e:  # noqa: BLE001
                print(f"PyTorch sees {name} but can't run on it ({str(e)[:120]}).\n"
                      "  This PyTorch build is too old for the card. Run launch.bat option 6 (or: python check_gpu.py --install).")
                return "cpu", "CPU", False
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", "GPU: Apple", False
        if torch.version.cuda is None:
            print("Running on the CPU: the installed PyTorch is the CPU-only build.\n"
                  "  Have an NVIDIA GPU? Run launch.bat option 6 (or: python check_gpu.py) to switch to the GPU build.")
        else:
            print("Running on the CPU: PyTorch has CUDA support but can't see an NVIDIA GPU (driver missing or no NVIDIA GPU).")
    except Exception as e:  # noqa: BLE001
        print(f"Couldn't check for a GPU ({e}) - using the CPU.")
    return "cpu", "CPU", False


def fp16_kwargs():
    """Newer Ultralytics replaced half=True with quantize=16; use whichever this version understands."""
    try:
        from ultralytics.utils import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:  # noqa: BLE001
        pass
    return {"half": True}


def switch_to_cpu(model, swap_small=True):
    """Last resort if the GPU fails mid-demo: keep running on the CPU instead of silently detecting nothing.
    Returns the model to use from now on: the small YOLO26n if it's on disk (a big GPU model is far too
    slow on a CPU, and after a driver reset its GPU copy may be unusable), otherwise the same model."""
    INFER_KW["device"] = "cpu"
    INFER_KW.pop("half", None)
    INFER_KW.pop("quantize", None)
    openvino = "OpenVINO" in MODEL_LABEL[0]
    DEVICE_LABEL[0] = "CPU (OpenVINO)" if openvino else "CPU (GPU failed)"
    try:
        import torch
        torch.set_num_threads(4)
    except Exception:  # noqa: BLE001
        pass
    small = _local("yolo26n.pt")
    if swap_small and not openvino and "yolo26n" not in MODEL_LABEL[0] and os.path.exists(small):
        try:
            from ultralytics import YOLO
            new = YOLO(small)
            MODEL_LABEL[0] = "yolo26n"
            return new
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't load {small} ({e}) - keeping {MODEL_LABEL[0]} on the CPU.")
    try:
        model.predictor = None              # rebuild Ultralytics' predictor on the new device
    except Exception:  # noqa: BLE001
        pass
    return model


def screen_size():
    """(width, height) of the main screen on Windows, else None."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:  # noqa: BLE001
        return None


def fit_window(w, h):
    """Shrink the window so the whole picture + panel fits on screen (e.g. a 1366x768 laptop or 150% scaling)."""
    size = screen_size()
    if not size or size[0] < 200 or size[1] < 200:
        return
    s = min(1.0, 0.95 * size[0] / w, 0.86 * size[1] / h)
    try:
        cv2.resizeWindow(WINDOW_NAME, int(w * s), int(h * s))
    except Exception:  # noqa: BLE001
        pass


def _local(name):
    return os.path.join(SCRIPT_DIR, name)


def _download(YOLO, name):
    """Load one of Ultralytics' official checkpoints, downloading it once next to this script."""
    print(f"{name} isn't downloaded yet - fetching it once (needs internet)...")
    return YOLO(_local(name))


def load_model(path_arg, gpu=False, allow_fallback=True):
    from ultralytics import YOLO
    path_arg = MODEL_ALIASES.get(path_arg, path_arg)
    if path_arg == "auto" and gpu:
        # A GPU runs the large YOLO26l (55.0 mAP) in real time; the OpenVINO copy is CPU-only, so skip it.
        if os.path.exists(_local("yolo26l.pt")):
            print(f"Loading model: {_local('yolo26l.pt')}")
            return YOLO(_local("yolo26l.pt"))
        try:
            return _download(YOLO, "yolo26l.pt")
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't download YOLO26l ({e}).")
        for name in GPU_ORDER:                        # offline: best model already on disk
            if os.path.exists(_local(name)):
                print(f"Loading model: {_local(name)}")
                return YOLO(_local(name))
    if path_arg == "auto":
        # use the fastest good model that is already downloaded (launch.bat option 6 fetches YOLO26)
        for name in AUTO_ORDER:
            local = _local(name)
            if not os.path.exists(local):
                continue
            if name == OPENVINO_DIR:
                try:                                  # check the OpenVINO runtime really works on this PC
                    m = YOLO(local, task="detect")
                    m(np.zeros((480, 640, 3), np.uint8), verbose=False, imgsz=OPENVINO_IMGSZ)
                    print(f"Loading model: {local} (OpenVINO, fast CPU inference)")
                    return m
                except Exception as e:  # noqa: BLE001
                    print(f"OpenVINO model not usable here ({e}) - using the PyTorch weights instead.")
                    continue
            path_arg = name
            break
        else:
            try:                                      # one-time ~5 MB download, saved next to this script
                return _download(YOLO, "yolo26n.pt")
            except Exception as e:  # noqa: BLE001
                print(f"Couldn't get YOLO26n ({e}).\nUsing the older YOLOv8. Run launch.bat option 6 with internet to upgrade.")
                path_arg = next((n for n in OLD_FALLBACK if os.path.exists(_local(n))), "yolov8n.pt")
    candidates = [path_arg] if os.path.isabs(path_arg) else [_local(path_arg), path_arg]
    for c in candidates:
        if os.path.exists(c):
            print(f"Loading model: {c}")
            return YOLO(c)
    if path_arg in MODEL_ALIASES.values():           # an official checkpoint: download it next to the script
        try:
            return _download(YOLO, path_arg)
        except Exception as e:  # noqa: BLE001
            if not allow_fallback:
                raise
            print(f"Couldn't download {path_arg} ({e}) - using the best model already on this laptop instead.")
            return load_model("auto", gpu=gpu, allow_fallback=False)
    raise FileNotFoundError(f"model file '{path_arg}' not found (give the full path to your .pt file)")


def detect(model, frame, conf=0.25, imgsz=640, tta=False):
    results = (model(frame, verbose=False, conf=conf, imgsz=imgsz, augment=True, **INFER_KW) if tta
               else model(frame, verbose=False, conf=conf, imgsz=imgsz, **INFER_KW))[0]
    names = model.names
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cls_id = int(box.cls[0])
        dets.append({"box": (x1, y1, x2, y2), "conf": float(box.conf[0]), "cls_id": cls_id, "cls": names[cls_id]})
    return dets


class AsyncDetector:
    """Runs the detector on a background thread.

    Doing detection inline caps the video at the model's speed (e.g. 5 FPS) and, worse, the webcam
    driver queues frames, so the picture lags further and further behind reality. Here the window shows
    every camera frame as it arrives, the newest frame is handed to the model whenever it is free (older
    ones are dropped), and boxes + decisions update each time the model finishes."""

    def __init__(self, model, conf, imgsz, tta, max_rate=15.0):
        self.model, self.conf, self.imgsz, self.tta = model, conf, imgsz, tta
        self.min_gap = 1.0 / max_rate if max_rate > 0 else 0.0   # cap detections/s: no point running faster than needed
        self.last_start = 0.0
        self.lock = threading.Lock()        # held while the model runs; take it before changing the model
        self.cv = threading.Condition()
        self.pending = None
        self.result = (0, [], 0.0)          # (sequence number, raw detections, inference ms)
        self.error = None
        self.errors_in_row = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, name="yolo-worker", daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.cv:
            self.pending = frame            # replaces any frame the model hasn't started yet
            self.cv.notify()

    def latest(self):
        with self.cv:
            return self.result

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None and self.running:
                    self.cv.wait(0.25)
                if not self.running:
                    return
                frame, self.pending = self.pending, None
            wait = self.last_start + self.min_gap - time.perf_counter()
            if wait > 0:
                time.sleep(wait)                # rest = less heat; the newest frame is still used
                with self.cv:
                    if self.pending is not None:
                        frame, self.pending = self.pending, None
            self.last_start = time.perf_counter()
            with self.lock:
                t0 = time.perf_counter()
                try:
                    dets = detect(self.model, frame, self.conf, self.imgsz, self.tta)
                    self.errors_in_row = 0
                except Exception as e:  # noqa: BLE001
                    if self.error is None or str(e) != str(self.error):
                        print(f"[detector] error: {e}")
                    self.error, dets = e, []
                    self.errors_in_row += 1
                ms = (time.perf_counter() - t0) * 1000
                with self.cv:
                    self.result = (self.result[0] + 1, dets, ms)

    def stop(self):
        with self.cv:
            self.running = False
            self.cv.notify()
        self.thread.join(timeout=2.0)


class VoiceListener:
    """Push-to-talk goals: press V, say "find my phone". Offline speech recognition with Vosk, on a
    background thread so the video keeps running. Needs `pip install vosk sounddevice` and a Vosk model
    folder next to this script (launch.bat option 6 sets both up)."""

    def __init__(self, seconds=5.0):
        self.seconds = seconds
        self.model = None
        self.state = "idle"          # idle | loading | listening | done | error
        self.text = ""
        self.error = ""
        self.thread = None

    @staticmethod
    def model_dir():
        for name in sorted(os.listdir(SCRIPT_DIR)):
            path = os.path.join(SCRIPT_DIR, name)
            if name.startswith("vosk-model") and os.path.isdir(path):
                return path
        return None

    def busy(self):
        return self.state in ("loading", "listening")

    def start(self, words=None):
        if self.busy():
            return
        self.state, self.text, self.error = "loading", "", ""
        self.thread = threading.Thread(target=self._run, args=(words,), name="voice", daemon=True)
        self.thread.start()

    def take(self):
        """Returns ('text', str) or ('error', str) once, after listening finished; else None."""
        if self.state == "done":
            self.state = "idle"
            return ("text", self.text)
        if self.state == "error":
            self.state = "idle"
            return ("error", self.error)
        return None

    def _run(self, words):
        try:
            import json
            import queue
            try:
                import sounddevice as sd
                import vosk
            except ImportError:
                raise RuntimeError("voice needs 'pip install vosk sounddevice' (launch.bat option 6)")
            except OSError as e:
                raise RuntimeError(f"no microphone driver ({e})")
            if self.model is None:
                path = self.model_dir()
                if not path:
                    raise RuntimeError("no Vosk speech model found - run launch.bat option 6 with internet")
                vosk.SetLogLevel(-1)
                self.model = vosk.Model(path)
            rec = vosk.KaldiRecognizer(self.model, 16000, json.dumps(words)) if words else vosk.KaldiRecognizer(self.model, 16000)
            q = queue.Queue()
            self.state = "listening"
            text = ""
            with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16", channels=1,
                                   callback=lambda data, frames, t, status: q.put(bytes(data))):
                t_end = time.time() + self.seconds
                while time.time() < t_end:
                    try:
                        data = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result()).get("text", "").strip()
                        if text and text != "[unk]":
                            break
            if not text:
                text = json.loads(rec.FinalResult()).get("text", "").strip()
            self.text = text.replace("[unk]", "").strip()
            self.state = "done"
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self.state = "error"


def voice_words(class_names):
    """Limit recognition to words that matter (class names, synonyms, filler) - far more accurate."""
    words = set(FILLER_WORDS) | {"where", "my", "the", "please", "find", "go", "to"}
    for name in class_names:
        words.update(str(name).lower().replace("_", " ").split())
    for syns in SYNONYMS.values():
        words.update(syns)
    return sorted(words) + ["[unk]"]


class Stabilizer:
    """Temporal filter: a detection is shown only once the same class has been seen at roughly the same
    place in `need` of the last `window` frames. Removes one-frame false alarms and box flicker."""

    def __init__(self, need=2, window=3):
        self.need, self.window = need, window
        self.tracks = []

    def update(self, dets):
        for t in self.tracks:
            t["hit"] = False
        for d in sorted(dets, key=lambda d: -d["conf"]):
            best, best_iou = None, 0.3
            for t in self.tracks:
                if not t["hit"] and t["cls"] == d["cls"]:
                    iou = box_iou(t["box"], d["box"])
                    if iou > best_iou:
                        best, best_iou = t, iou
            if best is None:
                self.tracks.append({"cls": d["cls"], "cls_id": d["cls_id"], "box": d["box"], "conf": d["conf"],
                                    "hist": [1], "hit": True})
            else:
                best["box"] = tuple(0.5 * a + 0.5 * b for a, b in zip(best["box"], d["box"]))
                best["conf"] = 0.5 * best["conf"] + 0.5 * d["conf"]
                best["hist"].append(1)
                best["hit"] = True
        for t in self.tracks:
            if not t["hit"]:
                t["hist"].append(0)
            t["hist"] = t["hist"][-self.window:]
        self.tracks = [t for t in self.tracks if sum(t["hist"]) > 0]
        return [{"box": t["box"], "conf": t["conf"], "cls": t["cls"], "cls_id": t["cls_id"]}
                for t in self.tracks if sum(t["hist"]) >= self.need]


# ----------------------------------------------------------------------------------------
# Walk mode: a person carries the camera and follows spoken directions (no robot needed)
# ----------------------------------------------------------------------------------------
class Speaker:
    """Speaks short directions, fully offline. On Windows it uses the speech voice built into Windows,
    through one long-running PowerShell process (nothing to install). Elsewhere: espeak or macOS `say`.
    The newest direction always interrupts an older one that is still being spoken."""

    def __init__(self):
        self.kind, self.ok, self.error = None, False, None
        self.proc = None
        self.exe = None

    def start(self):
        if self.kind is not None:
            return self.ok
        import shutil
        import subprocess
        if sys.platform.startswith("win"):
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; $s.Volume = 100; "
                      "while (($l = [Console]::In.ReadLine()) -ne $null) { if ($l.Trim()) { $s.SpeakAsyncCancelAll(); [void]$s.SpeakAsync($l) } }")
            try:
                self.proc = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.kind, self.ok = "windows", True
            except Exception as e:  # noqa: BLE001
                self.error = f"Windows speech unavailable ({e})"
        if not self.ok:
            self.exe = shutil.which("espeak-ng") or shutil.which("espeak") or shutil.which("say")
            if self.exe:
                self.kind, self.ok = "command", True
        if not self.ok:
            self.kind = "none"
            self.error = self.error or "no speech engine found"
        return self.ok

    def say(self, text):
        if not self.start():
            return False
        try:
            if self.kind == "windows":
                if self.proc.poll() is not None:
                    raise RuntimeError("the speech process stopped")
                self.proc.stdin.write(text.replace("\n", " ") + "\n")
                self.proc.stdin.flush()
            else:
                import subprocess
                if self.proc is not None and self.proc.poll() is None:
                    self.proc.terminate()                    # newest direction wins
                self.proc = subprocess.Popen([self.exe, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:  # noqa: BLE001
            self.ok, self.kind, self.error = False, "none", f"speech stopped working ({e})"
            return False

    def close(self):
        try:
            if self.proc is not None and self.proc.poll() is None:
                if self.kind == "windows":
                    self.proc.stdin.close()
                self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass


def say_distance(cm):
    if cm is None:
        return ""
    if cm < 100:
        return f", about {max(10, int(round(cm / 10.0)) * 10)} centimetres away"
    m = round(cm / 50.0) / 2.0
    return f", about {m:g} metre{'s' if m != 1 else ''} away"


class WalkGuide:
    """Turns the robot's decisions into spoken directions for a person carrying the camera. The same decisions
    would drive the wheels; here a person is the wheels. A direction is spoken once it has been stable for a
    moment, repeated every few seconds while it still applies, and urgent ones (obstacle, stop) go first."""
    SETTLE, URGENT_SETTLE, GAP, REPEAT = 0.5, 0.15, 1.5, 4.5

    def __init__(self, speaker):
        self.speaker = speaker
        self.cand, self.cand_t = None, 0.0
        self.spoken_key, self.spoken_t = None, -1e9
        self.text = None

    @staticmethod
    def phrase(decision, goal):
        st, cmd = decision["state"], decision["cmd"]
        name = str(goal).replace("_", " ")
        if st == "GOAL_REACHED":
            return "stop", f"Stop. The {name} is right in front of you.", True
        if st == "AVOIDING" and decision["obstacles"]:
            ob = decision["obstacles"][0]["det"]["cls"].replace("_", " ")
            side = "right" if cmd.endswith("RIGHT") else "left"
            return "avoid-" + side, f"Careful, {ob} ahead. Step to the {side}.", True
        if st == "NAVIGATING":
            if cmd.startswith("TURN LEFT"):
                return "left", "Turn a little left.", False
            if cmd.startswith("TURN RIGHT"):
                return "right", "Turn a little right.", False
            return "forward", f"The {name} is ahead{say_distance(decision.get('dist'))}. Walk straight.", False
        side = "right" if "RIGHT" in cmd else "left"
        return "search-" + side, f"Looking for the {name}. Turn slowly to your {side}.", False

    def update(self, decision, goal, now):
        if decision is None:
            return self.text
        key, text, urgent = self.phrase(decision, goal)
        if key != self.cand:
            self.cand, self.cand_t = key, now
        if now - self.cand_t < (self.URGENT_SETTLE if urgent else self.SETTLE):
            return self.text                              # not stable yet: keep showing the last direction
        new = key != self.spoken_key
        if (new and now - self.spoken_t >= (0.3 if urgent else self.GAP)) or \
                (not new and now - self.spoken_t >= (self.REPEAT * 1.6 if key == "stop" else self.REPEAT)):
            self.speaker.say(text)
            self.spoken_key, self.spoken_t, self.text = key, now, text
        return self.text

    def reset(self):
        self.cand, self.spoken_key, self.spoken_t, self.text = None, None, -1e9, None


class NetworkCapture:
    """A camera over Wi-Fi, e.g. a phone running the free IP Webcam app. Frames are read on a thread and only
    the newest is kept, so the picture never lags behind when Wi-Fi is slower than the stream."""

    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()
        self.frame, self.seq, self.given = None, 0, 0
        self.stopped = False
        if self.cap.isOpened():
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame, self.seq = f, self.seq + 1

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        end = time.time() + 0.1
        while time.time() < end:
            with self.lock:
                if self.seq != self.given and self.frame is not None:
                    self.given = self.seq
                    return True, self.frame.copy()
            time.sleep(0.005)
        return False, None

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, value):
        return False

    def release(self):
        self.stopped = True
        self.cap.release()


def camera_source(value):
    """--camera: a webcam number (0, 1...) or a phone/IP camera address. '192.168.1.5:8080' becomes
    'http://192.168.1.5:8080/video', the IP Webcam app's video address."""
    v = str(value).strip()
    if v.isdigit():
        return int(v)
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", v):
        v = "http://" + v + ("" if ":" in v else ":8080")
    if v.startswith("http") and re.match(r"^https?://[^/]+/?$", v):
        v = v.rstrip("/") + "/video"
    return v


def open_capture(args):
    if not args.video and isinstance(args.camera, str):
        print(f"Opening network camera {args.camera} ...")
        return NetworkCapture(args.camera)
    if args.video:
        print(f"Opening video: {args.video}")
        return cv2.VideoCapture(args.video)
    print(f"Opening webcam {args.camera}...")
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # opens much faster on Windows
        if not cap.isOpened():
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(args.camera)
    # On a GPU there's headroom for HD; on a CPU 640x480 keeps things fast.
    # MJPG lets most webcams deliver HD at 30 FPS (raw YUY2 HD is often only 5-10 FPS over USB).
    w, h = args.cam_width, args.cam_height
    if not w:
        w, h = (1280, 720) if DEVICE_LABEL[0].startswith("GPU") else (640, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Webcam: asked for {w}x{h}, got {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                      # always hand us the newest frame
    return cap


# ----------------------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SightSeeker Vision Brain: YOLO perception -> navigation decisions")
    parser.add_argument("--goal", type=str, default="bottle", help="goal in plain English, e.g. 'find my phone'")
    parser.add_argument("--image", type=str, help="run on a single image")
    parser.add_argument("--video", type=str, help="run on a video file instead of the webcam")
    parser.add_argument("--camera", type=camera_source, default=0,
                        help="webcam number (default 0), or a phone camera address, e.g. 192.168.1.5:8080 (IP Webcam app)")
    parser.add_argument("--twin", action="store_true",
                        help="digital twin: show a virtual rover driven by the same wheel commands (D toggles it)")
    parser.add_argument("--walk", action="store_true",
                        help="walk mode: speak directions for a person carrying the camera (W toggles it)")
    parser.add_argument("--model", type=str, default="auto",
                        help="auto (YOLO26l on an NVIDIA GPU, YOLO26n on a CPU), n / s / m / l / x, or a path to .pt weights")
    parser.add_argument("--conf", type=float, default=0.35, help="detector confidence threshold (higher = fewer false alarms)")
    parser.add_argument("--all-classes", action="store_true", help="show all 80 COCO classes (default: common indoor objects)")
    parser.add_argument("--tta", action="store_true", help="test-time augmentation (older YOLOv8 models only; YOLO26 ignores it)")
    parser.add_argument("--open-vocab", action="store_true",
                        help="open-vocabulary mode (YOLOE): the goal can be ANY object you describe, e.g. 'red toolbox'")
    parser.add_argument("--imgsz", type=int, default=0,
                        help="inference size: default 480 on a CPU (fast), 640 on a GPU; 640 sees small objects better")
    parser.add_argument("--cam-width", type=int, default=0, help="webcam width (default 1280 on a GPU, 640 on a CPU)")
    parser.add_argument("--cam-height", type=int, default=0, help="webcam height (default 720 on a GPU, 480 on a CPU)")
    parser.add_argument("--ai-fps", type=float, default=15, help="max detections per second (lower = cooler laptop; 0 = no limit)")
    parser.add_argument("--max-fps", type=float, default=30, help="max video frames per second shown (lower = cooler laptop)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch/OpenCV (fewer = cooler; CPU-only mode uses more)")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto (NVIDIA GPU if available, else CPU), cpu, 0 (first GPU), or intel:gpu for an OpenVINO model")
    parser.add_argument("--record", action="store_true",
                        help="record the annotated window to recordings/ from the start (R toggles recording live)")
    parser.add_argument("--calib-cm", type=float, default=50.0,
                        help="distance (cm) you hold the target at when pressing C to calibrate (default 50)")
    parser.add_argument("--hfov", type=float, default=70.0,
                        help="webcam horizontal field of view in degrees, for distance estimates (press C to calibrate instead)")
    parser.add_argument("--max", action="store_true",
                        help="MAXIMUM mode: most accurate model, biggest images, HD camera, no speed caps, every CPU core "
                             "(runs the laptop hot - plug it in)")
    parser.add_argument("--sync", action="store_true",
                        help="run detection inline on every frame (slower video; default for --video files)")
    parser.add_argument("--stop-at", type=float, default=0.60, help="stop when the target fills this fraction of the frame")
    parser.add_argument("--people-are-obstacles", action="store_true", help="treat people as obstacles (robot mode)")
    parser.add_argument("--serial", type=str, help="serial port of an Arduino rover, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--save", type=str, help="also record the annotated output to this .mp4 file")
    parser.add_argument("--no-window", action="store_true", help="process without showing a window (use with --video/--image)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = parser.parse_args()
    # Paths are relative to where you started; then work from this folder, where the models live
    # (Ultralytics also keeps YOLOE's text encoder in the current folder).
    for attr in ("video", "image", "save"):
        if getattr(args, attr):
            setattr(args, attr, os.path.abspath(getattr(args, attr)))
    if args.model != "auto" and args.model not in MODEL_ALIASES and os.path.exists(args.model):
        args.model = os.path.abspath(args.model)
    os.chdir(SCRIPT_DIR)

    print("Initializing SightSeeker Vision Brain...")
    device, DEVICE_LABEL[0], half = pick_device(args.device)
    on_gpu = DEVICE_LABEL[0].startswith("GPU")
    print(f"Device: {DEVICE_LABEL[0]}")
    MAX_MODE[0] = args.max
    CAMERA["hfov"] = args.hfov
    if load_calibration():
        print(f"Camera calibration loaded from {CALIB_FILE} (distances in cm are calibrated).")
    if args.max:
        # Maximum accuracy, whole machine: the biggest model, more pixels per object, no detection or
        # frame-rate caps, all CPU cores. On a GPU: YOLO26x at 960 px with a Full-HD webcam.
        # On a CPU YOLO26x would take seconds per frame, so it uses YOLO26l at 640 px instead.
        if args.model == "auto":
            args.model = "x" if on_gpu else "l"
        if not args.imgsz:
            args.imgsz = 960 if on_gpu else 640
        if not args.cam_width:
            args.cam_width, args.cam_height = (1920, 1080) if on_gpu else (1280, 720)
        args.ai_fps = 0
        args.max_fps = 0
        args.threads = os.cpu_count() or 8
        print(f"MAXIMUM mode: model {args.model}, {args.imgsz} px, camera {args.cam_width}x{args.cam_height}, "
              f"no speed caps, {args.threads} CPU threads. The laptop will run hot - keep it plugged in.")
    # By default PyTorch and OpenCV grab every CPU core even when the GPU does the work -> a hot laptop.
    cv2.setNumThreads(args.threads)
    try:
        import torch
        torch.set_num_threads(args.threads if (on_gpu or args.max) else max(args.threads, 4))
    except Exception:  # noqa: BLE001
        pass
    if not args.imgsz:
        args.imgsz = 640 if on_gpu else 480
    open_vocab = False
    if args.open_vocab:
        try:
            model = load_open_vocab()
            goal = goal_phrase(args.goal)
            class_names = set_prompts(model, goal)
            open_vocab = True
            print(f"Open-vocabulary goal: '{args.goal}' -> prompt '{goal}'")
        except Exception as e:  # noqa: BLE001
            print(f"Open-vocabulary mode unavailable ({e}).\nRun launch.bat option 6 (needs internet) - falling back to the standard model.")
    if not open_vocab:
        try:
            model = load_model(args.model, gpu=on_gpu)
        except ImportError as e:
            print(f"Error: {e}\nInstall the packages with launch.bat option 6 (or: pip install -r requirements.txt).")
            sys.exit(1)
        except Exception as e:  # noqa: BLE001
            print(f"Error loading model: {e}")
            sys.exit(1)
        class_names = list(model.names.values())
        goal, matched = ground_goal(args.goal, class_names)
        if goal is None:
            goal = "bottle" if "bottle" in class_names else class_names[0]
            print(f"Could not ground goal '{args.goal}' to a detectable object - using '{goal}'.")
        else:
            print(f"Goal: '{args.goal}' -> class '{goal}'")

    brain = NavBrain(goal, stop_at=args.stop_at, presenter_mode=not args.people_are_obstacles)
    src = str(getattr(model, "ckpt_path", None) or getattr(model, "model_name", None) or getattr(model, "model", None) or args.model)
    model_name = os.path.basename(os.path.normpath(src)) if len(src) < 300 else args.model
    MODEL_LABEL[0] = model_name.replace(".pt", "")
    if "openvino" in model_name:
        MODEL_LABEL[0] = model_name.replace("_openvino_model", "") + " (OpenVINO)"
        args.imgsz = OPENVINO_IMGSZ
        if args.device == "intel:gpu":
            INFER_KW["device"] = "intel:gpu"
            DEVICE_LABEL[0] = "Intel GPU (OpenVINO)"
        else:
            DEVICE_LABEL[0] = "CPU (OpenVINO)"
    else:
        if device == "intel:gpu":
            print("--device intel:gpu only works with the OpenVINO model (made by launch.bat option 6) - using the CPU.")
            device = "cpu"
        INFER_KW["device"] = device
        if half:
            INFER_KW.update(fp16_kwargs())   # FP16 on NVIDIA: faster, same accuracy
    coco_model = "person" in class_names and len(class_names) == 80   # a fine-tuned model has its own classes
    filt = {"indoor": not args.all_classes and not open_vocab and coco_model}
    end2end = bool(getattr(getattr(model, "model", None), "end2end", False))   # YOLO26 is NMS-free end-to-end
    if args.tta and (end2end or "26" in MODEL_LABEL[0] or open_vocab):
        print("--tta isn't supported by YOLO26 / YOLOE models (they skip it) - ignoring it.")
        args.tta = False
    if open_vocab:
        MODEL_LABEL[0] = OPEN_VOCAB_WEIGHTS.replace(".pt", "") + " (open vocab)"

    def class_filter(dets):
        if not filt["indoor"]:
            return dets
        return [d for d in dets if d["cls"] in INDOOR_CLASSES or d["cls"] == brain.goal]

    def fall_back_to_cpu():
        """Move detection to the CPU (smaller model + smaller images) and return the model now in use."""
        nonlocal model
        model = switch_to_cpu(model, swap_small=coco_model and not open_vocab)
        if "OpenVINO" not in MODEL_LABEL[0]:
            args.imgsz = 480
        return model

    def detect_or_fallback(frame):
        """Run the detector; if the GPU fails, switch to the CPU once and retry instead of crashing."""
        try:
            return detect(model, frame, args.conf, args.imgsz, args.tta)
        except Exception as e:  # noqa: BLE001
            if INFER_KW.get("device") in ("cpu", None):
                raise
            print(f"Detection on {DEVICE_LABEL[0]} failed ({e}) - switching to the CPU.")
            fall_back_to_cpu()
            return detect(model, frame, args.conf, args.imgsz, args.tta)

    stabilizer = Stabilizer()
    print(f"Model: {MODEL_LABEL[0]} on {DEVICE_LABEL[0]} | imgsz {args.imgsz} | conf >= {args.conf} | classes: "
          f"{'open vocabulary' if open_vocab else ('indoor set' if filt['indoor'] else f'all {len(class_names)}')}"
          f"{' | TTA on' if args.tta else ''}")
    serial_link = SerialBridge(args.serial) if args.serial else None
    show = not args.no_window
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowTitle(WINDOW_NAME, f"SightSeeker - Vision Brain ({MODEL_LABEL[0]} on {DEVICE_LABEL[0]})")
        except Exception:  # noqa: BLE001
            pass

    # ---------------- single image ----------------
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: could not read image {args.image}")
            sys.exit(1)
        t0 = time.perf_counter()
        try:
            dets = class_filter(detect_or_fallback(frame))
        except Exception as e:  # noqa: BLE001
            print(f"Error: the detector failed ({e}).")
            sys.exit(3)
        infer_ms = (time.perf_counter() - t0) * 1000
        decision = brain.update(dets, frame.shape[1], frame.shape[0], time.time())
        out = compose(frame, dets, decision, brain, 0.0, infer_ms, serial_link, True)
        print(f"Detected {len(dets)} objects | state={decision['state']} | command={decision['cmd']} | PWM L/R={decision['pwm']}")
        if args.save:
            if os.path.splitext(args.save)[1].lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                args.save = os.path.splitext(args.save)[0] + ".png"     # an image result can't be saved as video
            print(f"Saved {args.save}" if cv2.imwrite(args.save, out) else f"Could not save {args.save}")
        if show:
            fit_window(out.shape[1], out.shape[0])
            cv2.imshow(WINDOW_NAME, out)
            print("Press S to save, any other key to exit.")
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("s"), ord("S")):
                name = os.path.join(SCRIPT_DIR, f"detected_{os.path.splitext(os.path.basename(args.image))[0]}.png")
                print(f"Saved {name}" if cv2.imwrite(name, out) else f"Could not save {name}")
            cv2.destroyAllWindows()
        return

    # ---------------- video / webcam ----------------
    cap = open_capture(args)
    if not cap.isOpened():
        if isinstance(args.camera, str) and not args.video:
            print(f"Error: could not connect to {args.camera}. Is the IP Webcam app running ('Start server'), "
                  "and are the phone and laptop on the same Wi-Fi or hotspot?")
        else:
            print("Error: could not open the video source (webcam busy or not found? try --camera 1).")
        sys.exit(1)

    print("\n--- Controls ---\nT: type goal | V: voice goal | G: next goal | W: walk mode (spoken directions) | D: digital twin | "
          "C: calibrate distance | R: record | N: overlay | P: presenter mode | SPACE: pause | S: screenshot | Q/ESC: quit\n")
    print("Warming up the detector...")
    try:
        detect_or_fallback(np.zeros((480, 640, 3), np.uint8))   # the first call is always slow
    except Exception as e:  # noqa: BLE001
        print(f"Error: the detector failed to start ({e}).")
        cap.release()
        sys.exit(3)
    threaded = not args.video and not args.sync
    worker = AsyncDetector(model, args.conf, args.imgsz, args.tta, args.ai_fps) if threaded else None
    print(f"Detection: {'background thread (smooth video)' if threaded else 'inline on every frame'} | imgsz {args.imgsz}")

    def reprompt(g):
        """Change the open-vocabulary prompts safely while the worker thread may be using the model."""
        try:
            if worker is None:
                return set_prompts(model, g)
            with worker.lock:
                return set_prompts(model, g)
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't change the open-vocabulary prompt ({e}) - keeping the previous goal.")
            return None

    writer = None
    fps, prev_t = 0.0, time.perf_counter()
    ai_fps, prev_ai_t = 0.0, time.perf_counter()
    infer_ms = 0.0
    last_seq = 0
    paused = False
    nav_overlay = True
    typing = None
    message, message_until = None, 0.0
    out = None
    frames = 0
    read_fail = 0
    exit_code = 0
    last_result_t = -1e9                 # when the detector last delivered a result (threaded mode)
    cpu_fallback_done = False
    window_fitted = False
    dets, decision = [], None
    goal_idx = DEMO_GOALS.index(goal) if goal in DEMO_GOALS else -1
    voice = VoiceListener()
    rec_writer, rec_path, rec_wanted = None, None, args.record
    speaker = Speaker()
    twin = DigitalTwin() if args.twin else None
    twin_goal = None
    walk = WalkGuide(speaker) if args.walk else None
    walk_text = None
    if walk is not None:
        if speaker.start():
            print("Walk mode: speaking directions. Carry the camera and follow them.")
        else:
            print(f"Walk mode: {speaker.error} - directions are shown on screen only.")

    def set_goal_from_text(text):
        """Typed or spoken goal -> new target. Returns the message to show."""
        nonlocal class_names, last_seq, stabilizer, goal_idx
        if open_vocab:
            g = goal_phrase(text)
            names = reprompt(g)
            if names is None:
                return f"Couldn't switch to '{g}' - still looking for {brain.goal}"
            class_names = names
            last_seq = worker.latest()[0] if worker is not None else 0   # drop results made with the old prompts
            brain.set_goal(g)
            stabilizer = Stabilizer()
            return f"Open-vocabulary goal -> '{g}'"
        g, m = ground_goal(text, class_names)
        if not g:
            return f"Can't ground '{text}' to a COCO object. Try: bottle, cup, phone, book, person"
        brain.set_goal(g)
        goal_idx = DEMO_GOALS.index(g) if g in DEMO_GOALS else -1
        return f"'{text}' -> goal: {g}  (matched '{m}')"

    def start_recording():
        os.makedirs(os.path.join(SCRIPT_DIR, "recordings"), exist_ok=True)
        path = os.path.join(SCRIPT_DIR, "recordings", f"sightseeker_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        fps_out = int(min(60, max(5, round(fps)))) if fps > 0 else 15
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (out.shape[1], out.shape[0]))
        if not w.isOpened():
            return None, None
        print(f"Recording to {path}")
        return w, path

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    read_fail += 1
                    if args.video:
                        print("End of video stream.")
                        break
                    if read_fail > 150:                  # ~3 s with no picture at all
                        print("The webcam stopped sending pictures (in use by another app? try --camera 1).")
                        exit_code = 2
                        break
                    time.sleep(0.02)                     # webcams can take a moment to start, or drop a frame
                    continue
                read_fail = 0
                frames += 1
                now = time.perf_counter()
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    fps = (1.0 / dt) if fps == 0 else (0.9 * fps + 0.1 / dt)
                if worker is not None:
                    worker.submit(frame)
                    seq, raw, ms = worker.latest()
                    if seq != last_seq:                  # the model finished a new frame
                        last_seq, infer_ms = seq, ms
                        last_result_t = now
                        if worker.errors_in_row >= 3:
                            if not cpu_fallback_done and INFER_KW.get("device") not in ("cpu", None):
                                with worker.lock:
                                    worker.model = fall_back_to_cpu()
                                    worker.imgsz = args.imgsz
                                    worker.errors_in_row = 0
                                cpu_fallback_done = True
                                print("Detection keeps failing on the GPU - switched to the CPU.")
                                message, message_until = "GPU detection failed - switched to the CPU", time.time() + 6
                            elif time.time() >= message_until:
                                message, message_until = f"Detector error: {str(worker.error)[:70]}", time.time() + 1.5
                        dets = stabilizer.update(class_filter(raw))
                        decision = brain.update(dets, frame.shape[1], frame.shape[0], now)
                        adt = now - prev_ai_t
                        prev_ai_t = now
                        if adt > 0:
                            ai_fps = (1.0 / adt) if ai_fps == 0 else (0.8 * ai_fps + 0.2 / adt)
                    elif decision is None:
                        decision = brain.update([], frame.shape[1], frame.shape[0], now)
                else:
                    t0 = time.perf_counter()
                    dets = stabilizer.update(class_filter(detect_or_fallback(frame)))
                    infer_ms = (time.perf_counter() - t0) * 1000
                    decision = brain.update(dets, frame.shape[1], frame.shape[0], now)
                if serial_link is not None:
                    # Re-sent every frame to keep the rover's watchdog fed - but only while the decision is
                    # fresh. If the detector stalls or errors, send STOP rather than repeat an old command.
                    stale = worker is not None and (now - last_result_t > 1.0 or worker.errors_in_row > 0)
                    serial_link.send(*((0, 0) if stale else decision["pwm"]))

            if walk is not None and decision is not None and not paused:
                walk_text = walk.update(decision, brain.goal, time.perf_counter())
            if twin is not None and not paused:
                if brain.goal != twin_goal:                  # a new goal starts a new path
                    twin.reset()
                    twin_goal = brain.goal
                twin.update(decision, time.perf_counter())
            elif twin is not None:
                twin.last_t = None
            heard = voice.take()
            if heard:
                kind, text = heard
                if kind == "error":
                    message = f"Voice: {text}"
                elif text:
                    message = "Heard '" + text + "': " + set_goal_from_text(text)
                else:
                    message = "Voice: didn't catch that - press V and try again"
                message_until = time.time() + 5
            banner = None
            if voice.state == "loading":
                banner = "Loading speech model..."
            elif voice.state == "listening":
                banner = "Listening... say e.g. 'find my phone'"
            msg = message if time.time() < message_until else None
            out = compose(frame, dets, decision, brain, fps, infer_ms, serial_link, nav_overlay, typing, msg,
                          ai_fps if worker is not None else None,
                          overlay={"rec": rec_writer is not None, "voice": banner, "now": time.perf_counter(),
                                   "walk": walk_text if walk is not None else None, "walk_on": walk is not None,
                                   "twin": twin})
            if rec_wanted and rec_writer is None and (args.video or frames >= 20):
                rec_writer, rec_path = start_recording()
                if rec_writer is None:
                    rec_wanted = False
                    message, message_until = "Could not start recording", time.time() + 3
            if rec_writer is not None and not paused:
                rec_writer.write(out)
            if args.save and (args.video or frames >= 20):    # webcam: wait until the real frame rate is known
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    src_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else round(fps)
                    writer = cv2.VideoWriter(args.save, fourcc, src_fps if src_fps and src_fps > 1 else 15, (out.shape[1], out.shape[0]))
                if not paused:
                    writer.write(out)
            if args.max_frames and frames >= args.max_frames:
                break
            if not show:
                continue

            if not window_fitted:
                fit_window(out.shape[1], out.shape[0])
                window_fitted = True
            cv2.imshow(WINDOW_NAME, out)
            if paused:
                wait_ms = 30
            else:
                spare = (1.0 / args.max_fps) - (time.perf_counter() - prev_t) if args.max_fps > 0 else 0
                wait_ms = max(1, int(spare * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF                          # idle instead of spinning
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
            if key == 255:
                continue

            if typing is not None:                       # ---- goal typing mode ----
                if key in (13, 10):
                    if typing.strip():
                        message, message_until = set_goal_from_text(typing), time.time() + 4
                    typing = None
                elif key == 27:
                    typing = None
                elif key == 8:
                    typing = typing[:-1]
                elif 32 <= key < 127 and len(typing) < 40:
                    typing += chr(key)
                continue

            if key in (ord("q"), ord("Q"), 27):
                print("Exiting SightSeeker...")
                break
            elif key in (ord("t"), ord("T")):
                typing = ""
            elif key in (ord("d"), ord("D")):
                twin = None if twin is not None else DigitalTwin()
                twin_goal = brain.goal
                window_fitted = False                    # the window gets wider/narrower
                message, message_until = ("Digital twin ON: a virtual rover follows the same wheel commands"
                                          if twin is not None else "Digital twin OFF"), time.time() + 4
            elif key in (ord("w"), ord("W")):
                if walk is None:
                    walk, walk_text = WalkGuide(speaker), None
                    message = ("Walk mode ON: carry the camera and follow the spoken directions" if speaker.start()
                               else f"Walk mode ON (no sound: {speaker.error}) - follow the directions on screen")
                else:
                    walk, walk_text = None, None
                    message = "Walk mode OFF"
                message_until = time.time() + 4
            elif key in (ord("g"), ord("G")):
                goal_idx = (goal_idx + 1) % len(DEMO_GOALS)
                ok_switch = True
                if open_vocab:
                    names = reprompt(DEMO_GOALS[goal_idx])
                    ok_switch = names is not None
                    if ok_switch:
                        class_names = names
                        last_seq = worker.latest()[0] if worker is not None else 0
                        stabilizer = Stabilizer()
                if ok_switch:
                    brain.set_goal(DEMO_GOALS[goal_idx])
                    message, message_until = f"Goal -> {DEMO_GOALS[goal_idx]}", time.time() + 2
            elif key in (ord("v"), ord("V")):
                if voice.busy():
                    message, message_until = "Still listening...", time.time() + 2
                else:
                    voice.start(None if open_vocab else voice_words(class_names))
            elif key in (ord("c"), ord("C")):
                t = decision["target"] if decision else None
                px = long_side_px(t["box"], frame.shape[1], frame.shape[0]) if t else None
                size = REAL_SIZE_CM.get(t["cls"]) if t else None
                if px and size:
                    save_calibration(args.calib_cm * px / size / frame.shape[1])
                    message = f"Camera calibrated with the {t['cls']} at {args.calib_cm:.0f} cm - saved for next time"
                elif brain.goal not in REAL_SIZE_CM:
                    message = "Calibrate with an everyday object of known size: set the goal to bottle, cup or phone first"
                else:
                    message = (f"To calibrate: hold the {brain.goal} fully in view, {args.calib_cm:.0f} cm from the camera, "
                               "then press C")
                message_until = time.time() + 5
            elif key in (ord("r"), ord("R")):
                if rec_writer is None:
                    rec_wanted = True
                    rec_writer, rec_path = start_recording()
                    message = "Recording started" if rec_writer else "Could not start recording"
                    rec_wanted = rec_writer is not None
                else:
                    rec_writer.release()
                    message = f"Recording saved: recordings/{os.path.basename(rec_path)}"
                    print(f"Saved recording: {rec_path}")
                    rec_writer, rec_wanted = None, False
                message_until = time.time() + 4
            elif key in (ord("n"), ord("N")):
                nav_overlay = not nav_overlay
            elif key in (ord("a"), ord("A")) and (open_vocab or not coco_model):
                message = "Showing every object this model knows" if not open_vocab else "Open-vocabulary mode shows every prompted object"
                message_until = time.time() + 3
            elif key in (ord("a"), ord("A")):
                filt["indoor"] = not filt["indoor"]
                message = "Showing common indoor objects only" if filt["indoor"] else "Showing all 80 COCO classes"
                message_until = time.time() + 3
            elif key in (ord("p"), ord("P")):
                brain.presenter_mode = not brain.presenter_mode
                message = "Presenter mode ON: people/desks aren't obstacles" if brain.presenter_mode else "Robot mode: people ARE obstacles"
                message_until = time.time() + 3
            elif key in (ord("s"), ord("S")):
                name = os.path.join(SCRIPT_DIR, f"sightseeker_screenshot_{int(time.time())}.png")
                saved = cv2.imwrite(name, out)
                print(f"Screenshot saved to {name}" if saved else f"Could not save {name}")
                message, message_until = ("Screenshot saved" if saved else "Could not save the screenshot"), time.time() + 2
            elif key == ord(" "):
                paused = not paused
                if not paused:
                    prev_t = time.perf_counter()
    finally:
        if worker is not None:
            worker.stop()
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved video: {args.save}")
        if rec_writer is not None:
            rec_writer.release()
            print(f"Saved recording: {rec_path}")
        if serial_link is not None:
            serial_link.close()
        speaker.close()
        if show:
            cv2.destroyAllWindows()
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
