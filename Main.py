"""
Ping Pong Referee -- Real-time ball tracking + automated referee.

Side-view camera setup: click left end + right end of table.
Net is auto-computed at the midpoint.

Run:  python Main.py --camera 0
      python Main.py --camera 1 --use-saved

Controls:
    Q / ESC  = quit
    R        = recalibrate (ball color + table endpoints)
    SPACE    = pause / unpause
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# =====================================================================
# Table mapping
# =====================================================================

PLAYER_A = "Player A"
PLAYER_B = "Player B"
LEFT = "left"
RIGHT = "right"
OUT = "out"

SIDE_TO_PLAYER = {LEFT: PLAYER_A, RIGHT: PLAYER_B}
PLAYER_TO_SIDE = {PLAYER_A: LEFT, PLAYER_B: RIGHT}


def table_side(x: float) -> str:
    return LEFT if x <= 0.5 else RIGHT


def is_in_bounds(x: float) -> bool:
    return 0.0 <= x <= 1.0


def table_region(x: float) -> str:
    return table_side(x) if is_in_bounds(x) else OUT


def player_for_side(s: str) -> str:
    return SIDE_TO_PLAYER.get(s, "Unknown")


def side_for_player(p: str) -> str:
    return PLAYER_TO_SIDE.get(p, LEFT)


def opponent(p: str) -> str:
    return PLAYER_B if p == PLAYER_A else PLAYER_A


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class CVEvent:
    timestamp: int
    x: float
    y: float
    vy_prev: float
    vy_current: float


@dataclass
class PointResult:
    winner: str
    reason: str


# =====================================================================
# Scoring  (first to 11, lead by 2, server rotates every 2 pts)
# =====================================================================

class Scorer:
    def __init__(self, initial_server: str = PLAYER_A):
        self.score_a = 0
        self.score_b = 0
        self.initial_server = initial_server
        self.current_server = initial_server
        self.match_winner: str | None = None

    @property
    def total_points(self) -> int:
        return self.score_a + self.score_b

    def add_point(self, winner: str) -> None:
        if self.match_winner:
            return
        if winner == PLAYER_A:
            self.score_a += 1
        else:
            self.score_b += 1
        self._check_winner()
        if not self.match_winner:
            self._update_server()

    def is_match_over(self) -> bool:
        return self.match_winner is not None

    def _check_winner(self) -> None:
        if self.score_a >= 11 and self.score_a - self.score_b >= 2:
            self.match_winner = PLAYER_A
        elif self.score_b >= 11 and self.score_b - self.score_a >= 2:
            self.match_winner = PLAYER_B

    def _update_server(self) -> None:
        total = self.total_points
        other = opponent(self.initial_server)
        if min(self.score_a, self.score_b) >= 10:
            deuce_pts = total - 20
            self.current_server = self.initial_server if deuce_pts % 2 == 0 else other
        else:
            block = total // 2
            self.current_server = self.initial_server if block % 2 == 0 else other


# =====================================================================
# Game state machine  (bounce tracking -> point end)
# =====================================================================

class State(str, Enum):
    PLAYING = "playing"
    POINT_END = "point_end"


class GameStateMachine:
    def __init__(self):
        self.state = State.PLAYING
        self.last_bounce_side: str | None = None
        self.last_bounce_ts: int | None = None

    def process_event(self, event: CVEvent) -> PointResult | None:
        if self.state == State.POINT_END:
            return None
        r = table_region(event.x)
        if r == OUT:
            if self.last_bounce_side is not None:
                hitter = player_for_side(self.last_bounce_side)
                self.state = State.POINT_END
                return PointResult(winner=opponent(hitter), reason="Out of bounds")
            return None
        if self.last_bounce_side is None:
            self.last_bounce_side = r
            self.last_bounce_ts = event.timestamp
            return None
        if r == self.last_bounce_side:
            failed = player_for_side(r)
            self.state = State.POINT_END
            return PointResult(winner=opponent(failed), reason="Double bounce")
        self.last_bounce_side = r
        self.last_bounce_ts = event.timestamp
        return None

    def check_timeout(self, now_ms: int, timeout_ms: int = 2000) -> PointResult | None:
        if self.state == State.POINT_END:
            return None
        if self.last_bounce_ts is None or self.last_bounce_side is None:
            return None
        if now_ms - self.last_bounce_ts >= timeout_ms:
            loser = player_for_side(self.last_bounce_side)
            self.state = State.POINT_END
            return PointResult(winner=opponent(loser), reason="No return (2s timeout)")
        return None


# =====================================================================
# Referee engine  (wires state machine + scorer together)
# =====================================================================

class RefereeEngine:
    def __init__(self, initial_server: str = PLAYER_A):
        self.scorer = Scorer(initial_server)
        self.state_machine = GameStateMachine()

    def process_event(self, event: CVEvent) -> PointResult | None:
        if self.scorer.is_match_over():
            return None
        result = self.state_machine.process_event(event)
        if result is None:
            return None
        self.scorer.add_point(result.winner)
        if not self.scorer.is_match_over():
            self.state_machine = GameStateMachine()
        return result

    def check_timeout(self, now_ms: int) -> PointResult | None:
        if self.scorer.is_match_over():
            return None
        result = self.state_machine.check_timeout(now_ms)
        if result is None:
            return None
        self.scorer.add_point(result.winner)
        if not self.scorer.is_match_over():
            self.state_machine = GameStateMachine()
        return result


# =====================================================================
# Table normalizer  (2-point side-view: linear interpolation)
# =====================================================================

class TableNormalizer:
    """
    Side-view model: two endpoints define the table.
    normalize_x  -> 0.0 at left end (Player A), 1.0 at right end (Player B).
    get_table_y  -> interpolated pixel y of the table surface at a given pixel x.
    """

    def __init__(self, left_pt: list[int], right_pt: list[int]):
        self.lx, self.ly = float(left_pt[0]), float(left_pt[1])
        self.rx, self.ry = float(right_pt[0]), float(right_pt[1])

    def normalize_x(self, px: float) -> float:
        if self.rx == self.lx:
            return 0.5
        return (px - self.lx) / (self.rx - self.lx)

    def get_table_y(self, px: float) -> float:
        if self.rx == self.lx:
            return (self.ly + self.ry) / 2
        return self.ly + (self.ry - self.ly) * (px - self.lx) / (self.rx - self.lx)

    @property
    def net_pixel(self) -> tuple[int, int]:
        return (int((self.lx + self.rx) / 2), int((self.ly + self.ry) / 2))


# =====================================================================
# Ball tracker  (HSV color + Kalman filter)
# =====================================================================

class BallTracker:
    MAX_JUMP_PX = 250

    def __init__(self, hsv_lower: list[int], hsv_upper: list[int]):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self._init_kalman()
        self._kalman_ready = False
        self._lost_count = 16
        self._last_raw: tuple[int, int] | None = None
        self._last_good: tuple[float, float] | None = None
        self._roi_mask: np.ndarray | None = None
        self.debug_mask: np.ndarray | None = None

    def set_table_roi(self, table_pts: list[list[int]], shape: tuple,
                      h_pad: int = 200, v_pad_above: int = 300, v_pad_below: int = 80) -> None:
        """Rectangular band around the table line with generous space above."""
        fh, fw = shape[:2]
        (lx, ly), (rx, ry) = table_pts
        x1 = max(0, min(lx, rx) - h_pad)
        x2 = min(fw, max(lx, rx) + h_pad)
        y1 = max(0, min(ly, ry) - v_pad_above)
        y2 = min(fh, max(ly, ry) + v_pad_below)
        mask = np.zeros((fh, fw), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        self._roi_mask = mask

    def update(self, frame: np.ndarray) -> tuple[float, float, float, float] | None:
        det = self._detect(frame)
        self._last_raw = det
        if det is not None:
            cx, cy = det
            self._lost_count = 0
            self._last_good = (float(cx), float(cy))
            m = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not self._kalman_ready:
                self.kf.statePre[:2] = m
                self.kf.statePost[:2] = m
                self._kalman_ready = True
            self.kf.predict()
            self.kf.correct(m)
            s = self.kf.statePost
            return float(s[0][0]), float(s[1][0]), float(s[2][0]), float(s[3][0])
        self._lost_count += 1
        if self._kalman_ready and self._lost_count <= 15:
            p = self.kf.predict()
            return float(p[0][0]), float(p[1][0]), float(p[2][0]), float(p[3][0])
        return None

    @property
    def last_raw(self) -> tuple[int, int] | None:
        return self._last_raw

    def _detect(self, frame: np.ndarray) -> tuple[int, int] | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        if self._roi_mask is not None:
            mask = cv2.bitwise_and(mask, self._roi_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        self.debug_mask = mask.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 15 or area > 12000:
                continue
            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue
            circ = 4.0 * np.pi * area / (perim * perim)
            if circ < 0.3:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            if self._last_good is not None:
                if np.hypot(cx - self._last_good[0], cy - self._last_good[1]) > self.MAX_JUMP_PX:
                    continue
            score = circ * area
            if score > best_score:
                best_score = score
                best = (cx, cy)
        return best

    def _init_kalman(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0


def sample_ball_color(frame: np.ndarray, cx: int, cy: int) -> tuple[list[int], list[int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    r = 18
    region = hsv[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
    ah = float(np.median(region[:,:,0]))
    a_s = float(np.median(region[:,:,1]))
    av = float(np.median(region[:,:,2]))
    lo = [int(max(0, ah-15)), int(max(30, a_s-55)), int(max(30, av-55))]
    hi = [int(min(180, ah+15)), int(min(255, a_s+55)), int(min(255, av+55))]
    return lo, hi


# =====================================================================
# Bounce + out-of-bounds detectors
# =====================================================================

class BounceDetector:
    def __init__(self) -> None:
        self.positions: list[tuple[float, float]] = []
        self.cooldown = 9

    def update(self, x: float, y: float) -> tuple[bool, float, float]:
        self.positions.append((x, y))
        self.cooldown += 1
        n = len(self.positions)
        if n < 7:
            return False, 0.0, 0.0
        vy_prev = self._vy(-6, -3)
        vy_curr = self._vy(-3, 0)
        if abs(vy_prev) < 1.5 or abs(vy_curr) < 1.5:
            return False, vy_prev, vy_curr
        if vy_prev > 0 and vy_curr < 0 and self.cooldown > 8:
            self.cooldown = 0
            return True, vy_prev, vy_curr
        return False, vy_prev, vy_curr

    def reset(self) -> None:
        self.positions.clear()
        self.cooldown = 9

    def _vy(self, s: int, e: int) -> float:
        seg = self.positions[max(0, len(self.positions)+s):len(self.positions)+e]
        return (seg[-1][1] - seg[0][1]) / max(len(seg)-1, 1) if len(seg) >= 2 else 0.0


class OOBDetector:
    """Fires once when the ball's normalized x leaves [0, 1]."""

    def __init__(self) -> None:
        self.was_in = False
        self.fired = False

    def update(self, nx: float) -> bool:
        m = 0.08
        inside = -m <= nx <= 1 + m
        if inside:
            self.was_in = True
            self.fired = False
            return False
        if self.was_in and not self.fired:
            self.fired = True
            return True
        return False

    def reset(self) -> None:
        self.was_in = False
        self.fired = False


# =====================================================================
# Calibration  (click ball color + click 2 table endpoints)
# =====================================================================

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
DEFAULT_HSV_LO = [5, 100, 100]
DEFAULT_HSV_HI = [25, 255, 255]


def save_calibration(hsv_lo: list[int], hsv_hi: list[int], table_pts: list[list[int]]) -> None:
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"hsv_lower": hsv_lo, "hsv_upper": hsv_hi, "table_pts": table_pts}, f, indent=2)


def load_calibration() -> tuple | None:
    if not os.path.exists(CALIBRATION_FILE):
        return None
    with open(CALIBRATION_FILE) as f:
        d = json.load(f)
    if "table_pts" not in d:
        return None
    return d["hsv_lower"], d["hsv_upper"], d["table_pts"]


def calibrate(cap: cv2.VideoCapture) -> tuple[list[int], list[int], list[list[int]]]:
    win = "Ping Pong Referee - Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Step 1: ball color
    click = [None]
    def on_click(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click[0] = (x, y)
    cv2.setMouseCallback(win, on_click)

    hsv_lo, hsv_hi = DEFAULT_HSV_LO[:], DEFAULT_HSV_HI[:]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        cv2.putText(disp, "STEP 1: Click on the BALL", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(disp, "Press D=default orange | ESC=skip", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        if click[0]:
            cx, cy = click[0]
            hsv_lo, hsv_hi = sample_ball_color(frame, cx, cy)
            cv2.circle(disp, (cx,cy), 20, (0,255,0), 2)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_bgr[:,:,0] = 0; mask_bgr[:,:,2] = 0
            disp = cv2.addWeighted(disp, 0.7, mask_bgr, 0.3, 0)
            cv2.putText(disp, f"HSV: {hsv_lo}-{hsv_hi} | ENTER=confirm | Click=redo",
                        (20, disp.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('d'), ord('D')):
            hsv_lo, hsv_hi = DEFAULT_HSV_LO[:], DEFAULT_HSV_HI[:]
            break
        if key == 13 and click[0]:
            break
        if key == 27:
            break

    # Step 2: two table endpoints
    endpoints: list[list[int]] = []
    labels = ["Left End (A)", "Right End (B)"]
    colors = [(0, 255, 0), (255, 255, 0)]

    def on_endpoint(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(endpoints) < 2:
            endpoints.append([x, y])
    cv2.setMouseCallback(win, on_endpoint)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        n = len(endpoints)
        if n < 2:
            cv2.putText(disp, f"STEP 2: Click {labels[n]}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        else:
            cv2.putText(disp, "Both endpoints set!  ENTER=start | R=redo",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        for i, (ex, ey) in enumerate(endpoints):
            cv2.circle(disp, (ex, ey), 8, colors[i], -1)
            cv2.putText(disp, labels[i], (ex+12, ey-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        if len(endpoints) == 2:
            lp, rp = tuple(endpoints[0]), tuple(endpoints[1])
            cv2.line(disp, lp, rp, (0, 255, 255), 2)
            net = ((lp[0]+rp[0])//2, (lp[1]+rp[1])//2)
            cv2.line(disp, (net[0], net[1]-30), (net[0], net[1]+30), (255,255,255), 2)
            cv2.putText(disp, "NET", (net[0]+8, net[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(endpoints) == 2:
            break
        if key in (ord('r'), ord('R')):
            endpoints.clear()
        if key == 27:
            break

    cv2.destroyWindow(win)
    if len(endpoints) < 2:
        h, w = frame.shape[:2]
        endpoints = [[50, h//2], [w-50, h//2]]

    save_calibration(hsv_lo, hsv_hi, endpoints)
    return hsv_lo, hsv_hi, endpoints


# =====================================================================
# Drawing helpers
# =====================================================================

def draw_table(frame: np.ndarray, table_pts: list[list[int]]) -> None:
    lp = tuple(table_pts[0])
    rp = tuple(table_pts[1])
    cv2.line(frame, lp, rp, (0, 255, 255), 2)
    net = ((lp[0]+rp[0])//2, (lp[1]+rp[1])//2)
    cv2.line(frame, (net[0], net[1]-25), (net[0], net[1]+25), (255, 255, 255), 2)
    cv2.putText(frame, "A", (lp[0]-25, lp[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, "B", (rp[0]+10, rp[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


def draw_score(frame: np.ndarray, engine: RefereeEngine) -> None:
    s = engine.scorer
    sm = engine.state_machine
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w, 55), (30,30,30), -1)
    cv2.putText(frame, f"Player A: {s.score_a}  |  Player B: {s.score_b}",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Serve: {s.current_server}",
                (w-310, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.putText(frame, f"State: {sm.state.value}",
                (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    if s.match_winner:
        txt = f"{s.match_winner} WINS!"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        cx, cy = (w-tw)//2, h//2
        cv2.rectangle(frame, (cx-20, cy-th-20), (cx+tw+20, cy+20), (0,0,0), -1)
        cv2.putText(frame, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)


# =====================================================================
# Main loop
# =====================================================================

def run(camera_index: int, use_saved: bool = False) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return

    cal = None
    if use_saved:
        cal = load_calibration()
    if cal is None:
        cal = calibrate(cap)
    hsv_lo, hsv_hi, table_pts = cal

    tracker = BallTracker(hsv_lo, hsv_hi)
    normalizer = TableNormalizer(table_pts[0], table_pts[1])
    bounce_det = BounceDetector()
    oob_det = OOBDetector()
    engine = RefereeEngine()

    ret, first = cap.read()
    if ret:
        tracker.set_table_roi(table_pts, first.shape)

    t0 = time.time()
    trail: list[tuple[int, int]] = []
    paused = False
    frame = first if ret else None

    print("=" * 55)
    print("  PING PONG REFEREE -- LIVE")
    print("  Q=quit  R=recalibrate  SPACE=pause")
    print("=" * 55)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        if frame is None:
            break

        result = tracker.update(frame)
        disp = frame.copy()
        ts = int((time.time() - t0) * 1000)

        if result is not None:
            px, py, vx, vy = result
            ipx, ipy = int(px), int(py)
            trail.append((ipx, ipy))
            if len(trail) > 30:
                trail.pop(0)

            nx = normalizer.normalize_x(px)

            raw = tracker.last_raw
            if raw is not None:
                is_bounce, vyp, vyc = bounce_det.update(float(raw[0]), float(raw[1]))
            else:
                is_bounce, vyp, vyc = False, 0.0, 0.0

            if is_bounce:
                pt = engine.process_event(CVEvent(ts, nx, 0.5, vyp, vyc))
                if pt:
                    _print_point(engine, pt)

            if oob_det.update(nx):
                pt = engine.process_event(CVEvent(ts, nx, 0.5, 1.0, -1.0))
                if pt:
                    _print_point(engine, pt)

            cv2.circle(disp, (ipx, ipy), 8, (0,0,255), -1)
            cv2.circle(disp, (ipx, ipy), 10, (255,255,255), 2)
            for i in range(1, len(trail)):
                a = i / len(trail)
                cv2.line(disp, trail[i-1], trail[i], (0, int(100*a), int(255*a)), max(1, int(3*a)))
            side_label = "A" if nx <= 0.5 else "B"
            cv2.putText(disp, f"x={nx:.2f} [{side_label}]", (ipx+15, ipy-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        else:
            trail.clear()

        timeout_pt = engine.check_timeout(ts)
        if timeout_pt:
            _print_point(engine, timeout_pt)

        draw_table(disp, table_pts)
        draw_score(disp, engine)

        if tracker.debug_mask is not None:
            dh, dw = disp.shape[:2]
            iw, ih = dw//4, dh//4
            small = cv2.resize(tracker.debug_mask, (iw, ih))
            disp[dh-ih:dh, dw-iw:dw] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            cv2.putText(disp, "Color mask", (dw-iw+5, dh-ih+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.imshow("Ping Pong Referee", disp)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord('q'), 27):
            break
        if key in (ord('r'), ord('R')):
            hsv_lo, hsv_hi, table_pts = calibrate(cap)
            tracker = BallTracker(hsv_lo, hsv_hi)
            tracker.set_table_roi(table_pts, frame.shape)
            normalizer = TableNormalizer(table_pts[0], table_pts[1])
            bounce_det.reset()
            oob_det.reset()
            trail.clear()
        if key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    s = engine.scorer
    print("\n" + "=" * 55)
    print(f"  Final score  : Player A {s.score_a} - Player B {s.score_b}")
    if s.match_winner:
        print(f"  Winner       : {s.match_winner}")
    else:
        print("  Match not finished.")
    print(f"  Points played: {s.total_points}")
    print("=" * 55)


def _print_point(engine: RefereeEngine, pt: PointResult) -> None:
    s = engine.scorer
    print(f"\n  >>> POINT -> {pt.winner}  ({pt.reason})")
    print(f"      Score : Player A {s.score_a} - Player B {s.score_b}")
    print(f"      Server: {s.current_server}")
    if s.is_match_over():
        print(f"\n  === GAME OVER! {s.match_winner} WINS! ===")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ping Pong Referee")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--use-saved", action="store_true", help="Skip calibration, reuse saved")
    args = parser.parse_args()
    run(args.camera, use_saved=args.use_saved)
