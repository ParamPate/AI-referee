"""
Ball detection: HSV color tracker with Kalman filter, bounce detector, OOB detector.
"""
from __future__ import annotations

import cv2
import numpy as np


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
            if circ < 0.15:  # confidence threshold
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
