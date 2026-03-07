"""
Calibration: ball color picker + table endpoint selection, save/load JSON.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from tracker import sample_ball_color


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
