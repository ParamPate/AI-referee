"""
Drawing helpers: table overlay, scoreboard.
"""
from __future__ import annotations

import cv2
import numpy as np

from game import RefereeEngine


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
    phase = sm.state.value
    striker = getattr(sm, "current_striker", "?")
    cv2.putText(frame, f"Phase: {phase}  Striker: {striker}",
                (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    if s.match_winner:
        txt = f"{s.match_winner} WINS!"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        cx, cy = (w-tw)//2, h//2
        cv2.rectangle(frame, (cx-20, cy-th-20), (cx+tw+20, cy+20), (0,0,0), -1)
        cv2.putText(frame, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
