"""FastAPI server — runs the CV loop in a background thread and exposes
REST endpoints for a Next.js frontend.

Run:
    python server.py --camera 0
    python server.py --camera 0 --use-saved
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from calibration import calibrate, load_calibration
from game import CVEvent, RefereeEngine, RallyPhase, TableNormalizer
from tracker import BallTracker, BounceDetector, OOBDetector
from display import draw_table, draw_score

# =====================================================================
# Shared game state (written by CV thread, read by WebSocket handler)
# =====================================================================

class GameState:
    def __init__(self):
        self.score_a: int = 0
        self.score_b: int = 0
        self.match_winner: str | None = None
        self.ball_x: float | None = None   # normalized 0-1
        self.ball_y: float | None = None   # pixel y
        self.state: str = "playing"
        self.last_point: str | None = None  # e.g. "Player A — Double bounce"
        self.frame_jpg: bytes | None = None # raw JPEG bytes for /frame endpoint
        self.reset_requested: bool = False
        self._lock = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def score_snapshot(self) -> dict:
        """Lightweight snapshot without the frame — for REST polling."""
        with self._lock:
            return {
                "score_a": self.score_a,
                "score_b": self.score_b,
                "match_winner": self.match_winner,
                "ball_x": self.ball_x,
                "ball_y": self.ball_y,
                "state": self.state,
                "last_point": self.last_point,
            }

game_state = GameState()

# =====================================================================
# CV loop (runs in its own thread)
# =====================================================================

def cv_loop(camera_index: int, use_saved: bool) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return

    cal = load_calibration() if use_saved else None
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
    last_nx = 0.5
    frame = first if ret else None

    print("CV loop started. Open http://localhost:8000 in your browser.")

    while True:
        # Handle new-game reset request from API
        if game_state.reset_requested:
            engine = RefereeEngine()
            bounce_det = BounceDetector()
            oob_det = OOBDetector()
            trail.clear()
            game_state.update(
                score_a=0, score_b=0, match_winner=None,
                state="playing", last_point=None,
                reset_requested=False,
            )

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        ts = int((time.time() - t0) * 1000)
        result = tracker.update(frame)
        disp = frame.copy()

        ball_x_norm: float | None = None
        ball_y_pix: float | None = None

        if result is not None:
            px, py, vx, vy = result
            ipx, ipy = int(px), int(py)
            trail.append((ipx, ipy))
            if len(trail) > 30:
                trail.pop(0)

            nx = normalizer.normalize_x(px)
            last_nx = float(nx)
            ball_x_norm = nx
            ball_y_pix = py
            table_y_for_oob = normalizer.get_table_y(float(px))

            raw = tracker.last_raw
            if raw is not None:
                table_y = normalizer.get_table_y(float(raw[0]))
                is_bounce, vyp, vyc = bounce_det.update(float(raw[0]), float(raw[1]), table_y)
            else:
                is_bounce, vyp, vyc = False, 0.0, 0.0

            last_pt_str: str | None = None

            if is_bounce:
                pt = engine.process_event(CVEvent(ts, nx, 0.5, vyp, vyc))
                if pt:
                    last_pt_str = f"{pt.winner} — {pt.reason}"
                    bounce_det.reset()
                    oob_det.reset()

            if oob_det.update(nx, float(py), table_y_for_oob, float(px), frame.shape[1]):
                pt = engine.process_event(CVEvent(ts, nx, 0.5, 1.0, -1.0))
                if pt:
                    last_pt_str = f"{pt.winner} — {pt.reason}"
                    bounce_det.reset()
                    oob_det.reset()

            # Draw trail + ball
            for i in range(1, len(trail)):
                a = i / len(trail)
                cv2.line(disp, trail[i-1], trail[i], (0, int(100*a), int(255*a)), max(1, int(3*a)))
            cv2.circle(disp, (ipx, ipy), 8, (0, 0, 255), -1)
            cv2.circle(disp, (ipx, ipy), 10, (255, 255, 255), 2)
            side_label = "A" if nx <= 0.5 else "B"
            cv2.putText(disp, f"x={nx:.2f} [{side_label}]", (ipx+15, ipy-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            s = engine.scorer
            update_kwargs = dict(
                score_a=s.score_a,
                score_b=s.score_b,
                match_winner=s.match_winner,
                ball_x=ball_x_norm,
                ball_y=ball_y_pix,
                state=engine.state_machine.state.value,
            )
            if last_pt_str:
                update_kwargs["last_point"] = last_pt_str
            game_state.update(**update_kwargs)
        else:
            trail.clear()
            game_state.update(ball_x=None, ball_y=None)

            if engine.state_machine.phase == RallyPhase.RALLY and tracker.last_raw is None:
                pt = engine.process_event(
                    CVEvent(ts, last_nx, 0.5, 1.0, -1.0, oob_source="camera-range")
                )
                if pt:
                    bounce_det.reset()
                    oob_det.reset()
                    s = engine.scorer
                    game_state.update(
                        score_a=s.score_a,
                        score_b=s.score_b,
                        match_winner=s.match_winner,
                        state=engine.state_machine.state.value,
                        last_point=f"{pt.winner} — {pt.reason}",
                    )

        pt = engine.check_timeout(ts)
        if pt:
            bounce_det.reset()
            oob_det.reset()
            s = engine.scorer
            game_state.update(
                score_a=s.score_a,
                score_b=s.score_b,
                match_winner=s.match_winner,
                state=engine.state_machine.state.value,
                last_point=f"{pt.winner} — {pt.reason}",
            )

        draw_table(disp, table_pts)
        draw_score(disp, engine)

        # Encode frame as JPEG for /frame endpoint
        _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 70])
        game_state.update(frame_jpg=buf.tobytes())

        # Also show the local OpenCV window (optional — close it if headless)
        cv2.imshow("Ping Pong Referee", disp)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================================================================
# FastAPI app
# =====================================================================

app = FastAPI(title="Ping Pong Referee API")

# Allow Next.js dev server (and any origin in dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Next.js domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# REST endpoints (consumed by Next.js)
# ------------------------------------------------------------------

@app.get("/score")
async def get_score():
    """Current score + game state. Poll this or use /ws for real-time."""
    return JSONResponse(game_state.score_snapshot())


@app.post("/game/reset")
async def reset_game():
    """Start a new game — resets scores and all detectors."""
    game_state.update(reset_requested=True)
    return JSONResponse({"ok": True, "message": "New game started"})


@app.get("/frame")
async def get_frame():
    """Latest camera frame as JPEG image (snapshot, not streaming)."""
    jpg = game_state.frame_jpg
    if jpg is None:
        return Response(status_code=503, content="Frame not ready yet")
    return Response(content=jpg, media_type="image/jpeg")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Ping Pong Referee — Web Server")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--use-saved", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Start CV loop in background thread
    t = threading.Thread(target=cv_loop, args=(args.camera, args.use_saved), daemon=True)
    t.start()

    uvicorn.run(app, host=args.host, port=args.port)
