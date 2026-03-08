"""
Ping Pong Referee -- Real-time ball tracking + automated referee.

Side-view camera setup: click left end + right end of table.
Net is auto-computed at the midpoint.

Run:  python main.py --camera 0
      python main.py --camera 1 --use-saved

Controls:
    Q / ESC  = quit
    R        = recalibrate (ball color + table endpoints)
    SPACE    = pause / unpause
"""
from __future__ import annotations

import argparse
import time

import cv2

from calibration import calibrate, load_calibration
from display import draw_score, draw_table
from game import CVEvent, RefereeEngine, RallyPhase, TableNormalizer
from tracker import BallTracker, BounceDetector, OOBDetector


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
    last_nx = 0.5
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
            last_nx = float(nx)
            table_y_for_oob = normalizer.get_table_y(float(px))

            raw = tracker.last_raw
            if raw is not None:
                table_y_for_bounce = normalizer.get_table_y(float(raw[0]))
                is_bounce, vyp, vyc = bounce_det.update(float(raw[0]), float(raw[1]), table_y_for_bounce)
            else:
                is_bounce, vyp, vyc = False, 0.0, 0.0

            if is_bounce:
                pt = engine.process_event(CVEvent(ts, nx, 0.5, vyp, vyc))
                if pt:
                    _print_point(engine, pt)
                    bounce_det.reset()
                    oob_det.reset()

            if oob_det.update(nx, float(py), table_y_for_oob, float(px), frame.shape[1]):
                pt = engine.process_event(CVEvent(ts, nx, 0.5, 1.0, -1.0))
                if pt:
                    _print_point(engine, pt)
                    bounce_det.reset()
                    oob_det.reset()

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

            if engine.state_machine.phase == RallyPhase.RALLY and tracker.last_raw is None:
                pt = engine.process_event(
                    CVEvent(ts, last_nx, 0.5, 1.0, -1.0, oob_source="camera-range")
                )
                if pt:
                    _print_point(engine, pt)
                    bounce_det.reset()
                    oob_det.reset()

        pt = engine.check_timeout(ts)
        if pt:
            _print_point(engine, pt)
            bounce_det.reset()
            oob_det.reset()

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
