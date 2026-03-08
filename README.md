# AI Referee

A real-time automated ping pong referee system using computer vision. The system tracks the ball, detects bounces and out-of-bounds violations, and automatically scores rallies according to official ping pong rules.

## Features

- Real-time ball tracking via HSV color detection and Kalman filtering
- Automatic bounce detection using vertical velocity reversal
- Out-of-bounds detection with hysteresis
- Full serve and rally rule enforcement (state machine)
- Official ping pong scoring (first to 11, win by 2)
- Standalone OpenCV GUI mode or FastAPI web server mode
- Interactive calibration wizard

## How It Works

1. **Calibration** — Click on the ball to sample its color, then click the left and right ends of the table. Settings are saved to `calibration.json`.
2. **Tracking** — Each camera frame is processed with HSV color filtering and Kalman filtering to locate the ball and estimate its velocity.
3. **Event Detection** — Bounces are detected by monitoring vertical velocity reversals near the table surface. Out-of-bounds fires when the ball exits the normalized table region.
4. **Game Logic** — A state machine enforces serve rules (ball must bounce server's side first, then cross to the opponent's side) and awards points for faults, missed returns, and out-of-bounds.

The server runs the CV loop in a background thread and exposes a REST API.

**Endpoints:**

| Method | Endpoint      | Description                               |
| ------ | ------------- | ----------------------------------------- |
| `GET`  | `/score`      | Current scores, game state, ball position |
| `POST` | `/game/reset` | Reset scores and start a new game         |
| `GET`  | `/frame`      | Latest camera frame as JPEG               |

## Project Structure

```
AI-referee/
├── main.py            # Standalone GUI application
├── server.py          # FastAPI web server
├── game.py            # Game logic and state machine
├── tracker.py         # Ball detection and bounce/OOB detection
├── display.py         # OpenCV drawing utilities
├── calibration.py     # Interactive calibration wizard
├── calibration.json   # Saved calibration data
└── requirements.txt   # Python dependencies
```

## Key Technologies

| Technology         | Usage                                           |
| ------------------ | ----------------------------------------------- |
| OpenCV             | Camera capture, HSV filtering, contour analysis |
| Kalman Filter      | Smooth predictive ball tracking (x, y, vx, vy)  |
| Frame Differencing | Motion detection for mid-flight tracking        |
| FastAPI            | REST API for web frontend integration           |
| NumPy              | Numerical processing                            |

## Game Rules Enforced

- Serve must bounce on the server's side first, then cross to the opponent's side
- A missed return or out-of-bounds awards the point to the last valid striker
- If no return is detected within 3 seconds after a bounce, the point is awarded
- First player to 11 points wins, must win by 2
- Server alternates every 2 points
