"""
Game logic: table mapping, scoring, state machine, referee engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
# Scoring  (first to 11, lead by 2)
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
# Rally phase  (replaces the old binary State enum)
# =====================================================================

class RallyPhase(str, Enum):
    SERVE_START = "serve"           # Waiting for 1st bounce on server's side
    SERVE_CROSS = "serve-cross"     # 1st bounce done, waiting for receiver's side
    RALLY       = "rally"           # Normal rally
    POINT_END   = "point_end"       # Point ended, waiting for reset


# =====================================================================
# Game state machine  (striker-aware, with serve validation)
# =====================================================================

class GameStateMachine:
    POST_POINT_LOCKOUT_MS = 1500

    def __init__(self, server: str = PLAYER_A):
        self.phase = RallyPhase.SERVE_START
        self.server = server
        self.current_striker: str = server
        self._bounce_history: list[tuple[str, int]] = []
        self._last_point_ts: int = -(self.POST_POINT_LOCKOUT_MS + 1)

    # ── backward-compatible alias used by display module ─────────────
    @property
    def state(self) -> RallyPhase:
        return self.phase

    # ── convenience properties (read-only) ──────────────────────────
    @property
    def last_bounce_side(self) -> str | None:
        return self._bounce_history[-1][0] if self._bounce_history else None

    @property
    def last_bounce_ts(self) -> int | None:
        return self._bounce_history[-1][1] if self._bounce_history else None

    # ── public interface ─────────────────────────────────────────────
    def process_event(self, event: CVEvent) -> PointResult | None:
        if self.phase == RallyPhase.POINT_END:
            return None
        if event.timestamp - self._last_point_ts < self.POST_POINT_LOCKOUT_MS:
            return None
        is_oob = (event.vy_prev == 1.0 and event.vy_current == -1.0)
        if is_oob:
            return self._handle_oob(event)
        return self._handle_bounce(event)

    def check_timeout(self, now_ms: int, timeout_ms: int = 3000) -> PointResult | None:
        if self.phase == RallyPhase.POINT_END:
            return None
        if not self._bounce_history:
            return None
        if now_ms - self._last_point_ts < self.POST_POINT_LOCKOUT_MS:
            return None
        _, last_ts = self._bounce_history[-1]
        if now_ms - last_ts >= timeout_ms:
            result = PointResult(
                winner=self.current_striker,
                reason=f"No return by {opponent(self.current_striker)} (3s timeout)",
            )
            self._end_point(now_ms)
            return result
        return None

    def reset(self) -> None:
        self._bounce_history.clear()
        self.phase = RallyPhase.SERVE_START
        self.current_striker = self.server

    # ── internal helpers ─────────────────────────────────────────────
    def _end_point(self, ts: int) -> None:
        self.phase = RallyPhase.POINT_END
        self._last_point_ts = ts

    def _award(self, winner: str, reason: str, ts: int) -> PointResult:
        self._end_point(ts)
        return PointResult(winner=winner, reason=reason)

    # ── bounce handler ───────────────────────────────────────────────
    def _handle_bounce(self, event: CVEvent) -> PointResult | None:
        side = table_side(event.x)
        self._bounce_history.append((side, event.timestamp))

        # --- SERVE phase 1: first bounce must be on server's side ---
        if self.phase == RallyPhase.SERVE_START:
            expected = side_for_player(self.server)
            if side == expected:
                self.phase = RallyPhase.SERVE_CROSS
                return None
            return self._award(
                opponent(self.server),
                "Serve fault: first bounce on wrong side",
                event.timestamp,
            )

        # --- SERVE phase 2: second bounce must be on receiver's side -
        if self.phase == RallyPhase.SERVE_CROSS:
            expected = side_for_player(opponent(self.server))
            if side == expected:
                self.phase = RallyPhase.RALLY
                return None
            return self._award(
                opponent(self.server),
                "Serve fault: didn't reach opponent's side",
                event.timestamp,
            )

        # --- RALLY phase ─────────────────────────────────────────────
        if len(self._bounce_history) < 2:
            return None

        prev_side = self._bounce_history[-2][0]
        curr_side = self._bounce_history[-1][0]

        if curr_side == prev_side:
            # Double bounce on the same side
            striker_side = side_for_player(self.current_striker)
            if curr_side == striker_side:
                # Ball came back to striker's own side (net / return fault)
                return self._award(
                    opponent(self.current_striker),
                    f"Net/return fault by {self.current_striker}",
                    event.timestamp,
                )
            # Ball bounced twice on opponent's side — opponent failed to return
            return self._award(
                self.current_striker,
                f"Double bounce — {opponent(self.current_striker)} failed to return",
                event.timestamp,
            )

        # Ball crossed to the other side — valid return
        # The player on the *previous* bounce's side made the hit
        self.current_striker = player_for_side(prev_side)
        return None

    # ── OOB handler ──────────────────────────────────────────────────
    def _handle_oob(self, event: CVEvent) -> PointResult | None:
        # Serve phase — any OOB is a serve fault
        if self.phase in (RallyPhase.SERVE_START, RallyPhase.SERVE_CROSS):
            return self._award(
                opponent(self.server),
                "Serve fault: ball out of bounds",
                event.timestamp,
            )

        # Rally — attribute based on last bounce vs. striker
        if not self._bounce_history:
            return None

        last_side = self._bounce_history[-1][0]
        striker_side = side_for_player(self.current_striker)

        if last_side != striker_side:
            # Last bounce was on opponent's side — they didn't return it
            return self._award(
                self.current_striker,
                f"{opponent(self.current_striker)} failed to return",
                event.timestamp,
            )
        # Last bounce was on striker's own side — striker hit it out
        return self._award(
            opponent(self.current_striker),
            f"{self.current_striker} hit out of bounds",
            event.timestamp,
        )


# =====================================================================
# Referee engine  (wires state machine + scorer together)
# =====================================================================

class RefereeEngine:
    def __init__(self, initial_server: str = PLAYER_A):
        self.scorer = Scorer(initial_server)
        self.state_machine = GameStateMachine(server=initial_server)

    def process_event(self, event: CVEvent) -> PointResult | None:
        if self.scorer.is_match_over():
            return None
        result = self.state_machine.process_event(event)
        if result is None:
            return None
        self.scorer.add_point(result.winner)
        if not self.scorer.is_match_over():
            self.state_machine = GameStateMachine(server=self.scorer.current_server)
        return result

    def check_timeout(self, now_ms: int) -> PointResult | None:
        if self.scorer.is_match_over():
            return None
        result = self.state_machine.check_timeout(now_ms)
        if result is None:
            return None
        self.scorer.add_point(result.winner)
        if not self.scorer.is_match_over():
            self.state_machine = GameStateMachine(server=self.scorer.current_server)
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
