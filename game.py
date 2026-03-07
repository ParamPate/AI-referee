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
    # Ignore bounce/OOB events for this long after a point is awarded
    POST_POINT_LOCKOUT_MS = 1500

    def __init__(self):
        self.state = State.PLAYING
        # Each entry is (side, timestamp_ms)
        self._bounce_history: list[tuple[str, int]] = []
        self._last_point_ts: int = -(self.POST_POINT_LOCKOUT_MS + 1)

    # ── convenience properties (read-only) ──────────────────────────
    @property
    def last_bounce_side(self) -> str | None:
        return self._bounce_history[-1][0] if self._bounce_history else None

    @property
    def last_bounce_ts(self) -> int | None:
        return self._bounce_history[-1][1] if self._bounce_history else None

    # ── public interface ─────────────────────────────────────────────
    def process_event(self, event: CVEvent) -> PointResult | None:
        if self.state == State.POINT_END:
            return None
        # Ignore events during post-point lockout
        if event.timestamp - self._last_point_ts < self.POST_POINT_LOCKOUT_MS:
            return None
        # Sentinel: vy_prev==1.0 and vy_current==-1.0 means OOB
        is_oob = (event.vy_prev == 1.0 and event.vy_current == -1.0)
        if is_oob:
            return self._handle_oob(event)
        return self._handle_bounce(event)

    def check_timeout(self, now_ms: int, timeout_ms: int = 3000) -> PointResult | None:
        if self.state == State.POINT_END:
            return None
        if not self._bounce_history:
            return None
        if now_ms - self._last_point_ts < self.POST_POINT_LOCKOUT_MS:
            return None
        last_side, last_ts = self._bounce_history[-1]
        if now_ms - last_ts >= timeout_ms:
            loser = player_for_side(last_side)
            result = PointResult(winner=opponent(loser), reason="No return (3s timeout)")
            self.state = State.POINT_END
            self._last_point_ts = now_ms
            return result
        return None

    # ── private helpers ──────────────────────────────────────────────
    def _handle_bounce(self, event: CVEvent) -> PointResult | None:
        side = table_side(event.x)
        self._bounce_history.append((side, event.timestamp))

        # Need at least two bounces to decide anything
        if len(self._bounce_history) < 2:
            return None

        prev_side = self._bounce_history[-2][0]
        curr_side = self._bounce_history[-1][0]

        if curr_side == prev_side:
            # Ball bounced twice on the same side — that player failed to return
            failed = player_for_side(curr_side)
            result = PointResult(winner=opponent(failed), reason="Double bounce")
            self.state = State.POINT_END
            self._last_point_ts = event.timestamp
            return result
        return None

    def _handle_oob(self, event: CVEvent) -> PointResult | None:
        # Must have seen at least one bounce to attribute blame
        if not self._bounce_history:
            return None
        last_side = self._bounce_history[-1][0]
        hitter = player_for_side(last_side)
        result = PointResult(winner=opponent(hitter), reason="Out of bounds")
        self.state = State.POINT_END
        self._last_point_ts = event.timestamp
        return result


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
