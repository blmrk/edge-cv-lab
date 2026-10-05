# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""Zone entry/exit logic: a naive baseline and a debounced state machine.

The naive counter is the bug. The debounced counter is the fix. Both consume the
same track stream, so the difference is measurable on any recording.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal

from .geometry import Polygon, distance_to_edge, point_in_polygon
from .schema import TrackBox

Anchor = Literal["centroid", "footpoint"]


@dataclass(frozen=True)
class ZoneEvent:
    kind: Literal["enter", "exit"]
    track_id: int
    ts_ms: int
    frame: int


def _anchor(box: TrackBox, anchor: Anchor):
    return box.footpoint if anchor == "footpoint" else box.centroid


class NaiveZoneCounter:
    """Emit an event every time the anchor point flips sides. Jitter on the boundary = event spam."""

    def __init__(self, polygon: Polygon, anchor: Anchor = "centroid"):
        self.polygon, self.anchor = polygon, anchor
        self._inside: dict[int, bool] = {}

    def update(self, box: TrackBox) -> list[ZoneEvent]:
        now = point_in_polygon(_anchor(box, self.anchor), self.polygon)
        was = self._inside.get(box.track_id, False)
        self._inside[box.track_id] = now
        if now == was:
            return []
        return [ZoneEvent("enter" if now else "exit", box.track_id, box.ts_ms, box.frame)]


@dataclass
class _State:
    inside: bool = False
    streak: int = 0  # consecutive frames disagreeing with the committed state
    entered_ts: int = 0
    last_exit_ts: int | None = None
    last_ts: int = 0
    last_frame: int = 0
    pending: ZoneEvent | None = field(default=None)
    origin: tuple[float, float] | None = None  # anchor when the track was first seen
    moved: bool = False


class DebouncedZoneCounter:
    """Hysteresis state machine.

    - enter_frames / exit_frames: the anchor must sit on the other side for N consecutive
      frames before the state commits. Kills boundary jitter.
    - margin_px: the anchor must also be at least this far past the edge for a frame to count
      towards the other side. Frame counts alone cannot stop a vehicle parked on the edge: its
      jitter lands each side about half the time, so a long enough idle eventually yields both
      runs and a phantom visit. Set it above the footpoint jitter of a parked vehicle.
      Trade-off: a vehicle that stops closer than margin_px to an edge never flips state (a stop just
      inside is not counted, a wait just outside keeps the visit open), and a zone must be at least
      2 * margin_px wide to count anything. Keep stopping areas clear of the zone's edges.
    - min_dwell_ms: a visit shorter than this is discarded entirely (enter and exit both dropped). To make that
      possible the enter is held back: by default until the visit closes, so it is returned together with its exit
      (stamped with the time the vehicle entered) and a live consumer never sees a vehicle that is still inside.
    - enter_after_dwell: return the held enter as soon as the visit has lasted min_dwell_ms instead of with its
      exit. Exits still wait for exit_frames, or lost_ms for a lost track, so a live count of enters minus exits runs
      ahead of the vehicles inside; for live occupancy use open_visits(now, seen_within_ms) (docs/case-study-balance.md).
      Same events and timestamps either way; only when they are returned changes.
    - cooldown_ms: after an exit, the same track cannot re-enter for this long.
    - lost_ms: a track that vanishes while inside (tracker dropped it, ID changed, object occluded)
      is closed after this long, stamped with the time it was last seen. Without this, a lost
      track is a visit that never ends.
    - min_travel_px: an enter commits only once the anchor has moved at least this far from where the
      track was first seen. A box that never moves (a lane marking scored as a car) never enters.
      Trade-off: a vehicle already standing in the zone when its track starts counts only once it has
      moved this far, stamped then (and dropped if it leaves within min_dwell_ms); one that never moves
      never counts. Set it above the anchor jitter of a parked vehicle. 0, the default, turns it off.
    - anchor="footpoint": shadows and tall vehicles stretch the box, but move the ground
      contact point far less than the centroid.
    """

    def __init__(
        self,
        polygon: Polygon,
        anchor: Anchor = "footpoint",
        enter_frames: int = 5,
        exit_frames: int = 8,
        min_dwell_ms: int = 1000,
        cooldown_ms: int = 1500,
        lost_ms: int = 3000,
        margin_px: float = 10,
        min_travel_px: float = 0,
        enter_after_dwell: bool = False,
    ):
        self.polygon, self.anchor, self.margin_px, self.min_travel_px = polygon, anchor, margin_px, min_travel_px
        self.enter_frames, self.exit_frames = enter_frames, exit_frames
        self.min_dwell_ms, self.cooldown_ms, self.lost_ms = min_dwell_ms, cooldown_ms, lost_ms
        self.enter_after_dwell = enter_after_dwell
        self._s: dict[int, _State] = {}

    def _close(self, tid: int, s: _State, ts_ms: int, frame: int) -> list[ZoneEvent]:
        s.inside, s.streak = False, 0
        pending, s.pending = s.pending, None
        if ts_ms - s.entered_ts < self.min_dwell_ms:
            return []
        s.last_exit_ts = ts_ms
        return ([pending] if pending else []) + [ZoneEvent("exit", tid, ts_ms, frame)]

    def expire(self, now_ms: int) -> list[ZoneEvent]:
        """Close visits whose track has not been seen for lost_ms. Called from update(); call it
        yourself on frames with no boxes at all if you need exits to be prompt."""
        out: list[ZoneEvent] = []
        for tid, s in self._s.items():
            if s.inside and now_ms - s.last_ts > self.lost_ms:
                out += self._close(tid, s, s.last_ts, s.last_frame)
        return out

    def update(self, box: TrackBox) -> list[ZoneEvent]:
        out = self.expire(box.ts_ms) + self._update(box)
        s = self._s[box.track_id]  # after _update: a visit closed by this box has already returned its enter
        if self.enter_after_dwell and s.pending and box.ts_ms - s.entered_ts >= self.min_dwell_ms:
            out.append(s.pending)
            s.pending = None
        return out

    def _update(self, box: TrackBox) -> list[ZoneEvent]:
        s = self._s.setdefault(box.track_id, _State())
        s.last_ts, s.last_frame = box.ts_ms, box.frame
        pt = _anchor(box, self.anchor)
        s.origin = s.origin or pt
        s.moved = s.moved or math.dist(pt, s.origin) >= self.min_travel_px
        now = point_in_polygon(pt, self.polygon)
        if now != s.inside and distance_to_edge(pt, self.polygon) < self.margin_px:
            now = s.inside  # too close to the edge to count as evidence for the other side

        if now == s.inside:
            s.streak = 0
            return []

        s.streak += 1
        if now and s.streak >= self.enter_frames:
            if s.last_exit_ts is not None and box.ts_ms - s.last_exit_ts < self.cooldown_ms:
                return []  # still cooling down; keep streak so it commits once allowed
            if not s.moved:
                return []  # has not moved yet; keep streak so it commits once it has
            s.inside, s.streak, s.entered_ts = True, 0, box.ts_ms
            # Hold the enter back (see min_dwell_ms and enter_after_dwell), so short visits emit nothing.
            s.pending = ZoneEvent("enter", box.track_id, box.ts_ms, box.frame)
            return []

        if not now and s.streak >= self.exit_frames:
            return self._close(box.track_id, s, box.ts_ms, box.frame)
        return []

    def open_visits(self, now_ms: int, seen_within_ms: int | None = None) -> list[tuple[int, float]]:
        """(track_id, seconds inside so far) for every track currently committed as inside. seen_within_ms keeps only
        tracks seen that recently: a live occupancy gauge that does not wait lost_ms for a track that has gone quiet."""
        return [(tid, (now_ms - s.entered_ts) / 1000) for tid, s in self._s.items()
                if s.inside and (seen_within_ms is None or now_ms - s.last_ts <= seen_within_ms)]

    def flush(self) -> list[ZoneEvent]:
        """End of stream: release enter events for tracks still inside."""
        out = [s.pending for s in self._s.values() if s.pending]
        for s in self._s.values():
            s.pending = None
        return out  # type: ignore[return-value]


def run(counter, boxes: Iterable[TrackBox]) -> list[ZoneEvent]:
    events: list[ZoneEvent] = []
    for b in boxes:
        events.extend(counter.update(b))
    if hasattr(counter, "flush"):
        events.extend(counter.flush())
    return events
