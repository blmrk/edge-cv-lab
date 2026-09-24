"""Seeded synthetic traffic. Used for fixtures, visuals and the no-video simulator.

Scene (1280x720): lane A runs through the zone, lane B runs just below it. Vehicles in lane A
may stop inside the zone (a real visit), idle on its left edge with box jitter, or drive straight
through. Lane B vehicles never visit, but some carry a shadow that stretches their box upward.
"""
from __future__ import annotations

import random
from typing import Iterator

from .schema import TrackBox

FRAME_W, FRAME_H, FPS = 1280, 720, 30
ZONE = [(400, 200), (800, 200), (800, 600), (400, 600)]
LANE_A_FOOT, LANE_B_FOOT = 430, 660
BOX_W, BOX_H = 90, 60


def _vehicle(rnd: random.Random, lane: str) -> tuple[list[tuple[float, float, float]], bool]:
    """Returns ([(cx, foot_y, extra_height)], is_true_visit)."""
    foot = LANE_A_FOOT if lane == "A" else LANE_B_FOOT
    speed = rnd.uniform(5, 8)
    pts: list[tuple[float, float, float]] = []
    x, visit = -BOX_W, False
    behaviour = rnd.choices(["through", "stop", "edge_idle"], [0.35, 0.45, 0.2])[0] if lane == "A" else "through"
    shadow = lane == "B" and rnd.random() < 0.5

    def drive_to(target):
        nonlocal x
        while x < target:
            x += speed
            extra = 130 if shadow and 380 < x < 820 else 0
            pts.append((x + rnd.uniform(-1, 1), foot + rnd.uniform(-1, 1), extra))

    if behaviour == "edge_idle":
        drive_to(398)
        for _ in range(rnd.randint(90, 240)):  # 3-8 s hovering on the boundary
            pts.append((400 + rnd.uniform(-6, 6), foot + rnd.uniform(-2, 2), 0))
    if behaviour in ("stop", "edge_idle"):
        stop_x = rnd.uniform(520, 700)
        drive_to(stop_x)
        for _ in range(rnd.randint(60, 300)):  # 2-10 s served inside the zone
            pts.append((stop_x + rnd.uniform(-2, 2), foot + rnd.uniform(-2, 2), 0))
        visit = True
    drive_to(FRAME_W + BOX_W)
    if behaviour == "through":
        visit = lane == "A" and (400 / speed) / FPS >= 1.0  # crossing the zone takes >= min dwell
    return pts, visit


def generate_traffic(seed: int = 11, vehicles: int = 24) -> tuple[list[TrackBox], int]:
    """Returns (boxes sorted by frame, ground-truth visit count)."""
    rnd = random.Random(seed)
    boxes: list[TrackBox] = []
    truth, next_free = 0, {"A": 0, "B": 0}
    for tid in range(1, vehicles + 1):
        lane = "A" if rnd.random() < 0.6 else "B"
        pts, visit = _vehicle(rnd, lane)
        truth += visit
        start = next_free[lane]
        next_free[lane] = start + len(pts) - 60 + rnd.randint(30, 120)  # queue behind the previous one
        for i, (cx, foot, extra) in enumerate(pts):
            f = start + i
            boxes.append(TrackBox(f, int(f * 1000 / FPS), tid,
                                  (cx - BOX_W / 2, foot - BOX_H - extra, cx + BOX_W / 2, foot), 0.9, "car"))
    boxes.sort(key=lambda b: (b.frame, b.track_id))
    return boxes, truth


def by_frame(boxes: list[TrackBox]) -> Iterator[tuple[int, list[TrackBox]]]:
    cur, bucket = None, []
    for b in boxes:
        if cur is not None and b.frame != cur:
            yield cur, bucket
            bucket = []
        cur = b.frame
        bucket.append(b)
    if bucket:
        yield cur, bucket


# ---------------------------------------------------------------------------------------------
# Queue scenario: the identity-transfer failure. 10 fps, like a busy edge device.
#
#   cars queue -> pass behind a pillar (no detections) -> stop at a service window -> drive off.
#   The car at the window pulls away hard; motion blur drops its detections for a few frames.
#   The next car then emerges from behind the pillar and stops where the first one was.
#
# An IoU tracker leaves the first car's track parked at the window, and the second car inherits
# that ID: two customers become one long visit.
QUEUE_FPS = 10
PILLAR = (440, 560)   # x-range with no detections
WINDOW_X = 600


def generate_queue(seed: int = 5, cars: int = 6):
    """Returns (detections, ground_truth_tracks, true_visits). Imports kept local to avoid cycles."""
    from .detections import Detection

    rnd = random.Random(seed)
    foot, dets, gt = LANE_A_FOOT, [], []
    window_free_at = 0
    for tid in range(1, cars + 1):
        dwell = rnd.randint(40, 70)                       # 4-7 s at the window
        xs = []
        x = -BOX_W
        while x < 470:                                    # approach, then wait hidden behind the pillar
            x += 18
            xs.append(x)
        approach_len = len(xs)
        start = max(0, window_free_at - approach_len - rnd.randint(25, 45))   # arrive while the lead is served
        wait = max(0, window_free_at + rnd.randint(8, 15) - (start + approach_len))
        xs += [470.0] * wait
        while x < WINDOW_X:                               # pull up to the window
            x = min(WINDOW_X, x + 14)
            xs.append(x)
        xs += [float(WINDOW_X)] * dwell
        depart_at = len(xs)
        v = 0.0
        while x < FRAME_W + BOX_W:                        # pull away: 10, 25, 40, 40 ... px per frame
            v = min(40.0, v + 15.0) if v else 10.0
            x += v
            xs.append(x)
        window_free_at = start + depart_at
        for i, cx in enumerate(xs):
            f = start + i
            ts = int(f * 1000 / QUEUE_FPS)
            jx, jy = rnd.uniform(-1.5, 1.5), rnd.uniform(-1.5, 1.5)
            box = (cx - BOX_W / 2 + jx, foot - BOX_H + jy, cx + BOX_W / 2 + jx, foot + jy)
            gt.append(TrackBox(f, ts, tid, box, 1.0, "car"))
            hidden = PILLAR[0] <= cx <= PILLAR[1]
            blurred = depart_at <= i < depart_at + 3
            if not hidden and not blurred:
                dets.append(Detection(f, ts, box, round(rnd.uniform(0.6, 0.95), 2), "car"))
    dets.sort(key=lambda d: d.frame)
    gt.sort(key=lambda b: (b.frame, b.track_id))
    return dets, gt, cars
