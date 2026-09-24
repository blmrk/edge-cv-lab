"""Footpoint tracker: associate on where objects touch the ground, not on box overlap.

A small teaching implementation of three published ideas. It is NOT a reimplementation of either paper:
  - Ground-plane association on the bottom-centre of the box, via an optional homography
    (UCMCTrack, Yi et al., AAAI 2024, arXiv:2312.08952).
  - A scene prior: vehicles move along the lane, so the match gate is long along the lane direction
    and narrow across it (road-structure priors, FastTracker, arXiv:2508.14370).
  - Velocity is damped while a track is unobserved, so an occluded track waits near where it
    vanished instead of coasting away (occlusion handling, same paper).

For benchmark-grade results run the authors' code through replay/motformat.py.
"""
from __future__ import annotations

import math

from ..detections import Detection
from ..schema import TrackBox
from . import register


def _project(h, p):
    if h is None:
        return p
    x, y = p
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    return ((h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w)


@register("groundplane")
class GroundPlaneTracker:
    def __init__(self, lane_dir=(1.0, 0.0), gate_along: float = 120.0, gate_across: float = 40.0,
                 gate_growth: float = 40.0, gate_max: float = 320.0, damping: float = 0.7,
                 max_age: int = 100, min_score: float = 0.3, homography=None, alpha: float = 0.6, beta: float = 0.3):
        n = math.hypot(*lane_dir) or 1.0
        self.lane = (lane_dir[0] / n, lane_dir[1] / n)
        self.ga, self.gc, self.growth, self.gmax = gate_along, gate_across, gate_growth, gate_max
        self.damping, self.max_age, self.min_score, self.h = damping, max_age, min_score, homography
        self.alpha, self.beta = alpha, beta
        self._t: dict[int, dict] = {}
        self._next = 1

    def _cost(self, t, p, age) -> float:
        dx, dy = p[0] - t["pred"][0], p[1] - t["pred"][1]
        along = dx * self.lane[0] + dy * self.lane[1]
        across = -dx * self.lane[1] + dy * self.lane[0]
        ga = min(self.ga + self.growth * (age - 1), self.gmax)  # uncertainty grows while unobserved
        return math.hypot(along / ga, across / self.gc)          # < 1 means inside the elliptical gate

    def update(self, frame: int, ts_ms: int, dets: list[Detection]) -> list[TrackBox]:
        dets = [d for d in dets if d.score >= self.min_score]
        pts = [_project(self.h, d.footpoint) for d in dets]
        for t in self._t.values():
            t["pred"] = (t["pos"][0] + t["vel"][0], t["pos"][1] + t["vel"][1])

        pairs = sorted((self._cost(t, p, frame - t["seen"]), tid, i)
                       for tid, t in self._t.items() for i, p in enumerate(pts))
        used_t, used_d, out = set(), set(), []
        for cost, tid, i in pairs:
            if cost >= 1.0:
                break
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid); used_d.add(i)
            t, p = self._t[tid], pts[i]
            gap = frame - t["seen"]
            rx, ry = p[0] - t["pred"][0], p[1] - t["pred"][1]
            t["pos"] = (t["pred"][0] + self.alpha * rx, t["pred"][1] + self.alpha * ry)
            t["vel"] = (t["vel"][0] + self.beta * rx / gap, t["vel"][1] + self.beta * ry / gap)
            t["seen"] = frame
            out.append(TrackBox(frame, ts_ms, tid, dets[i].bbox, dets[i].score, dets[i].cls))

        for tid, t in self._t.items():
            if tid not in used_t:                       # unobserved: hold position, bleed off speed
                t["pos"], t["vel"] = t["pred"], (t["vel"][0] * self.damping, t["vel"][1] * self.damping)
        for i, d in enumerate(dets):
            if i not in used_d:
                tid, self._next = self._next, self._next + 1
                self._t[tid] = {"pos": pts[i], "pred": pts[i], "vel": (0.0, 0.0), "seen": frame}
                out.append(TrackBox(frame, ts_ms, tid, d.bbox, d.score, d.cls))
        self._t = {k: t for k, t in self._t.items() if frame - t["seen"] <= self.max_age}
        return out
