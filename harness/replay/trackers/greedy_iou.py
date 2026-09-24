"""Baseline: greedy IoU association against each track's last box. No motion model.

Deliberately simple. It stands in for the family of image-plane IoU trackers and fails the same
way they do when a box jumps further than its own width between frames.
"""
from __future__ import annotations

from ..detections import Detection
from ..schema import TrackBox
from . import register


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@register("greedy_iou")
class GreedyIoUTracker:
    def __init__(self, iou_min: float = 0.3, max_age: int = 30, min_score: float = 0.3):
        self.iou_min, self.max_age, self.min_score = iou_min, max_age, min_score
        self._tracks: dict[int, dict] = {}
        self._next = 1

    def update(self, frame: int, ts_ms: int, dets: list[Detection]) -> list[TrackBox]:
        dets = [d for d in dets if d.score >= self.min_score]
        pairs = sorted(((iou(t["bbox"], d.bbox), tid, i) for tid, t in self._tracks.items()
                        for i, d in enumerate(dets)), reverse=True)
        used_t, used_d, out = set(), set(), []
        for score, tid, i in pairs:
            if score < self.iou_min:
                break
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid); used_d.add(i)
            self._tracks[tid] = {"bbox": dets[i].bbox, "seen": frame}
            out.append(TrackBox(frame, ts_ms, tid, dets[i].bbox, dets[i].score, dets[i].cls))
        for i, d in enumerate(dets):
            if i not in used_d:
                tid, self._next = self._next, self._next + 1
                self._tracks[tid] = {"bbox": d.bbox, "seen": frame}
                out.append(TrackBox(frame, ts_ms, tid, d.bbox, d.score, d.cls))
        self._tracks = {k: t for k, t in self._tracks.items() if frame - t["seen"] <= self.max_age}
        return out
