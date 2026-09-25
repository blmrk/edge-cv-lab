"""Detections: boxes with no identity. One JSON object per line.

{"frame": 12, "ts_ms": 1200, "bbox": [x1, y1, x2, y2], "score": 0.91, "cls": "car"}

Saving detections separately from tracks is what makes tracker comparisons fair: every tracker
sees exactly the same boxes, and the detector (the slow part) runs once.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Detection:
    frame: int
    ts_ms: int
    bbox: tuple[float, float, float, float]
    score: float = 1.0
    cls: str = "object"

    @property
    def footpoint(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


def read_detections(path: str | Path) -> list[Detection]:
    out = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                out.append(Detection(int(d["frame"]), int(d["ts_ms"]), tuple(float(v) for v in d["bbox"]),
                                     float(d.get("score", 1.0)), str(d.get("cls", "object"))))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{n}: bad detection record: {exc}") from exc
    return out


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


def is_video_file(src: str | Path) -> bool:
    """True only for a single existing video file. A directory, glob or .txt list is frames, which carry no fps."""
    return Path(src).is_file() and Path(src).suffix.lower() in VIDEO_SUFFIXES


def frame_from_filename(path: str | Path) -> int | None:
    """0-based frame from a 1-based numbered image name: img00001.jpg -> 0, 000123.jpg -> 122.
    None if the stem does not end in digits. Numbering by file name, not by read order, keeps
    detections aligned with ground truth when a frame in the middle is unreadable and skipped."""
    m = re.search(r"(\d+)$", Path(path).stem)
    return int(m.group(1)) - 1 if m else None


def frames(dets: list[Detection]) -> Iterator[tuple[int, int, list[Detection]]]:
    """Yields (frame, ts_ms, detections) for EVERY frame in range, including empty ones.
    Trackers must see empty frames, otherwise track ageing is wrong."""
    by = defaultdict(list)
    for d in dets:
        by[d.frame].append(d)
    if not by:
        return
    first, last = min(by), max(by)
    ts0, ts1 = by[first][0].ts_ms, by[last][0].ts_ms
    step = (ts1 - ts0) / max(last - first, 1)
    for f in range(first, last + 1):
        yield f, (by[f][0].ts_ms if by[f] else int(ts0 + (f - first) * step)), by[f]
