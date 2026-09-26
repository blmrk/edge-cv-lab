"""Detections: boxes with no identity. One JSON object per line.

{"frame": 12, "ts_ms": 1200, "bbox": [x1, y1, x2, y2], "score": 0.91, "cls": "car"}

Saving detections separately from tracks is what makes tracker comparisons fair: every tracker
sees exactly the same boxes, and the detector (the slow part) runs once.
"""
from __future__ import annotations

import json
import math
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


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


def is_image(path: str | Path) -> bool:
    """A file the detector reads as a single image frame (ultralytics' image formats)."""
    return Path(str(path)).suffix.lower() in IMAGE_SUFFIXES


def is_frame_source(src: str | Path) -> bool:
    """A directory, glob, .txt list or image: frames, which carry no fps. Anything else (a video in any
    format, a stream URL, a webcam index) is read as video, numbered in read order at its own fps."""
    s = str(src)
    glob = "://" not in s and any(c in s for c in "*?[")  # a stream URL may carry a '?' query
    return Path(s).is_dir() or glob or s.lower().endswith(".txt") or is_image(s)


def frame_from_filename(path: str | Path) -> int | None:
    """0-based frame from a 1-based numbered image name: img00001.jpg -> 0, 000123.jpg -> 122.
    None if the stem does not end in digits. Numbering by file name, not by read order, keeps
    detections aligned with ground truth when a frame in the middle is unreadable and skipped."""
    m = re.search(r"(\d+)$", Path(path).stem)
    return int(m.group(1)) - 1 if m else None


def frame_window(first: int, last: int, fps: float, start: float = 0.0,
                 seconds: float | None = None) -> tuple[int, int]:
    """Inclusive frame range for a time window of a clip, frame f shown at f / fps seconds.
    Frames at or after `start`, before `start + seconds` (to the end if `seconds` is None),
    clamped to [first, last]. Raises ValueError for a negative or empty window."""
    if start < 0 or (seconds is not None and seconds <= 0):
        raise ValueError(f"bad window: start {start} s, length {seconds} s")
    eps = 1e-6                                              # 0.1 * 30 is 3.0000000000000004, still frame 3
    lo = max(first, math.ceil(start * fps - eps))
    hi = last if seconds is None else min(last, math.ceil((start + seconds) * fps - eps) - 1)
    if lo > hi:
        raise ValueError(f"window {start} s + {seconds} s holds no frame of {first}..{last} at {fps} fps")
    return lo, hi


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
