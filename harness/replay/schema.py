"""Track record schema. One JSON object per line (JSONL), one line per box per frame.

{"frame": 12, "ts_ms": 400, "track_id": 3, "bbox": [x1, y1, x2, y2], "score": 0.91, "cls": "car"}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TrackBox:
    frame: int
    ts_ms: int  # integer milliseconds. Never floats: float timestamps break exact matching.
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float = 1.0
    cls: str = "object"

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def footpoint(self) -> tuple[float, float]:
        """Bottom-centre of the box: the best cheap proxy for where the object touches the ground."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


def read_tracks(path: str | Path) -> Iterator[TrackBox]:
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield TrackBox(
                    frame=int(d["frame"]),
                    ts_ms=int(d["ts_ms"]),
                    track_id=int(d["track_id"]),
                    bbox=tuple(float(v) for v in d["bbox"]),  # type: ignore[arg-type]
                    score=float(d.get("score", 1.0)),
                    cls=str(d.get("cls", "object")),
                )
            except (KeyError, ValueError, TypeError) as exc:
                # Fail loud. Silently skipping bad rows is how datasets quietly rot.
                raise ValueError(f"{path}:{n}: bad track record: {exc}") from exc
