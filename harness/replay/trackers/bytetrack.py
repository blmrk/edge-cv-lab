# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""ByteTrack as Ultralytics runs it inside model.track() (its BYTETracker, bytetrack.yaml defaults), fed saved
detections instead of video: detection filters can sit in front of it, and a variant takes seconds, not a video run.
Needs ultralytics (pip install -e '.[video]', or run it in the edge image); written against BYTETracker(args).

python -m replay.track --dets runs/dets.agnostic.jsonl --tracker bytetrack --param new_track_thresh=0.4 --out ...

Params override bytetrack.yaml keys. Saved detections are rounded (0.1 px, 3 decimals), so the tracks come close to a
video run's without matching it box for box. ByteTrack associates regardless of class, so every box goes in as class 0
and a track takes its class name from the detection it matched. Track IDs come from a counter shared by every
BYTETracker in the process: run one at a time, as run_tracker does.
"""
from __future__ import annotations

from ..detections import Detection
from ..schema import TrackBox
from . import register


@register("bytetrack")
def make(**params):
    try:
        import numpy as np
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
    except ImportError as exc:
        raise ImportError("pip install -e '.[video]'  (the bytetrack tracker runs Ultralytics' BYTETracker)") from exc
    cfg = YAML.load(check_yaml("bytetrack.yaml"))
    if unknown := sorted(set(params) - set(cfg)):
        raise TypeError(f"bytetrack: unknown bytetrack.yaml keys {unknown}; known: {sorted(cfg)}")
    impl = BYTETracker(IterableSimpleNamespace(**{**cfg, **params}))

    class _Adapter:
        def update(self, frame: int, ts_ms: int, dets: list[Detection]) -> list[TrackBox]:
            arr = np.array([[*d.bbox, d.score, 0] for d in dets], dtype=np.float32).reshape(-1, 6)
            rows = impl.update(Boxes(arr, (1, 1)))  # orig_shape only feeds the normalised views, unused here
            # rows: x1, y1, x2, y2, track id, score, class, index of the matched detection
            return [TrackBox(frame, ts_ms, int(r[4]), tuple(float(v) for v in r[:4]), float(r[5]), dets[int(r[7])].cls)
                    for r in rows]
    return _Adapter()
