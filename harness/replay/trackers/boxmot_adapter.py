"""Adapter for the `boxmot` package (pip install boxmot): ByteTrack, OC-SORT, BoT-SORT and others
behind one update(dets, img) call. UNTESTED in this repo's CI because boxmot pulls in torch.
Check the class names against the boxmot version you install; they have been renamed before.

Motion-only trackers ignore the image, so a blank frame is passed. Appearance-based trackers
(BoT-SORT with ReID, StrongSORT, DeepOCSORT) need real frames and cannot run from detections alone.
"""
from __future__ import annotations

from ..detections import Detection
from ..schema import TrackBox
from . import register

_MOTION_ONLY = {"bytetrack": "ByteTrack", "ocsort": "OcSort"}


def _factory(cls_name: str):
    def make(frame_size=(1280, 720), **params):
        try:
            import boxmot
            import numpy as np
        except ImportError as exc:
            raise ImportError("pip install boxmot  (needed for boxmot_* trackers)") from exc
        impl = getattr(boxmot, cls_name)(**params)
        blank = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        class _Adapter:
            def update(self, frame: int, ts_ms: int, dets: list[Detection]) -> list[TrackBox]:
                arr = np.array([[*d.bbox, d.score, 0] for d in dets], dtype=float).reshape(-1, 6)
                rows = impl.update(arr, blank)  # rows: x1, y1, x2, y2, id, conf, cls, det_index
                return [TrackBox(frame, ts_ms, int(r[4]), tuple(float(v) for v in r[:4]), float(r[5]),
                                 dets[int(r[7])].cls if len(r) > 7 and 0 <= int(r[7]) < len(dets) else "object")
                        for r in rows]
        return _Adapter()
    return make


for _name, _cls in _MOTION_ONLY.items():
    register(f"boxmot_{_name}")(_factory(_cls))
