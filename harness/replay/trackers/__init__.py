"""Tracker adapters. A tracker is anything with update() and a name in the registry.

    class MyTracker:
        def __init__(self, **params): ...
        def update(self, frame: int, ts_ms: int, dets: list[Detection]) -> list[TrackBox]: ...

update() is called once per frame in order, including frames with no detections, and returns the
boxes it wants to report for THAT frame with stable track_ids. Register with @register("name").

Three ways to plug a tracker in:
  1. Write it here in pure Python (see greedy_iou.py, groundplane.py).
  2. Wrap a Python package (see boxmot_adapter.py).
  3. Run any research repo on its own and exchange MOTChallenge text files (see replay/motformat.py).
     This needs no adapter code at all and is the right route for FastTracker, UCMCTrack, TrackTrack.
"""
from __future__ import annotations

from typing import Callable, Protocol

from ..detections import Detection, frames
from ..schema import TrackBox


class Tracker(Protocol):
    def update(self, frame: int, ts_ms: int, dets: list[Detection]) -> list[TrackBox]: ...


_REGISTRY: dict[str, Callable[..., Tracker]] = {}


def register(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


def available() -> list[str]:
    _load()
    return sorted(_REGISTRY)


def create(name: str, **params) -> Tracker:
    _load()
    if name not in _REGISTRY:
        raise KeyError(f"unknown tracker {name!r}. available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**params)


def run_tracker(tracker: Tracker, dets: list[Detection]) -> list[TrackBox]:
    out: list[TrackBox] = []
    for f, ts, ds in frames(dets):
        out.extend(tracker.update(f, ts, ds))
    return out


def _load():
    from . import boxmot_adapter, greedy_iou, groundplane  # noqa: F401  (import = register)
