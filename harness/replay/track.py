"""python -m replay.track --dets runs/dets.jsonl --tracker groundplane --out runs/groundplane.jsonl
Tracker parameters: --param gate_along=150 --param max_age=60"""
from __future__ import annotations

import argparse
import json

from .detections import read_detections
from .motformat import write_tracks
from .trackers import available, create, run_tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tracker", required=True, help=", ".join(available()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=JSON")
    a = ap.parse_args()
    params = {k: json.loads(v) for k, v in (p.split("=", 1) for p in a.param)}
    tracks = run_tracker(create(a.tracker, **params), read_detections(a.dets))
    write_tracks(tracks, a.out)
    print(f"{a.tracker}: {len({t.track_id for t in tracks})} ids, {len(tracks)} boxes -> {a.out}")


if __name__ == "__main__":
    main()
