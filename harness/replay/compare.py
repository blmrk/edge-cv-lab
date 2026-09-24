"""Run several trackers on the same detections and print one comparison table (Markdown).

python -m replay.compare --dets fixtures/queue.dets.jsonl --zone fixtures/zone.json \
    --truth fixtures/queue.truth.json --gt fixtures/queue.gt.jsonl --trackers greedy_iou greedy_iou:max_age=100 groundplane
Add --tracks name=path.jsonl for results imported from an external tracker.
"""
from __future__ import annotations

import argparse
import json

from .detections import read_detections
from .idmetrics import identity_report
from .schema import read_tracks
from .score import match
from .trackers import available, create, run_tracker
from .zones import DebouncedZoneCounter, run


def evaluate(tracks, poly, truth_ms, gt, tol_ms):
    boxes = sorted(tracks, key=lambda b: (b.frame, b.track_id))
    enters = [e.ts_ms for e in run(DebouncedZoneCounter(poly), boxes) if e.kind == "enter"]
    row = match(enters, truth_ms, tol_ms)
    if gt:
        row.update(identity_report(gt, boxes))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--gt", help="optional ground-truth tracks JSONL for identity metrics")
    ap.add_argument("--trackers", nargs="*", default=[], help=f"built-in: {', '.join(available())}")
    ap.add_argument("--tracks", nargs="*", default=[], metavar="NAME=PATH", help="pre-computed tracks")
    ap.add_argument("--tolerance-ms", type=int, default=2000)
    a = ap.parse_args()

    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    truth = json.load(open(a.truth))["enters_ms"]
    gt = list(read_tracks(a.gt)) if a.gt else None
    dets = read_detections(a.dets)

    rows = {}
    for spec in a.trackers:                       # "name" or "name:key=json,key=json"
        name, _, raw = spec.partition(":")
        params = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in raw.split(",") if kv)}
        rows[spec] = evaluate(run_tracker(create(name, **params), dets), poly, truth, gt, a.tolerance_ms)
    for spec in a.tracks:
        name, path = spec.split("=", 1)
        rows[name] = evaluate(list(read_tracks(path)), poly, truth, gt, a.tolerance_ms)

    cols = ["labelled", "predicted", "missed_visits", "false_visits", "f1"] + (
        ["id_switches", "id_transfers", "pred_ids"] if gt else [])
    print("| tracker | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
