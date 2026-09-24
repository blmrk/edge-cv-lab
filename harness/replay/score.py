"""Score counters against hand-labelled ground truth (tools/label.html -> truth.json).

Counts alone can hide errors that cancel out (one missed visit + one double count = "perfect").
So each predicted enter is matched one-to-one to a labelled enter within a time tolerance.

python -m replay.score --tracks runs/bytetrack.jsonl --zone zone.json --truth truth.json [--tolerance-ms 2000]
"""
from __future__ import annotations

import argparse
import json

from .schema import read_tracks
from .zones import DebouncedZoneCounter, NaiveZoneCounter, run


def match(pred_ms: list[int], true_ms: list[int], tol_ms: int) -> dict:
    """Greedy in time order. Both lists sorted. Each label matches at most one prediction."""
    pred, true = sorted(pred_ms), sorted(true_ms)
    i = j = tp = 0
    while i < len(pred) and j < len(true):
        d = pred[i] - true[j]
        if abs(d) <= tol_ms:
            tp, i, j = tp + 1, i + 1, j + 1
        elif d < 0:
            i += 1  # prediction with no label nearby: false positive
        else:
            j += 1  # label with no prediction nearby: miss
    fp, fn = len(pred) - tp, len(true) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"predicted": len(pred), "labelled": len(true), "matched": tp, "false_visits": fp, "missed_visits": fn,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "count_error_pct": round(100 * (len(pred) - len(true)) / max(len(true), 1), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tolerance-ms", type=int, default=2000,
                    help="labelling reaction time + debounce delay; 2 s is generous but safe for sparse traffic")
    a = ap.parse_args()

    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    truth = json.load(open(a.truth))["enters_ms"]
    boxes = sorted(read_tracks(a.tracks), key=lambda b: (b.frame, b.track_id))
    out = {}
    for name, counter in {"naive_centroid": NaiveZoneCounter(poly, "centroid"),
                          "debounced_footpoint": DebouncedZoneCounter(poly)}.items():
        enters = [e.ts_ms for e in run(counter, boxes) if e.kind == "enter"]
        out[name] = match(enters, truth, a.tolerance_ms)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
