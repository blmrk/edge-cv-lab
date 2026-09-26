"""Score counters against hand-labelled ground truth (tools/label.html -> truth.json).

Counts alone can hide errors that cancel out (one missed visit + one double count = "perfect").
So each predicted enter is matched one-to-one to a labelled enter within a time tolerance.

python -m replay.score --tracks runs/bytetrack.jsonl --zone zone.json --truth truth.json [--tolerance-ms 2000]
python -m replay.score ... --explain   one row per debounced enter: matched or false, and how far its track moved
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from .schema import read_tracks
from .zones import DebouncedZoneCounter, NaiveZoneCounter, run


def _matched(pred: list[int], true: list[int], tol_ms: int) -> set[int]:
    """Indices of the sorted predictions that match a label. Greedy in time order; each label matches at most once."""
    i = j = 0
    hits = set()
    while i < len(pred) and j < len(true):
        d = pred[i] - true[j]
        if abs(d) <= tol_ms:
            hits.add(i)
            i, j = i + 1, j + 1
        elif d < 0:
            i += 1  # prediction with no label nearby: false positive
        else:
            j += 1  # label with no prediction nearby: miss
    return hits


def match(pred_ms: list[int], true_ms: list[int], tol_ms: int) -> dict:
    pred, true = sorted(pred_ms), sorted(true_ms)
    tp = len(_matched(pred, true, tol_ms))
    fp, fn = len(pred) - tp, len(true) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"predicted": len(pred), "labelled": len(true), "matched": tp, "false_visits": fp, "missed_visits": fn,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "count_error_pct": round(100 * (len(pred) - len(true)) / max(len(true), 1), 1)}


def explain(boxes, poly, truth_ms: list[int], tol_ms: int = 2000) -> list[dict]:
    """One row per debounced enter, in time order: matched to a label or not, and what its track looked like.
    travel_px is the diagonal of the box around every footpoint the track had: near 0 means it never moved."""
    by_id = defaultdict(list)
    for b in boxes:
        by_id[b.track_id].append(b)
    enters = sorted((e for e in run(DebouncedZoneCounter(poly), boxes) if e.kind == "enter"), key=lambda e: e.ts_ms)
    hits = _matched([e.ts_ms for e in enters], sorted(truth_ms), tol_ms)
    rows = []
    for k, e in enumerate(enters):
        tb = by_id[e.track_id]
        xs, ys = [b.footpoint[0] for b in tb], [b.footpoint[1] for b in tb]
        rows.append({"ts_ms": e.ts_ms, "track_id": e.track_id, "matched": k in hits,
                     "travel_px": round(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1),
                     "w": round(sum(b.bbox[2] - b.bbox[0] for b in tb) / len(tb), 1),
                     "h": round(sum(b.bbox[3] - b.bbox[1] for b in tb) / len(tb), 1),
                     "score": round(sum(b.score for b in tb) / len(tb), 3), "boxes": len(tb)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tolerance-ms", type=int, default=2000,
                    help="labelling reaction time + debounce delay; 2 s is generous but safe for sparse traffic")
    ap.add_argument("--explain", action="store_true", help="table of the debounced counter's enters, least travel first")
    a = ap.parse_args()

    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    truth = json.load(open(a.truth))["enters_ms"]
    boxes = sorted(read_tracks(a.tracks), key=lambda b: (b.frame, b.track_id))
    if a.explain:
        cols = ["ts_ms", "track_id", "matched", "travel_px", "w", "h", "score", "boxes"]
        print("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
        print("|" + "---|" * len(cols))
        for r in sorted(explain(boxes, poly, truth, a.tolerance_ms), key=lambda r: (r["travel_px"], r["ts_ms"])):
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return
    out = {}
    for name, counter in {"naive_centroid": NaiveZoneCounter(poly, "centroid"),
                          "debounced_footpoint": DebouncedZoneCounter(poly)}.items():
        enters = [e.ts_ms for e in run(counter, boxes) if e.kind == "enter"]
        out[name] = match(enters, truth, a.tolerance_ms)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
