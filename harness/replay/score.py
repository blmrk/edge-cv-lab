"""Score counters against hand-labelled ground truth (tools/label.html -> truth.json).

Counts alone can hide errors that cancel out (one missed visit + one double count = "perfect").
So each predicted enter is matched one-to-one to a labelled enter within a time tolerance.

python -m replay.score --tracks runs/bytetrack.jsonl --zone zone.json --truth truth.json [--tolerance-ms 2000]
python -m replay.score ... --explain   one row per debounced enter with how far its track moved, then the moving
                                      enters scored against the labels on their own
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

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


def explain(boxes, poly, truth_ms: list[int], tol_ms: int = 2000, static_px: float = 30) -> tuple[list[dict], dict]:
    """One row per debounced enter, in time order, describing its track; plus the moving enters scored on their own.
    travel_px is the diagonal of the box around every footpoint the track had: near 0 means it never moved.
    No per-row matched flag: time alone cannot say whether a label belongs to a parked phantom or to the vehicle
    that entered a moment later, so the moving enters (travel_px >= static_px) are matched against the labels alone."""
    by_id = defaultdict(list)
    for b in boxes:
        by_id[b.track_id].append(b)
    enters = sorted((e for e in run(DebouncedZoneCounter(poly), boxes) if e.kind == "enter"), key=lambda e: e.ts_ms)
    rows = []
    for e in enters:
        tb = by_id[e.track_id]
        xs, ys = [b.footpoint[0] for b in tb], [b.footpoint[1] for b in tb]
        rows.append({"ts_ms": e.ts_ms, "track_id": e.track_id,
                     "travel_px": round(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1),
                     "w": round(sum(b.bbox[2] - b.bbox[0] for b in tb) / len(tb), 1),
                     "h": round(sum(b.bbox[3] - b.bbox[1] for b in tb) / len(tb), 1),
                     "score": round(sum(b.score for b in tb) / len(tb), 3), "boxes": len(tb)})
    moving = [r["ts_ms"] for r in rows if r["travel_px"] >= static_px]
    return rows, {"static": len(rows) - len(moving), "moving": len(moving), "static_px": static_px,
                  "moving_vs_labels": match(moving, truth_ms, tol_ms)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tolerance-ms", type=int, default=2000,
                    help="labelling reaction time + debounce delay; 2 s is generous but safe for sparse traffic")
    ap.add_argument("--explain", action="store_true", help="table of the debounced counter's enters, least travel first")
    ap.add_argument("--static-px", type=float, default=30, help="--explain: a track that moved less than this is static")
    a = ap.parse_args()

    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    truth = json.load(open(a.truth))["enters_ms"]
    boxes = sorted(read_tracks(a.tracks), key=lambda b: (b.frame, b.track_id))
    if a.explain:
        rows, s = explain(boxes, poly, truth, a.tolerance_ms, a.static_px)
        cols = ["ts_ms", "track_id", "travel_px", "w", "h", "score", "boxes"]
        print("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
        print("|" + "---|" * len(cols))
        for r in sorted(rows, key=lambda r: (r["travel_px"], r["ts_ms"])):
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")
        m = s["moving_vs_labels"]
        print(f"\nenters {len(rows)} | static (moved < {a.static_px:g} px) {s['static']} | moving {s['moving']}\n"
              f"moving enters alone vs labels: matched {m['matched']}, false {m['false_visits']}, "
              f"missed {m['missed_visits']}, f1 {m['f1']}")
        return
    out = {}
    for name, counter in {"naive_centroid": NaiveZoneCounter(poly, "centroid"),
                          "debounced_footpoint": DebouncedZoneCounter(poly)}.items():
        enters = [e.ts_ms for e in run(counter, boxes) if e.kind == "enter"]
        out[name] = match(enters, truth, a.tolerance_ms)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
