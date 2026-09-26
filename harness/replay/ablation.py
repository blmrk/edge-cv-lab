"""Counter ablation: add the zone-logic fixes one at a time on the same tracks, so each step's effect is measured alone.

python -m replay.ablation --tracks runs/bytetrack.jsonl --zone zone.json --truth truth.json [--tolerance-ms 2000]
python -m replay.ablation --tracks fixtures/boundary_jitter.jsonl --zone fixtures/zone.json --expected 1

Steps are cumulative and follow the case study's "Fixes" table. The last step is DebouncedZoneCounter's defaults.
Lost-track closing (lost_ms) is part of every debounced step.
"""
from __future__ import annotations

import argparse
import json

from .schema import read_tracks
from .score import match
from .zones import DebouncedZoneCounter, NaiveZoneCounter, run

no_dwell = {"min_dwell_ms": 0, "cooldown_ms": 0}
STEPS = [
    ("baseline (naive, centroid)", lambda p: NaiveZoneCounter(p, "centroid")),
    ("+ footpoint anchor", lambda p: NaiveZoneCounter(p, "footpoint")),
    ("+ hysteresis (5 in / 8 out)", lambda p: DebouncedZoneCounter(p, "footpoint", 5, 8, margin_px=0, **no_dwell)),
    ("+ edge margin (10 px)", lambda p: DebouncedZoneCounter(p, "footpoint", 5, 8, margin_px=10, **no_dwell)),
    ("+ min dwell 1 s, cooldown 1.5 s", lambda p: DebouncedZoneCounter(p)),
]


def ablate(boxes, poly, truth_ms: list[int] | None = None, tol_ms: int = 2000) -> list[dict]:
    """One row per step: enters, plus visit matching against truth_ms when given."""
    rows = []
    for name, make in STEPS:
        enters = [e.ts_ms for e in run(make(poly), boxes) if e.kind == "enter"]
        row = {"step": name, "enters": len(enters)}
        if truth_ms is not None:
            row.update(match(enters, truth_ms, tol_ms))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--truth", help="truth.json from tools/label.html (enters_ms)")
    ap.add_argument("--expected", type=int, help="visit count only, when there is no truth.json")
    ap.add_argument("--tolerance-ms", type=int, default=2000)
    a = ap.parse_args()
    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    boxes = sorted(read_tracks(a.tracks), key=lambda b: (b.frame, b.track_id))
    truth = json.load(open(a.truth))["enters_ms"] if a.truth else None
    expected = len(truth) if truth is not None else a.expected
    rows = ablate(boxes, poly, truth, a.tolerance_ms)
    cols = ["enters"] + (["enter_error_pct"] if expected else []) + (
        ["matched", "false_visits", "missed_visits", "f1"] if truth is not None else [])
    print("| step | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in rows:
        if expected:
            r["enter_error_pct"] = f"{(r['enters'] - expected) / expected * 100:+.1f}%"
        print(f"| {r['step']} | " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
