"""Phantom boxes: each fix against the same ByteTrack baseline, one row each, so every fix is measured on its own.

docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge python -m replay.phantoms \
    --dets runs/dets.agnostic.jsonl --zone zone.json --truth truth.json --out-dir runs/phantoms
Runs in the edge image: the bytetrack tracker needs ultralytics. --out-dir gets each step's tracks, for
scripts/trackeval_run.py.

Columns: the debounced counter's enters, split into static and moving as `replay.score --explain` splits them; visits
matched against the labels, for all enters (f1) and for the moving enters alone (moving f1); and the queue fixture,
whose six cars each stop in the zone (fixtures/queue.*): how many of them the step still counts, and its false visits.
The queue runs on groundplane, not ByteTrack: ByteTrack hands one ID from car to car at the queue's window and counts
1 of 6 even at its defaults, so only a tracker that counts all six shows what a fix takes away. Steps that change a
ByteTrack setting have no groundplane equivalent and show n/a there.
Steps are alternatives, not cumulative: each changes one thing against the baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detections import read_detections
from .motformat import write_tracks
from .score import explain, match
from .trackers import create, run_tracker
from .trackers.greedy_iou import iou
from .zones import DebouncedZoneCounter, run

FX = Path(__file__).resolve().parent.parent / "fixtures"


def min_area(dets, px2: float):
    return [d for d in dets if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) >= px2]


def static_boxes(dets, min_share: float, iou_min: float = 0.5):
    """Boxes that sit still. Each detection joins the first box seen that it overlaps at IoU >= iou_min, or starts a
    new one; a box is static when detections join it in at least min_share of the frames `dets` span."""
    if not dets:
        return []
    # ponytail: scans every box so far for each detection; 0.3 s per half of the 3199-frame clip. Grid it if it grows.
    seen: list[tuple[tuple, set]] = []
    for d in dets:
        hit = next((s for s in seen if iou(s[0], d.bbox) >= iou_min), None)
        if hit:
            hit[1].add(d.frame)
        else:
            seen.append((d.bbox, {d.frame}))
    span = max(d.frame for d in dets) - min(d.frame for d in dets) + 1
    return [box for box, frames in seen if len(frames) >= min_share * span]


def static_mask(dets, min_share: float, iou_min: float = 0.5):
    """Drops detections overlapping a static box at IoU >= iou_min, cross-fitted: the first half of the clip is masked
    with boxes learned from the second half and the second half with boxes learned from the first, so no frame is
    masked by boxes it helped learn. Learns from detections only, never from labels."""
    if not dets:
        return []
    mid = (min(d.frame for d in dets) + max(d.frame for d in dets) + 1) // 2
    halves = [d for d in dets if d.frame < mid], [d for d in dets if d.frame >= mid]
    kept = []
    for this, other in (halves, halves[::-1]):
        static = static_boxes(other, min_share, iou_min)
        kept += [d for d in this if all(iou(d.bbox, s) < iou_min for s in static)]
    return kept


STEPS = [  # (key, step, bytetrack params, detection filter, counter params)
    ("baseline", "baseline: ByteTrack defaults", {}, None, {}),
    ("travel30", "zone rule: moved 30 px before an enter", {}, None, {"min_travel_px": 30}),
    ("floor030", "detections under score 0.3 dropped", {}, lambda ds: [d for d in ds if d.score >= 0.3], {}),
    ("birth040", "track birth score 0.4 (default 0.25)", {"new_track_thresh": 0.4}, None, {}),
    ("birth050", "track birth score 0.5", {"new_track_thresh": 0.5}, None, {}),
    ("area1024", "boxes under 32 x 32 px dropped", {}, lambda ds: min_area(ds, 32 * 32), {}),
    ("mask025", "static-box mask, boxes in 25% of frames", {}, lambda ds: static_mask(ds, 0.25), {}),
    ("mask010", "static-box mask, boxes in 10% of frames", {}, lambda ds: static_mask(ds, 0.10), {}),
]


def _track(dets, params, keep, tracker="bytetrack"):
    boxes = run_tracker(create(tracker, **params), keep(dets) if keep else dets)
    return sorted(boxes, key=lambda b: (b.frame, b.track_id))


def table(dets, poly, truth_ms, queue=None, tol_ms: int = 2000, steps=None):
    """One row per step, plus each step's tracks by key. queue: (dets, polygon, enters_ms) scored the same way.
    steps: rows shaped like STEPS (None: STEPS)."""
    rows, tracks = [], {}
    for key, step, params, keep, counter in STEPS if steps is None else steps:
        tracks[key] = _track(dets, params, keep)
        found, s = explain(tracks[key], poly, truth_ms, tol_ms, **counter)
        m = match([r["ts_ms"] for r in found], truth_ms, tol_ms)
        row = {"step": step, "enters": len(found), "static": s["static"], "moving": s["moving"],
               **{k: m[k] for k in ("matched", "false_visits", "missed_visits", "f1")},
               "moving_f1": s["moving_vs_labels"]["f1"]}
        if queue and params:
            row.update(queue_matched="n/a", queue_false="n/a")
        elif queue:
            q_dets, q_poly, q_truth = queue
            q_tracks = _track(q_dets, {}, keep, "groundplane")
            q = match([e.ts_ms for e in run(DebouncedZoneCounter(q_poly, **counter), q_tracks) if e.kind == "enter"],
                      q_truth, tol_ms)
            row.update(queue_matched=q["matched"], queue_false=q["false_visits"])
        rows.append(row)
    return rows, tracks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dets", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--truth", required=True, help="truth.json from tools/label.html (enters_ms)")
    ap.add_argument("--out-dir", help="write each step's tracks here as <key>.jsonl")
    ap.add_argument("--tolerance-ms", type=int, default=2000)
    a = ap.parse_args()
    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    queue = (read_detections(FX / "queue.dets.jsonl"), [tuple(p) for p in json.load(open(FX / "zone.json"))["polygon"]],
             json.load(open(FX / "queue.truth.json"))["enters_ms"])
    rows, tracks = table(read_detections(a.dets), poly, json.load(open(a.truth))["enters_ms"], queue, a.tolerance_ms)
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        for key, boxes in tracks.items():
            write_tracks(boxes, Path(a.out_dir) / f"{key}.jsonl")
    cols = [c for c in rows[0] if c != "step"]
    print("| step | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in rows:
        print(f"| {r['step']} | " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
