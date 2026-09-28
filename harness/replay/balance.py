"""Zone balance: do enters and exits reconcile, and what does a standing imbalance say?

python -m replay.balance --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --gt runs/mtid.gt.jsonl [--explain]

Enters minus exits, counted up to a moment, is the counter's occupancy: the vehicles it says are inside. Two views:
  after the fact: every event placed at its own timestamp, as a store of events sees it once the stream is over
  live: every event placed when the counter emitted it, as a dashboard sees it at that moment
They differ because the debounced counter holds each enter until its visit closes. With annotated tracks (--gt), both are
scored frame by frame against true occupancy: the annotated vehicles whose footpoint is inside the zone.
Each visit still open at the end gets a reason from its track's last box: inside at end (seen within lost_ms of the end,
anchor inside), lost inside (anchor inside, gone for longer than lost_ms), or left (last seen outside the zone).
One row per counter, from the naive counter to the debounced one, so each step's effect on the balance shows.
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from itertools import accumulate

from .geometry import point_in_polygon
from .schema import read_tracks
from .zones import DebouncedZoneCounter, NaiveZoneCounter, _anchor

LOST_MS = 3000  # DebouncedZoneCounter's default lost_ms: the line between "inside at end" and "lost inside"

COUNTERS = [  # (name, anchor, factory)
    ("naive, centroid", "centroid", lambda p: NaiveZoneCounter(p, "centroid")),
    ("naive, footpoint", "footpoint", lambda p: NaiveZoneCounter(p, "footpoint")),
    ("debounced, no lost-track close", "footpoint", lambda p: DebouncedZoneCounter(p, lost_ms=10**12)),
    ("debounced", "footpoint", lambda p: DebouncedZoneCounter(p)),
    ("debounced + zone rule 30 px", "footpoint", lambda p: DebouncedZoneCounter(p, min_travel_px=30)),
    ("debounced + zone rule 30 px + enter after dwell", "footpoint",
     lambda p: DebouncedZoneCounter(p, min_travel_px=30, enter_after_dwell=True)),
]


def emitted(counter, boxes) -> list[tuple[int, object]]:
    """(emitted at, event) for every event, in emission order: an event is emitted at the time of the box whose update()
    returned it, and flush() emits at the last box's time. boxes: sorted by (frame, track_id)."""
    out, last = [], 0
    for b in boxes:
        out += [(b.ts_ms, e) for e in counter.update(b)]
        last = b.ts_ms
    if hasattr(counter, "flush"):
        out += [(last, e) for e in counter.flush()]
    return out


def occupancy(timed_events, frame_ts) -> dict[int, int]:
    """Enters minus exits placed at or before each frame's time. timed_events: (time, event); frame_ts: (frame, ts)."""
    steps = sorted((t, 1 if e.kind == "enter" else -1) for t, e in timed_events)
    times, running = [t for t, _ in steps], list(accumulate(d for _, d in steps))
    out = {}
    for f, ts in frame_ts:
        i = bisect.bisect_right(times, ts)
        out[f] = running[i - 1] if i else 0
    return out


def true_occupancy(gt_boxes, poly, anchor="footpoint") -> dict[int, int]:
    """Annotated vehicles whose anchor is inside the zone, per annotated frame."""
    out: dict[int, int] = defaultdict(int)
    for b in gt_boxes:
        out[b.frame] += point_in_polygon(_anchor(b, anchor), poly)
    return dict(out)


def open_visits(events, boxes, poly, anchor, lost_ms: int = LOST_MS) -> list[dict]:
    """One row per enter with no later exit on its track, with why it is open (see the module docstring)."""
    last, end = {}, max(b.ts_ms for b in boxes)
    for b in boxes:
        last[b.track_id] = b
    state: dict[int, object] = {}
    for e in sorted(events, key=lambda e: (e.ts_ms, e.kind == "exit")):
        state[e.track_id] = e if e.kind == "enter" else None
    rows = []
    for tid, e in sorted(state.items()):
        if e is None:
            continue
        b = last[tid]
        inside = point_in_polygon(_anchor(b, anchor), poly)
        reason = "left" if not inside else "inside at end" if end - b.ts_ms <= lost_ms else "lost inside"
        rows.append({"track_id": tid, "enter_ms": e.ts_ms, "last_seen_ms": b.ts_ms, "reason": reason})
    return rows


def table(boxes, poly, truth=None) -> list[dict]:
    """One row per counter in COUNTERS. truth: true_occupancy() output, or None to skip the occupancy scores."""
    boxes = sorted(boxes, key=lambda b: (b.frame, b.track_id))
    frame_ts = sorted({(b.frame, b.ts_ms) for b in boxes})
    rows = []
    for name, anchor, make in COUNTERS:
        live = emitted(make(poly), boxes)
        events = [e for _, e in live]
        enters, exits = sum(e.kind == "enter" for e in events), sum(e.kind == "exit" for e in events)
        reasons = [r["reason"] for r in open_visits(events, boxes, poly, anchor)]
        row = {"counter": name, "enters": enters, "exits": exits, "balance": enters - exits,
               "open_inside_at_end": reasons.count("inside at end"), "open_lost_inside": reasons.count("lost inside"),
               "open_left": reasons.count("left")}
        if truth is not None:  # only the annotated frame range: past it nothing is known, not "empty"
            lo, hi = min(truth), max(truth)
            scored = [(f, truth.get(f, 0)) for f, _ in frame_ts if lo <= f <= hi]
            after = occupancy([(e.ts_ms, e) for e in events], frame_ts)
            now = occupancy(live, frame_ts)
            errs = [abs(after[f] - t) for f, t in scored]
            row.update(occupancy_mae=round(sum(errs) / len(errs), 3), occupancy_max_error=max(errs),
                       live_mae=round(sum(abs(now[f] - t) for f, t in scored) / len(scored), 3),
                       end_counted=after[scored[-1][0]], end_true=scored[-1][1])
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--gt", help="annotated tracks JSONL (e.g. mtid_to_gt.py output): scores occupancy against them")
    ap.add_argument("--explain", action="store_true", help="also list the debounced counter's open visits")
    a = ap.parse_args()
    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    boxes = sorted(read_tracks(a.tracks), key=lambda b: (b.frame, b.track_id))
    truth = true_occupancy(read_tracks(a.gt), poly) if a.gt else None
    rows = table(boxes, poly, truth)
    cols = [c for c in rows[0] if c != "counter"]
    print("| counter | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in rows:
        print(f"| {r['counter']} | " + " | ".join(str(r[c]) for c in cols) + " |")
    if a.explain:
        events = [e for _, e in emitted(DebouncedZoneCounter(poly), boxes)]
        print("\ndebounced counter, open visits:")
        for r in open_visits(events, boxes, poly, "footpoint"):
            print(r)


if __name__ == "__main__":
    main()
