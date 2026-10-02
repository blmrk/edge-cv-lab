"""Straddles against annotations: detections across two annotated vehicles side by side, and the visits they make.

python -m replay.straddle --dets runs/clip.dets.jsonl --gt runs/clip.gt.jsonl --ignored runs/clip.ignored.json
python -m replay.straddle ... --tracks runs/clip.bytetrack.jsonl --zone runs/clip.zone.json [--min-travel-px 30]
    also one row per debounced enter whose track holds a straddle, and totals over every enter

replay.between's counterpart for clips with per-frame annotated boxes (scripts/detrac_to_gt.py). Detections centred in
an ignored region are dropped first: nothing there is annotated, so nothing there can be judged. A kept detection is
unmatched when no annotated box in its frame overlaps it at IoU 0.5 or more, and a straddle when it is unmatched and
bridges two annotated boxes in its frame as replay.between.bridge defines it, scores aside: the two sit side by side
(their vertical ranges overlap by at least half the shorter height, and they overlap each other at IoU under 0.1), the
detection covers at least 0.2 of its own area on each, matches neither (IoU under 0.5), and its centre lies between
theirs in x. An enter is put down to a straddle when its track's box in the enter's frame stands for one: a track box
stands for the detection it overlaps most in its frame, and only if they overlap at IoU above 0.5. Tracks are built from
every detection, masked ones included, so a box whose own detection was masked must not stand for a neighbour.
What it cannot tell apart: a vehicle the annotators missed, seen in the gap between two annotated ones, is a straddle
here, just as it is a bridge box to replay.between. And it undercounts: a box across one annotated vehicle and one in an
ignored region, or across two vehicles one behind the other, is unmatched but not a straddle.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .between import bridge, enter_flags
from .detections import read_detections
from .schema import read_tracks
from .trackers.greedy_iou import iou


def mask(dets, regions):
    """The detections whose bbox centre lies in none of `regions` ([x1, y1, x2, y2], edges included)."""
    def inside(d, r):
        cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
        return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]
    return [d for d in dets if not any(inside(d, r) for r in regions)]


def _by_frame(boxes):
    out = defaultdict(list)
    for b in boxes:
        out[b.frame].append(b)
    return out


def _unmatched(d, frame_gt) -> bool:
    return all(iou(d.bbox, g.bbox) < 0.5 for g in frame_gt)


def straddles(dets, gt) -> dict[int, list[tuple]]:
    """{frame: [(detection, annotated box, annotated box), ...]} for every frame holding a straddle, the two annotated
    boxes being the ones it bridges. dets: already masked; gt: annotated boxes, with ignored regions already dropped."""
    gt_by, out = _by_frame(gt), defaultdict(list)
    for d in dets:
        frame_gt = gt_by.get(d.frame, [])
        pair = _unmatched(d, frame_gt) and bridge(d, frame_gt, outscored=False)
        if pair:
            out[d.frame].append((d, *pair))
    return dict(out)


def summary(dets, gt) -> dict:
    """Counts over the kept detections and the frames holding at least one of them."""
    gt_by, flagged = _by_frame(gt), straddles(dets, gt)
    frames = len({d.frame for d in dets})
    n = sum(len(v) for v in flagged.values())
    return {"detections": len(dets), "frames": frames,
            "unmatched": sum(_unmatched(d, gt_by.get(d.frame, [])) for d in dets), "straddles": n,
            "straddle_frames": len(flagged), "straddle_share": round(n / len(dets), 4) if dets else 0.0,
            "straddle_frame_share": round(len(flagged) / frames, 4) if frames else 0.0}


def enters_table(dets, tracks, gt, poly, **counter_kw) -> list[dict]:
    """One row per debounced enter, in time order: how many of its track's boxes are straddles, and whether the box it
    entered with is one. A track box stands for a detection only at IoU above 0.5 (see the module docstring).
    dets: already masked. counter_kw goes to DebouncedZoneCounter."""
    flagged = {d for v in straddles(dets, gt).values() for d, *_ in v}
    rows = []
    for n, (e, tb, flags) in enumerate(enter_flags(dets, tracks, poly, lambda d, _: d in flagged, min_iou=0.5,
                                                   **counter_kw)):
        hits = sum(flags)
        rows.append({"enter": n, "ts_ms": e.ts_ms, "frame": e.frame, "track_id": e.track_id, "boxes": len(tb),
                     "straddle_boxes": hits, "straddle_share": round(hits / len(tb), 2),
                     "straddle_at_enter": any(f for t, f in zip(tb, flags) if t.frame == e.frame)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dets", required=True)
    ap.add_argument("--gt", required=True, help="annotated boxes JSONL (scripts/detrac_to_gt.py <out>.gt.jsonl)")
    ap.add_argument("--ignored", required=True, help='{"regions_xyxy": [[x1, y1, x2, y2], ...]} (<out>.ignored.json)')
    ap.add_argument("--tracks", help="with --zone: the enters table")
    ap.add_argument("--zone")
    ap.add_argument("--min-travel-px", type=float, default=0, help="the debounced counter's zone rule (0: off)")
    a = ap.parse_args()
    if bool(a.tracks) != bool(a.zone):
        ap.error("--tracks and --zone go together")
    every = read_detections(a.dets)
    dets, gt = mask(every, json.load(open(a.ignored))["regions_xyxy"]), list(read_tracks(a.gt))
    s = summary(dets, gt)
    print(f"detections {len(every)} | {len(every) - len(dets)} centred in an ignored region, dropped | "
          f"{s['detections']} kept in {s['frames']} frames")
    print(f"unmatched {s['unmatched']} | straddles {s['straddles']} ({s['straddle_share']} of kept) "
          f"in {s['straddle_frames']} frames ({s['straddle_frame_share']} of frames)")
    if not a.tracks:
        return
    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    rows = enters_table(dets, list(read_tracks(a.tracks)), gt, poly, min_travel_px=a.min_travel_px)
    print(f"\nenters {len(rows)} | straddle at enter {sum(r['straddle_at_enter'] for r in rows)} | "
          f"straddle share 0.5 or more {sum(r['straddle_share'] >= 0.5 for r in rows)}\n")
    cols = ["enter", "ts_ms", "frame", "track_id", "boxes", "straddle_boxes", "straddle_share", "straddle_at_enter"]
    print("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        if r["straddle_boxes"]:
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
