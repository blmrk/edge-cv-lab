"""Boxes between two vehicles side by side: does a box across the gap become a counted visit?

python -m replay.between --dets runs/gap.dets.jsonl --tracks runs/gap.bytetrack.jsonl --zone runs/gap.zone.json
python -m replay.between ... --sheet runs/gap.sheet.jpg --video ../media/clip.mp4   also draws every enter's box at the
    moment it entered, numbered in time order, 32 to a page (runs/gap.sheet-1.jpg, ...), for a visual check

Works from detections alone, for clips with no annotations. A bridge box is a detection that two other detections in
its frame both outscore, where those two sit side by side (their vertical ranges overlap by at least half the shorter
height, and they overlap each other at IoU under 0.1), the box covers at least `touch` of its own area on each, matches
neither (IoU under 0.5 with both), and its centre lies between theirs in x. That flags boxes straddling two vehicles,
but also a real vehicle seen in the gap between two nearer ones: flags are candidates, and the sheet is the check.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .detections import read_detections
from .schema import read_tracks
from .trackers.greedy_iou import iou
from .zones import DebouncedZoneCounter, run


def _inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def bridge(box, frame_dets, touch: float = 0.2, outscored: bool = True):
    """The two detections `box` bridges, or None. frame_dets: every detection in its frame (box itself may be one).
    outscored=False drops the score test, for neighbours with no real score: annotated boxes (replay.straddle)."""
    b = box.bbox
    if _area(b) <= 0:
        return None
    near = [d for d in frame_dets if (not outscored or d.score > box.score)
            and _inter(b, d.bbox) >= touch * _area(b) and iou(b, d.bbox) < 0.5]
    cx = (b[0] + b[2]) / 2
    for i, p in enumerate(near):
        for q in near[i + 1:]:
            rows_overlap = min(p.bbox[3], q.bbox[3]) - max(p.bbox[1], q.bbox[1])
            if rows_overlap < 0.5 * min(p.bbox[3] - p.bbox[1], q.bbox[3] - q.bbox[1]) or iou(p.bbox, q.bbox) >= 0.1:
                continue
            lo, hi = sorted(((p.bbox[0] + p.bbox[2]) / 2, (q.bbox[0] + q.bbox[2]) / 2))
            if lo < cx < hi:
                return p, q
    return None


def enter_flags(dets, tracks, poly, flag, min_iou: float = 0, **counter_kw) -> list[tuple]:
    """(enter event, its track's boxes, a flag per box) for every debounced enter, in time order. A track box stands for
    the detection it overlaps most in its frame, if they overlap at IoU above min_iou (0: at all), and its flag is
    flag(that detection, every detection in the frame); a box that stands for none is False. counter_kw goes to
    DebouncedZoneCounter."""
    by_frame, by_id = defaultdict(list), defaultdict(list)
    for d in dets:
        by_frame[d.frame].append(d)
    tracks = sorted(tracks, key=lambda t: (t.frame, t.track_id))
    for t in tracks:
        by_id[t.track_id].append(t)
    enters = sorted((e for e in run(DebouncedZoneCounter(poly, **counter_kw), tracks) if e.kind == "enter"),
                    key=lambda e: (e.ts_ms, e.track_id))
    out = []
    for e in enters:
        flags = []
        for t in by_id[e.track_id]:
            src = max(by_frame[t.frame], key=lambda d: iou(d.bbox, t.bbox), default=None)
            flags.append(src is not None and iou(src.bbox, t.bbox) > min_iou and flag(src, by_frame[t.frame]))
        out.append((e, by_id[e.track_id], flags))
    return out


def enters_table(dets, tracks, poly, **counter_kw) -> list[dict]:
    """One row per debounced enter, in time order: how many of its track's boxes are bridge boxes (see enter_flags for
    how a track box maps to a detection). bridge_share is rounded for display; cut on bridge_boxes / boxes. counter_kw
    goes to DebouncedZoneCounter."""
    rows = []
    for n, (e, tb, flags) in enumerate(enter_flags(dets, tracks, poly, lambda d, ds: bridge(d, ds) is not None,
                                                   **counter_kw)):
        flagged = sum(flags)
        rows.append({"enter": n, "ts_ms": e.ts_ms, "frame": e.frame, "track_id": e.track_id, "boxes": len(tb),
                     "bridge_boxes": flagged, "bridge_share": round(flagged / len(tb), 2),
                     "score": round(sum(t.score for t in tb) / len(tb), 3)})
    return rows


def sheet(rows, tracks, video: str, out: str, per_page: int = 32, tile: int = 240) -> list[str]:
    """Every enter's box at the frame it entered, cropped, numbered like `rows`. Needs opencv (pip install -e '.[viz]')."""
    import cv2
    import numpy as np

    at = {(t.frame, t.track_id): t.bbox for t in tracks}
    cap, tiles = cv2.VideoCapture(video), []
    for r in rows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"{video}: cannot read frame {r['frame']}")
        x1, y1, x2, y2 = map(int, at[(r["frame"], r["track_id"])])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 4)
        cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, int(max(x2 - x1, y2 - y1) * 0.9)
        crop = cv2.resize(img[max(0, cy - half):cy + half, max(0, cx - half):cx + half], (tile, tile))
        cv2.putText(crop, f"#{r['enter']} id{r['track_id']}", (4, 18), 0, 0.6, (255, 255, 255), 2)
        tiles.append(crop)
    blank, paths = np.zeros((tile, tile, 3), np.uint8), []
    for page, i in enumerate(range(0, len(tiles), per_page), 1):
        chunk = tiles[i:i + per_page]
        chunk += [blank] * (-len(chunk) % 8)
        path = str(Path(out).with_name(f"{Path(out).stem}-{page}{Path(out).suffix}"))
        cv2.imwrite(path, cv2.vconcat([cv2.hconcat(chunk[k:k + 8]) for k in range(0, len(chunk), 8)]))
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--sheet", help="write contact sheets of every enter here (needs --video and opencv)")
    ap.add_argument("--video")
    ap.add_argument("--min-travel-px", type=float, default=0, help="the debounced counter's zone rule (0: off)")
    a = ap.parse_args()
    if a.sheet and not a.video:
        ap.error("--sheet needs --video")
    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    dets, tracks = read_detections(a.dets), list(read_tracks(a.tracks))
    by_frame = defaultdict(list)
    for d in dets:
        by_frame[d.frame].append(d)
    flagged = sorted(d.score for ds in by_frame.values() for d in ds if bridge(d, ds))
    frames = len({f for f, ds in by_frame.items() if any(bridge(d, ds) for d in ds)})
    print(f"detections {len(dets)} in {len(by_frame)} frames | bridge boxes {len(flagged)} in {frames} frames"
          + (f", median score {flagged[len(flagged) // 2]}" if flagged else ""))
    rows = enters_table(dets, tracks, poly, min_travel_px=a.min_travel_px)
    print(f"enters {len(rows)} | with any bridge box {sum(r['bridge_boxes'] > 0 for r in rows)} | "
          f"half or more {sum(2 * r['bridge_boxes'] >= r['boxes'] for r in rows)}\n")  # unrounded
    cols = ["enter", "ts_ms", "track_id", "boxes", "bridge_boxes", "bridge_share", "score"]
    print("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in sorted((r for r in rows if r["bridge_boxes"]), key=lambda r: (-r["bridge_share"], r["ts_ms"])):
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    if a.sheet:
        print("\n" + "\n".join(sheet(rows, tracks, a.video, a.sheet)))


if __name__ == "__main__":
    main()
