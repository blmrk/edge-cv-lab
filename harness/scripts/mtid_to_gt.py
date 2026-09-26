"""MTID infrastructure-camera annotations -> harness ground truth.

python scripts/mtid_to_gt.py --annotations ../media/candidates/mtid/annotations/Infrastructure --out runs/mtid \
    [--fps 30] [--gap-frames 0] [--max-jump-px 85] [--exclude Cyclist]

Writes <out>.gt.jsonl: ground-truth tracks in the harness TrackBox schema (for --gt in replay.compare).

Format, as found in the downloaded annotations: one sub-folder per annotation batch, each with a ';'-separated
annotations.csv:
  RGB mask file;RGB file;Object ID;Annotation tag;
  ./rgbMasks/seq3-infra_0000001.png;seq3-infra_0000001.jpg;200;Car;;0;232 409 298 360 ...
The frame number is in the RGB file name and is 1-based; the harness is 0-based, so MTID frame 1 is frame 0 of
media/sample.mp4. The last field is the object's outline as x y pairs; the box is its extent, unclipped.
Tags: Car, Van, Lorry, Bus, Cyclist. Cyclists are excluded by default because scripts/dump_detections.py keeps
only COCO car, bus and truck, so a tracker on those detections can never follow a cyclist.

Object IDs are numbered per annotation batch (one sub-folder each; numbering restarts at 200 in every batch), so an
identity is keyed by (batch, Object ID). A vehicle on screen across a batch boundary gets a new ID in the next batch;
it is rejoined when its box on the batch's last frame overlaps (IoU >= LINK_IOU) its box on the next batch's first
frame. On the real annotations that rejoins 12 vehicles.

Within a batch, IDs are also reused: a number that belonged to one vehicle is later given to another. A reappearance
after more than --gap-frames unannotated frames therefore starts a new identity. The default gap comes from the real
annotations (run this script; it prints the gap line):
  - every frame from the first to the last annotated one has at least one annotated object, so there are no
    unannotated-frame runs to bridge;
  - within a batch's Object ID there are 9 reappearances after a gap, the shortest 4 frames and the longest 419;
  - each is a different vehicle: the box comes back at another place in the image, not where it left.
So any gap means reuse, and the default splits on any unannotated frame (0). A larger value only helps footage
where a visible vehicle is left unannotated for a few frames.

An ID can also be reused with no gap, which no gap setting can split. So an identity also starts anew when its
footpoint (bottom-centre of the box) moves more than --max-jump-px per frame elapsed. The script prints the fastest
step it kept and the slowest it split; on the real annotations with the defaults those are 65.5 px/frame (a bus whose
box widens as it enters the frame) and 199.6 px/frame, over 2 jump splits. The default, 85, sits between the two.
Licence: MTID is CC BY 4.0 (see media/SOURCES.md). The annotations are not committed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay.schema import TrackBox  # noqa: E402

FPS = 30
GAP_FRAMES = 0
MAX_JUMP_PX = 85.0
LINK_IOU = 0.3  # box overlap that rejoins one vehicle across an annotation-batch boundary
EXCLUDE = ("Cyclist",)
_FRAME = re.compile(r"(\d+)\.\w+$")


class _Row(NamedTuple):
    frame: int  # 0-based harness frame
    oid: int  # MTID Object ID, reused across vehicles
    tag: str
    bbox: tuple[float, float, float, float]
    batch: str = ""  # annotation sub-folder: Object IDs are numbered per batch


def read_rows(root: Path) -> list[_Row]:
    rows: list[_Row] = []
    seen: dict[tuple[int, int], str] = {}
    for csv in sorted(Path(root).glob("*/annotations.csv")):
        with open(csv) as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("RGB mask file"):
                    continue
                parts = line.split(";")
                try:
                    frame = int(_FRAME.search(parts[1]).group(1)) - 1
                    oid = int(parts[2])
                    xy = [float(v) for v in parts[-1].split()]
                    if len(xy) < 4 or len(xy) % 2:
                        raise ValueError(f"outline needs x y pairs, got {len(xy)} values")
                except (IndexError, AttributeError, ValueError) as exc:
                    raise ValueError(f"{csv}:{n}: bad annotation row: {exc}") from exc
                where = f"{csv}:{n}"
                if (frame, oid) in seen:
                    raise ValueError(f"{where}: Object ID {oid} annotated twice in MTID frame {frame + 1} "
                                     f"(first at {seen[frame, oid]})")
                seen[frame, oid] = where
                xs, ys = xy[0::2], xy[1::2]
                rows.append(_Row(frame, oid, parts[3], (min(xs), min(ys), max(xs), max(ys)), csv.parent.name))
    return rows


def _speed(a: _Row, b: _Row) -> float:
    """Footpoint (bottom-centre) displacement per frame elapsed, in px."""
    (ax, ay), (bx, by) = ((a.bbox[0] + a.bbox[2]) / 2, a.bbox[3]), ((b.bbox[0] + b.bbox[2]) / 2, b.bbox[3])
    return ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / (b.frame - a.frame)


def split_identities(rows: list[_Row], gap_frames: int = GAP_FRAMES, max_jump_px: float = MAX_JUMP_PX):
    """Cut each Object ID's rows into one segment per vehicle."""
    by_id: dict[tuple[str, int], list[_Row]] = defaultdict(list)
    for r in rows:
        by_id[r.batch, r.oid].append(r)  # the same number in another batch is another vehicle
    segments: list[list[_Row]] = []
    gaps: list[int] = []
    reused: set[tuple[str, int]] = set()
    gap_splits = jump_splits = 0
    kept_max, split_min = 0.0, None
    for oid, rs in by_id.items():
        rs.sort(key=lambda r: r.frame)
        seg = [rs[0]]
        for a, b in zip(rs, rs[1:]):
            gap = b.frame - a.frame - 1
            if gap:
                gaps.append(gap)
            if gap > gap_frames:
                gap_splits += 1
            else:
                v = _speed(a, b)
                if v <= max_jump_px:
                    kept_max = max(kept_max, v)
                    seg.append(b)
                    continue
                jump_splits += 1
                split_min = v if split_min is None else min(split_min, v)
            segments.append(seg)
            seg = [b]
            reused.add(oid)
        segments.append(seg)
    stats = {"object_ids": len(by_id), "reused_ids": len(reused), "gap_splits": gap_splits,
             "jump_splits": jump_splits, "gaps": sorted(gaps),
             "kept_max_px_per_frame": round(kept_max, 1),
             "split_min_px_per_frame": None if split_min is None else round(split_min, 1)}
    return segments, stats


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def link_batches(segments: list[list[_Row]], min_iou: float = LINK_IOU) -> int:
    """Object IDs restart in every annotation batch, so a vehicle on screen across a batch boundary has two IDs.
    Rejoin a segment ending on a batch's last frame with one starting on the next batch's first frame when their
    boxes overlap (best overlap first). Edits segments in place; returns the number of links."""
    span: dict[str, tuple[int, int]] = {}
    for seg in segments:
        for r in (seg[0], seg[-1]):
            lo, hi = span.get(r.batch, (r.frame, r.frame))
            span[r.batch] = (min(lo, r.frame), max(hi, r.frame))
    order = sorted(span, key=lambda b: span[b][0])
    links = 0
    for b1, b2 in zip(order, order[1:]):
        last, first = span[b1][1], span[b2][0]
        if first != last + 1:
            continue
        ends = [s for s in segments if s[-1].batch == b1 and s[-1].frame == last]
        starts = [s for s in segments if s[0].batch == b2 and s[0].frame == first]
        pairs = sorted(((_iou(e[-1].bbox, st[0].bbox), i, j) for i, e in enumerate(ends) for j, st in enumerate(starts)),
                       reverse=True)
        used_e, used_s = set(), set()
        for v, i, j in pairs:
            if v < min_iou or i in used_e or j in used_s:
                continue
            used_e.add(i), used_s.add(j)
            ends[i].extend(starts[j])
            starts[j].clear()
            links += 1
        segments[:] = [s for s in segments if s]
    return links


def convert(root: Path, fps: float = FPS, gap_frames: int = GAP_FRAMES, exclude=EXCLUDE,
            max_jump_px: float = MAX_JUMP_PX):
    rows = read_rows(root)
    segments, stats = split_identities(rows, gap_frames, max_jump_px)
    stats["batch_links"] = link_batches(segments)
    frames = {r.frame for r in rows}
    stats["unannotated_frames"] = (max(frames) - min(frames) + 1 - len(frames)) if frames else 0
    stats["frame_range"] = (min(frames) + 1, max(frames) + 1) if frames else (0, 0)  # MTID numbering
    kept = [s for s in ([r for r in seg if r.tag not in exclude] for seg in segments) if s]
    kept.sort(key=lambda s: (s[0].frame, s[0].oid))
    tracks = [TrackBox(r.frame, int(r.frame * 1000 / fps), tid, r.bbox, 1.0, r.tag.lower())
              for tid, seg in enumerate(kept, 1) for r in seg]
    tracks.sort(key=lambda b: (b.frame, b.track_id))
    stats["excluded_boxes"] = len(rows) - len(tracks)
    return tracks, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="the MTID Infrastructure folder (sub-folders of annotations.csv)")
    ap.add_argument("--out", required=True, help="output stem, e.g. runs/mtid")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--gap-frames", type=int, default=GAP_FRAMES,
                    help="an Object ID back after more unannotated frames than this is a new vehicle")
    ap.add_argument("--max-jump-px", type=float, default=MAX_JUMP_PX,
                    help="an Object ID whose footpoint moves faster than this (px per frame) is a new vehicle")
    ap.add_argument("--exclude", nargs="*", default=list(EXCLUDE), help="annotation tags to leave out")
    a = ap.parse_args()

    tracks, s = convert(Path(a.annotations), a.fps, a.gap_frames, tuple(a.exclude), a.max_jump_px)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.gt.jsonl", "w") as fh:
        for t in tracks:
            fh.write(json.dumps({"frame": t.frame, "ts_ms": t.ts_ms, "track_id": t.track_id,
                                 "bbox": [round(v, 1) for v in t.bbox], "score": 1.0, "cls": t.cls}) + "\n")
    n_ids = len({t.track_id for t in tracks})
    left_out = "/".join(x.lower() for x in a.exclude) or "no"
    print(f"{out}.gt.jsonl: {len(tracks)} boxes, {n_ids} identities, {s['excluded_boxes']} {left_out} boxes excluded")
    print(f"{s['batch_links']} vehicles rejoined across annotation batches (IoU >= {LINK_IOU:g} at the boundary frame)")
    print(f"{s['reused_ids']} of {s['object_ids']} per-batch Object IDs reused, split {s['gap_splits'] + s['jump_splits']} "
          f"times: {s['gap_splits']} after more than {a.gap_frames} unannotated frames, "
          f"{s['jump_splits']} on a footpoint jump over {a.max_jump_px:g} px/frame")
    slowest = s["split_min_px_per_frame"]
    print(f"footpoint speed within an Object ID: fastest kept {s['kept_max_px_per_frame']} px/frame, "
          + (f"slowest split {slowest} px/frame" if slowest is not None else "none split on a jump"))
    g = s["gaps"]
    print(f"gaps within an Object ID: {len(g)}" + (f" (shortest {g[0]}, longest {g[-1]} frames)" if g else "")
          + f"; MTID frames {s['frame_range'][0]}-{s['frame_range'][1]} with no annotation at all: "
          f"{s['unannotated_frames']}")


if __name__ == "__main__":
    main()
