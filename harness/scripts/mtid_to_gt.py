"""MTID infrastructure-camera annotations -> harness ground truth.

python scripts/mtid_to_gt.py --annotations ../media/candidates/mtid/annotations/Infrastructure --out runs/mtid \
    [--fps 30] [--gap-frames 0] [--exclude Cyclist]

Writes <out>.gt.jsonl: ground-truth tracks in the harness TrackBox schema (for --gt in replay.compare).

Format, as found in the downloaded annotations: one sub-folder per annotation batch, each with a ';'-separated
annotations.csv:
  RGB mask file;RGB file;Object ID;Annotation tag;
  ./rgbMasks/seq3-infra_0000001.png;seq3-infra_0000001.jpg;200;Car;;0;232 409 298 360 ...
The frame number is in the RGB file name and is 1-based; the harness is 0-based, so MTID frame 1 is frame 0 of
media/sample.mp4. The last field is the object's outline as x y pairs; the box is its extent, unclipped.
Tags: Car, Van, Lorry, Bus, Cyclist. Cyclists are excluded by default because scripts/dump_detections.py keeps
only COCO car, bus and truck, so a tracker on those detections can never follow a cyclist.

Object IDs are reused: a number that belonged to one vehicle is later given to another. A reappearance after more
than --gap-frames unannotated frames therefore starts a new identity. Identities are numbered from 1 in order of
first appearance. The default gap comes from the real annotations (run this script; it prints the gap line):
  - every frame from the first to the last annotated one has at least one annotated object, so there are no
    unannotated-frame runs to bridge;
  - within an Object ID there are 53 reappearances after a gap, the shortest 1 frame and the longest 1646;
  - each of those 53 is a different vehicle: the box comes back at another place in the image, not where it left.
So any gap means reuse, and the default splits on any unannotated frame (0). A larger value only helps footage
where a visible vehicle is left unannotated for a few frames.
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
EXCLUDE = ("Cyclist",)
_FRAME = re.compile(r"(\d+)\.\w+$")


class _Row(NamedTuple):
    frame: int  # 0-based harness frame
    oid: int  # MTID Object ID, reused across vehicles
    tag: str
    bbox: tuple[float, float, float, float]


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
                rows.append(_Row(frame, oid, parts[3], (min(xs), min(ys), max(xs), max(ys))))
    return rows


def split_identities(rows: list[_Row], gap_frames: int = GAP_FRAMES):
    """Cut each Object ID's rows into one segment per vehicle."""
    by_id: dict[int, list[_Row]] = defaultdict(list)
    for r in rows:
        by_id[r.oid].append(r)
    segments: list[list[_Row]] = []
    gaps: list[int] = []
    reused: set[int] = set()
    gap_splits = 0
    for oid, rs in by_id.items():
        rs.sort(key=lambda r: r.frame)
        seg = [rs[0]]
        for a, b in zip(rs, rs[1:]):
            gap = b.frame - a.frame - 1
            if gap:
                gaps.append(gap)
            if gap > gap_frames:
                segments.append(seg)
                seg = []
                gap_splits += 1
                reused.add(oid)
            seg.append(b)
        segments.append(seg)
    stats = {"object_ids": len(by_id), "reused_ids": len(reused), "gap_splits": gap_splits, "gaps": sorted(gaps)}
    return segments, stats


def convert(root: Path, fps: float = FPS, gap_frames: int = GAP_FRAMES, exclude=EXCLUDE):
    rows = read_rows(root)
    segments, stats = split_identities(rows, gap_frames)
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
    ap.add_argument("--exclude", nargs="*", default=list(EXCLUDE), help="annotation tags to leave out")
    a = ap.parse_args()

    tracks, s = convert(Path(a.annotations), a.fps, a.gap_frames, tuple(a.exclude))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.gt.jsonl", "w") as fh:
        for t in tracks:
            fh.write(json.dumps({"frame": t.frame, "ts_ms": t.ts_ms, "track_id": t.track_id,
                                 "bbox": [round(v, 1) for v in t.bbox], "score": 1.0, "cls": t.cls}) + "\n")
    n_ids = len({t.track_id for t in tracks})
    left_out = "/".join(x.lower() for x in a.exclude) or "no"
    print(f"{out}.gt.jsonl: {len(tracks)} boxes, {n_ids} identities, {s['excluded_boxes']} {left_out} boxes excluded")
    print(f"{s['reused_ids']} of {s['object_ids']} Object IDs reused: {s['gap_splits']} reappearances split "
          f"after more than {a.gap_frames} unannotated frames")
    g = s["gaps"]
    print(f"gaps within an Object ID: {len(g)}" + (f" (shortest {g[0]}, longest {g[-1]} frames)" if g else "")
          + f"; MTID frames {s['frame_range'][0]}-{s['frame_range'][1]} with no annotation at all: "
          f"{s['unannotated_frames']}")


if __name__ == "__main__":
    main()
