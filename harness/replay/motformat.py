"""Bridge to any tracker that speaks MOTChallenge text files, which is nearly every research repo.

  1. python -m replay.motformat export-dets --dets runs/dets.jsonl --out seq/det/det.txt
  2. run FastTracker / UCMCTrack / TrackTrack on seq/ following their README
  3. python -m replay.motformat import-tracks --mot results/seq.txt --dets runs/dets.jsonl --out runs/fasttracker.jsonl
  4. (optional) python -m replay.motformat export-tracks --tracks runs/x.jsonl --out trackeval/x.txt   -> TrackEval

MOT rows: frame, id, left, top, width, height, conf, x, y, z   (frame is 1-based)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .detections import read_detections
from .schema import TrackBox, read_tracks


def export_dets(dets, path):
    with open(path, "w") as fh:
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            fh.write(f"{d.frame + 1},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{d.score:.4f},-1,-1,-1\n")


def export_tracks(tracks, path):
    with open(path, "w") as fh:
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            fh.write(f"{t.frame + 1},{t.track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{t.score:.4f},-1,-1,-1\n")


def import_tracks(path, ts_by_frame: dict[int, int]) -> list[TrackBox]:
    out = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            p = line.replace(" ", ",").split(",")
            p = [v for v in p if v != ""]
            try:
                f, tid, x, y, w, h = int(float(p[0])) - 1, int(float(p[1])), *map(float, p[2:6])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}:{n}: not a MOT row: {exc}") from exc
            if f not in ts_by_frame:
                raise ValueError(f"{path}:{n}: frame {f + 1} is not in the detections file; wrong sequence?")
            out.append(TrackBox(f, ts_by_frame[f], tid, (x, y, x + w, y + h), float(p[6]) if len(p) > 6 else 1.0))
    return sorted(out, key=lambda b: (b.frame, b.track_id))


def write_tracks(tracks, path):
    with open(path, "w") as fh:
        for t in tracks:
            fh.write(json.dumps({**asdict(t), "bbox": [round(v, 1) for v in t.bbox]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export-dets", "import-tracks", "export-tracks"])
    ap.add_argument("--dets"); ap.add_argument("--tracks"); ap.add_argument("--mot"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "export-dets":
        export_dets(read_detections(a.dets), a.out)
    elif a.cmd == "export-tracks":
        export_tracks(list(read_tracks(a.tracks)), a.out)
    else:
        from .detections import frames
        ts = {f: t for f, t, _ in frames(read_detections(a.dets))}
        write_tracks(import_tracks(a.mot, ts), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
