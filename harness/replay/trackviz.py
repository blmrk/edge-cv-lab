"""Animated tracker comparison: one strip per tracker, same detections, colour = track ID.

Grey shapes are the real vehicles (ground truth), drawn even when the detector cannot see them.
Coloured boxes are what each tracker reports. Each strip shows IDs used, visits closed, and the
dwell timer of any visit still open, so a merged visit is visible as a timer that never resets.

python -m replay.trackviz --dets fixtures/queue.dets.jsonl --gt fixtures/queue.gt.jsonl --zone fixtures/zone.json \
    --trackers "greedy_iou:max_age=5" greedy_iou groundplane --occluder 440 560 --out ../docs/img/trackers.gif
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

from .detections import frames, read_detections
from .schema import read_tracks
from .trackers import create
from .viz import BG, FONT, LINE, ROAD, TEXT, ZONE_C, _palette, draw_zone

W = 1280


def main():
    import imageio.v2 as imageio

    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", help="ground-truth tracks, drawn as grey vehicles")
    ap.add_argument("--trackers", nargs="+", required=True, help='"name" or "name:key=json,key=json"')
    ap.add_argument("--labels", nargs="*", help="display names, same order as --trackers")
    ap.add_argument("--occluder", nargs=2, type=int, metavar=("X1", "X2"))
    ap.add_argument("--crop", nargs=2, type=int, default=[230, 520], metavar=("Y1", "Y2"))
    ap.add_argument("--every", type=int, default=2, help="render every Nth frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()

    from .zones import DebouncedZoneCounter

    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    dets = read_detections(a.dets)
    gt = defaultdict(list)
    for b in (read_tracks(a.gt) if a.gt else []):
        gt[b.frame].append(b)

    runs = []
    for i, spec in enumerate(a.trackers):
        name, _, raw = spec.partition(":")
        params = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in raw.split(",") if kv)}
        runs.append({"label": (a.labels or a.trackers)[i], "tracker": create(name, **params),
                     "counter": DebouncedZoneCounter(poly), "ids": set(), "closed": 0})

    y1, y2 = a.crop
    images = []
    last = max(d.frame for d in dets)
    for f, ts, ds in frames(dets):
        strips = []
        for r in runs:
            boxes = r["tracker"].update(f, ts, ds)
            for b in boxes:
                r["ids"].add(b.track_id)
                r["closed"] += sum(1 for e in r["counter"].update(b) if e.kind == "exit")
            if f == last:                                             # end of stream: close what is still open
                r["closed"] += sum(1 for e in r["counter"].expire(ts + 10**9) if e.kind == "exit")
            if f % a.every and f != last:
                continue
            img = np.full((720, W, 3), BG, np.uint8)
            cv2.rectangle(img, (0, 330), (W, 470), ROAD, -1)
            for x in range(0, W, 80):
                cv2.line(img, (x, 400), (x + 40, 400), LINE, 2)
            draw_zone(img, poly)
            for g in gt.get(f, []):                                   # the real vehicles
                gx1, gy1, gx2, gy2 = map(int, g.bbox)
                cv2.rectangle(img, (gx1 + 4, gy1 + 10), (gx2 - 4, gy2), (95, 95, 95), -1)
            if a.occluder:                                            # drawn over the vehicles
                cv2.rectangle(img, (a.occluder[0], 300), (a.occluder[1], 500), (70, 74, 82), -1)
                cv2.putText(img, "pillar", (a.occluder[0] + 22, 292), FONT, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
            for b in boxes:
                bx1, by1, bx2, by2 = map(int, b.bbox)
                c = _palette(b.track_id)
                cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 3)
                cv2.putText(img, f"#{b.track_id}", (bx1, by1 - 8), FONT, 0.8, c, 2, cv2.LINE_AA)

            strip = img[y1:y2].copy()
            cv2.rectangle(strip, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.putText(strip, r["label"], (16, 28), FONT, 0.8, TEXT, 2, cv2.LINE_AA)
            stats = f"IDs used {len(r['ids'])}    visits closed {r['closed']}"
            cv2.putText(strip, stats, (470, 28), FONT, 0.7, TEXT, 1, cv2.LINE_AA)
            for tid, secs in r["counter"].open_visits(ts)[:1]:
                col = (70, 80, 235) if secs > 12 else ZONE_C
                cv2.putText(strip, f"open visit #{tid}: {secs:4.1f} s", (930, 28), FONT, 0.7, col, 2, cv2.LINE_AA)
            strips.append(strip)
        if strips:
            frame = np.vstack(strips)
            h = int(frame.shape[0] * a.width / W)
            images.append(cv2.cvtColor(cv2.resize(frame, (a.width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

    images += [images[-1]] * (a.fps * 2)                              # hold the final frame
    imageio.mimsave(a.out, images, duration=1 / a.fps, loop=0)
    for r in runs:
        print(f"{r['label']}: IDs {len(r['ids'])}, visits closed {r['closed']}")
    print("wrote", a.out, f"({len(images)} frames)")


if __name__ == "__main__":
    main()
