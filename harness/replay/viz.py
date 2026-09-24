"""Visuals from a tracks JSONL. Works with no video at all (draws a schematic road),
or over a real frame with --video.

python -m replay.viz compare      --tracks T --zone Z --out compare.gif [--start 0 --seconds 12]
python -m replay.viz heatmap      --tracks T --zone Z --out heatmap.png
python -m replay.viz trajectories --tracks T --zone Z --out trajectories.png
python -m replay.viz timeline     --tracks T --zone Z --out timeline.png

Needs: pip install -e ".[viz]"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

from .schema import TrackBox, read_tracks
from .zones import DebouncedZoneCounter, NaiveZoneCounter, ZoneEvent

W, H = 1280, 720
BG, ROAD, LINE = (24, 22, 20), (48, 44, 40), (110, 105, 100)
OK, BAD, ZONE_C, TEXT = (120, 200, 80), (70, 80, 235), (230, 190, 60), (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _palette(tid: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[(tid * 47) % 180, 170, 240]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def background(video: str | None) -> np.ndarray:
    if video:
        ok, frame = cv2.VideoCapture(video).read()
        if not ok:
            raise SystemExit(f"cannot read a frame from {video}")
        return cv2.resize(frame, (W, H))
    img = np.full((H, W, 3), BG, np.uint8)
    for top, bottom in ((330, 470), (560, 700)):  # schematic lanes
        cv2.rectangle(img, (0, top), (W, bottom), ROAD, -1)
        for x in range(0, W, 80):
            cv2.line(img, (x, (top + bottom) // 2), (x + 40, (top + bottom) // 2), LINE, 2)
    return img


def draw_zone(img, poly, colour=ZONE_C, fill=0.10):
    pts = np.array(poly, np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, fill, img, 1 - fill, 0, img)
    cv2.polylines(img, [pts], True, colour, 2)


def _frames(boxes):
    out = defaultdict(list)
    for b in boxes:
        out[b.frame].append(b)
    return out


def compare(boxes, poly, out, video, start, seconds, fps_in=30, fps_out=10):
    import imageio.v2 as imageio

    bg, frames = background(video), _frames(boxes)
    panels = [("naive: centroid, flip on every crossing", NaiveZoneCounter(poly, "centroid"), "centroid", BAD),
              ("debounced: footpoint + hysteresis + min dwell", DebouncedZoneCounter(poly), "footpoint", OK)]
    counts, flash = [0, 0], [0, 0]
    step, images = fps_in // fps_out, []
    last = max(frames)
    end = min(last, start + seconds * fps_in) if seconds else last

    for f in range(0, end + 1):
        evs = [[e for b in frames.get(f, []) for e in c.update(b)] for _, c, _, _ in panels]
        for i, e in enumerate(evs):
            n = sum(1 for x in e if x.kind == "enter")
            counts[i] += n
            flash[i] = 4 if n else max(0, flash[i] - (1 if f % step == 0 else 0))
        if f < start or f % step:
            continue
        row = []
        for i, (title, _c, anchor, colour) in enumerate(panels):
            img = bg.copy()
            draw_zone(img, poly, colour if flash[i] else ZONE_C, 0.35 if flash[i] else 0.10)
            for b in frames.get(f, []):
                x1, y1, x2, y2 = map(int, b.bbox)
                c = _palette(b.track_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, f"#{b.track_id}", (x1, y1 - 8), FONT, 0.7, c, 2, cv2.LINE_AA)
                ax, ay = map(int, b.footpoint if anchor == "footpoint" else b.centroid)
                cv2.circle(img, (ax, ay), 7, (255, 255, 255), -1)
                cv2.circle(img, (ax, ay), 7, (0, 0, 0), 2)
            cv2.rectangle(img, (0, 0), (W, 120), (0, 0, 0), -1)
            cv2.putText(img, title, (24, 44), FONT, 1.0, TEXT, 2, cv2.LINE_AA)
            cv2.putText(img, f"visits counted: {counts[i]}", (24, 100), FONT, 1.5, colour, 3, cv2.LINE_AA)
            row.append(cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
        images.append(cv2.cvtColor(np.hstack(row), cv2.COLOR_BGR2RGB))
    imageio.mimsave(out, images, duration=1 / fps_out, loop=0)
    return counts


def heatmap(boxes, poly, out, video):
    acc = np.zeros((H, W), np.float32)
    for b in boxes:  # each box-frame is one unit of dwell time at its ground contact point
        x, y = map(int, b.footpoint)
        if 0 <= x < W and 0 <= y < H:
            acc[y, x] += 1
    acc = cv2.GaussianBlur(acc, (0, 0), 18)
    norm = np.power(acc / (acc.max() or 1), 0.5)  # gamma lifts the faint through-traffic
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(norm * 1.4, 0, 0.85)[..., None]
    img = (background(video) * (1 - alpha) + heat * alpha).astype(np.uint8)
    draw_zone(img, poly, fill=0.0)
    cv2.putText(img, "dwell heatmap (footpoints, time-weighted)", (24, 44), FONT, 1.0, TEXT, 2, cv2.LINE_AA)
    cv2.imwrite(out, img)


def trajectories(boxes, poly, out, video):
    img = background(video)
    draw_zone(img, poly)
    paths = defaultdict(list)
    for b in boxes:
        paths[b.track_id].append(b.footpoint)
    for tid, pts in paths.items():
        cv2.polylines(img, [np.array(pts, np.int32)], False, _palette(tid), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, pts[0])), 5, _palette(tid), -1)
    cv2.putText(img, f"footpoint trajectories, {len(paths)} tracks (colour = track id)", (24, 44),
                FONT, 1.0, TEXT, 2, cv2.LINE_AA)
    cv2.imwrite(out, img)


def timeline(boxes, poly, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def intervals(counter):
        evs: list[ZoneEvent] = []
        for b in boxes:
            evs.extend(counter.update(b))
        if hasattr(counter, "flush"):
            evs.extend(counter.flush())
        open_, spans = {}, defaultdict(list)
        for e in sorted(evs, key=lambda e: e.ts_ms):
            if e.kind == "enter":
                open_[e.track_id] = e.ts_ms
            elif e.track_id in open_:
                spans[e.track_id].append((open_.pop(e.track_id) / 1000, e.ts_ms / 1000))
        return spans, sum(1 for e in evs if e.kind == "enter")

    runs = [("Naive (centroid)", NaiveZoneCounter(poly, "centroid"), "#eb5046"),
            ("Debounced (footpoint)", DebouncedZoneCounter(poly), "#50c878")]
    tids = sorted({b.track_id for b in boxes})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (name, counter, colour) in zip(axes, runs):
        spans, n = intervals(counter)
        for row, tid in enumerate(tids):
            for a, z in spans.get(tid, []):
                ax.barh(row, max(z - a, 0.15), left=a, height=0.7, color=colour)
        ax.set_title(f"{name}: {n} visits counted", loc="left", fontweight="bold")
        ax.set_yticks(range(len(tids)), [f"#{t}" for t in tids], fontsize=7)
        ax.set_ylabel("track")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("seconds")
    fig.suptitle("Zone occupancy per track: each bar is one counted visit", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "heatmap", "trajectories", "timeline"])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", help="optional: use its first frame as the background")
    ap.add_argument("--start", type=int, default=0, help="compare: first frame to render")
    ap.add_argument("--seconds", type=int, default=0, help="compare: clip length, 0 = all")
    a = ap.parse_args()

    poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    boxes = sorted(read_tracks(a.tracks), key=lambda b: (b.frame, b.track_id))
    if a.cmd == "compare":
        print("visits [naive, debounced]:", compare(boxes, poly, a.out, a.video, a.start, a.seconds))
    elif a.cmd == "heatmap":
        heatmap(boxes, poly, a.out, a.video)
    elif a.cmd == "trajectories":
        trajectories(boxes, poly, a.out, a.video)
    else:
        timeline(boxes, poly, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
