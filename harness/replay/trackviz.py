"""Animated tracker comparison: one strip per tracker, same detections, colour = track ID.

Grey shapes are the real vehicles (ground truth), drawn even when the detector cannot see them.
Coloured boxes are what each tracker reports. Each strip shows IDs used, visits closed, and the
dwell timer of any visit still open, so a merged visit is visible as a timer that never resets.

python -m replay.trackviz --dets fixtures/queue.dets.jsonl --gt fixtures/queue.gt.jsonl --zone fixtures/zone.json \
    --trackers "greedy_iou:max_age=5" greedy_iou groundplane --occluder 440 560 --out ../docs/img/trackers.gif

Over real footage: --video draws each frame on the matching frame of the clip the detections came from (frame f is
the (f+1)-th frame read, as dump_detections numbers them), and --start/--seconds render a window of it. Trackers
still run from the first frame, so IDs match a full run; the strip counters start at the window.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

from .detections import frame_window, frames, read_detections
from .schema import read_tracks
from .trackers import create
from .viz import BG, FONT, LINE, ROAD, TEXT, ZONE_C, _palette, draw_zone

W = 1280
TAG_COLOURS = 24  # over footage, ID colours cycle through this many hues so each gets an exact GIF palette entry


class _Clip:
    """A video's frames in read order, fetched forward only."""

    def __init__(self, path: str):
        self.path, self.cap, self.pos = path, cv2.VideoCapture(path), 0
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")

    def frame(self, f: int) -> np.ndarray:
        if f < self.pos:
            raise ValueError(f"frame {f} already passed ({self.pos})")
        while self.pos < f:                                           # grab skips decoding the frames not drawn
            if not self.cap.grab():
                raise SystemExit(f"{self.path} ends before frame {f}")
            self.pos += 1
        ok, img = self.cap.read()
        if not ok:
            raise SystemExit(f"{self.path} ends before frame {f}")
        self.pos += 1
        return img


def save_fixed_camera_gif(images: list[np.ndarray], out: str, fps: int, hold: int = 20, colors: int = 128, keep=()):
    """GIF of footage from a fixed camera, a fraction of the size of a plain GIF. A pixel that changed by less than
    `hold` levels since it was last drawn keeps its drawn value (sensor and codec noise), and every frame shares one
    palette, so the static background repeats exactly and Pillow writes it as transparent runs.
    `keep`: RGB colours given exact palette entries (overlay colours), which median cut over footage would merge."""
    from PIL import Image

    held, frames = images[0].astype(np.int16), []
    for img in images:
        cur = img.astype(np.int16)
        moved = np.abs(cur - held).max(axis=2) >= hold
        held[moved] = cur[moved]
        frames.append(held.astype(np.uint8))
    sample = np.concatenate(frames[:: max(1, len(frames) // 8)], axis=0)
    keep = [tuple(int(v) for v in c) for c in dict.fromkeys(tuple(c) for c in keep)]
    base = Image.fromarray(sample).quantize(colors - len(keep), method=Image.Quantize.MEDIANCUT)
    entries = [v for c in keep for v in c] + base.getpalette()[: 3 * (colors - len(keep))]
    palette = Image.new("P", (1, 1))
    palette.putpalette(entries + [0] * (768 - len(entries)))
    gif = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=round(1000 / fps), loop=0, optimize=True)


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
    ap.add_argument("--crop", nargs=2, type=int, metavar=("Y1", "Y2"),
                    help="rows kept per strip (default 230 520, or the whole frame with --video)")
    ap.add_argument("--video", help="draw over this clip's frames instead of the schematic road")
    ap.add_argument("--start", type=float, default=0.0, help="first second of the clip to render")
    ap.add_argument("--seconds", type=float, help="seconds to render from --start (default: to the end)")
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

    stamps = {d.frame: d.ts_ms for d in dets}
    first, last = min(stamps), max(stamps)
    clip = _Clip(a.video) if a.video else None
    fps = (clip.cap.get(cv2.CAP_PROP_FPS) if clip else 0) or \
        1000 * (last - first) / max(stamps[last] - stamps[first], 1)   # dets ts_ms are truncated: prefer the clip's
    try:
        f0, f1 = frame_window(first, last, fps, a.start, a.seconds)
    except ValueError as exc:
        ap.error(str(exc))
    y1, y2 = a.crop or ([0, 10**6] if clip else [230, 520])
    images = []
    for f, ts, ds in frames(dets):
        if f > f1:
            break
        strips = []
        phase = f0 if clip else 0                                     # over footage the window's first frame is drawn
        shown = f >= f0 and ((f - phase) % a.every == 0 or f == f1)
        bg = clip.frame(f) if clip and shown else None
        for r in runs:
            boxes = r["tracker"].update(f, ts, ds)
            for b in boxes:
                exits = sum(1 for e in r["counter"].update(b) if e.kind == "exit")
                if f >= f0:                                           # counters start at the window
                    r["ids"].add(b.track_id)
                    r["closed"] += exits
            if f == last:                                             # end of stream: close what is still open
                r["closed"] += sum(1 for e in r["counter"].expire(ts + 10**9) if e.kind == "exit")
            if not shown:
                continue
            if bg is not None:
                img = bg.copy()
            else:
                img = np.full((720, W, 3), BG, np.uint8)
                cv2.rectangle(img, (0, 330), (W, 470), ROAD, -1)
                for x in range(0, W, 80):
                    cv2.line(img, (x, 400), (x + 40, 400), LINE, 2)
            draw_zone(img, poly)
            for g in gt.get(f, []):                                   # the real vehicles
                gx1, gy1, gx2, gy2 = map(int, g.bbox)
                if bg is not None:                                    # footage shows the vehicle: outline only
                    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (200, 200, 200), 1)
                else:
                    cv2.rectangle(img, (gx1 + 4, gy1 + 10), (gx2 - 4, gy2), (95, 95, 95), -1)
            if a.occluder:                                            # drawn over the vehicles
                cv2.rectangle(img, (a.occluder[0], 300), (a.occluder[1], 500), (70, 74, 82), -1)
                cv2.putText(img, "pillar", (a.occluder[0] + 22, 292), FONT, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
            for b in boxes:
                bx1, by1, bx2, by2 = map(int, b.bbox)
                c = _palette(b.track_id % TAG_COLOURS if bg is not None else b.track_id)
                cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 3)
                if bg is None:
                    cv2.putText(img, f"#{b.track_id}", (bx1, by1 - 8), FONT, 0.8, c, 2, cv2.LINE_AA)
                    continue
                (tw, th), _ = cv2.getTextSize(f"#{b.track_id}", FONT, 0.9, 2)   # footage: filled tag, readable
                ty = max(by1, y1 + int(40 * img.shape[1] / W) + th + 10)       # on any background, below the header
                cv2.rectangle(img, (bx1, ty - th - 10), (bx1 + tw + 8, ty), c, -1)
                cv2.putText(img, f"#{b.track_id}", (bx1 + 4, ty - 5), FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

            strip = img[y1:y2].copy()
            if strip.shape[1] != W:                                   # e.g. a 1024 px clip: same header layout
                strip = cv2.resize(strip, (W, round(strip.shape[0] * W / strip.shape[1])))
            if bg is not None:
                cv2.putText(strip, f"{ts / 1000:5.1f} s", (W - 130, strip.shape[0] - 16), FONT, 0.8, TEXT, 2,
                            cv2.LINE_AA)
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
    if clip:
        overlay = [_palette(i) for i in range(TAG_COLOURS)] + [ZONE_C, TEXT, (0, 0, 0), (200, 200, 200)]
        save_fixed_camera_gif(images, a.out, a.fps, keep=[c[::-1] for c in overlay])   # BGR -> RGB
    else:
        imageio.mimsave(a.out, images, duration=1 / a.fps, loop=0)
    for r in runs:
        print(f"{r['label']}: IDs {len(r['ids'])}, visits closed {r['closed']}")
    print("wrote", a.out, f"({len(images)} frames)")


if __name__ == "__main__":
    main()
