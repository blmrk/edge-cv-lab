# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""Animated tracker comparison: one strip per tracker, same detections, colour = track ID.

Grey shapes are the real vehicles (ground truth), drawn even when the detector cannot see them.
Coloured boxes are what each tracker reports. Each strip shows IDs used, visits closed, and the
dwell timer of any visit still open, so a merged visit is visible as a timer that never resets.

python -m replay.trackviz --dets fixtures/queue.dets.jsonl --gt fixtures/queue.gt.jsonl --zone fixtures/zone.json \
    --trackers "greedy_iou:max_age=5" greedy_iou groundplane --occluder 440 560 --out ../docs/img/trackers.gif

Over real footage: --video draws each frame on the matching frame of the clip the detections came from (frame f is
the (f+1)-th frame read, as dump_detections numbers them), and --start/--seconds render a window of it. Trackers
still run from the first frame, so IDs match a full run; the strip counters start at the window.

Fixes per run, one value per tracker in --trackers order: --filters drops detections before that run's tracker (none,
or a second-box filter of replay.secondbox such as contain090_same), --min-travel-px sets that run's zone rule (0: off).
A run's label is followed by what was applied: "bytetrack + contain090_same + zone rule 30 px". In a run with the zone
rule, a track it holds back (not yet moved that far, so its enter cannot commit) is drawn grey and tagged "#ID held".
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

from . import secondbox
from .detections import frame_window, frames, read_detections
from .schema import read_tracks
from .trackers import create
from .viz import BG, FONT, LINE, ROAD, TEXT, ZONE_C, _palette, draw_zone

W = 1280
TAG_COLOURS = 24  # over footage, ID colours cycle through this many hues so each gets an exact GIF palette entry
GREY = (200, 200, 200)  # ground-truth outlines over footage, and tracks the zone rule holds back; a reserved GIF entry
FILTERS = {k: keep for k, _, _, keep, _ in secondbox.STEPS if k.startswith("contain")}  # frame-local: run per frame


def label_scale(label: str) -> float:
    """Font scale of a strip's header label: 0.8, or less for a label that would run into the stats at x=470."""
    return min(0.8, 0.8 * 440 / max(cv2.getTextSize(label, FONT, 0.8, 2)[0][0], 1))


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


def overlay_rgba(on_black: np.ndarray, on_white: np.ndarray) -> np.ndarray:
    """The same overlay drawn on black and on white, as one straight-alpha RGBA image: what shows of the white is
    what the overlay lets through (blended zone fill, anti-aliased text and edges), so alpha needs no drawing call
    of its own."""
    p = on_black.astype(np.int32)
    a = 255 - np.clip(on_white.astype(np.int32) - p, 0, 255).max(axis=2, keepdims=True)
    rgb = np.where(a > 0, (p * 255 + a // 2) // np.maximum(a, 1), 0)
    return np.concatenate([np.clip(rgb, 0, 255), a], axis=2).astype(np.uint8)


def _over(bg: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:].astype(np.int32)
    return ((bg.astype(np.int32) * (255 - a) + rgba[..., :3].astype(np.int32) * a + 127) // 255).astype(np.uint8)


def save_fixed_camera_gif(images: list[np.ndarray], out: str, fps: int, hold: int = 20, colors: int = 128, keep=(),
                          overlays: list[np.ndarray] | None = None):
    """GIF of footage from a fixed camera, a fraction of the size of a plain GIF. A pixel that changed by less than
    `hold` levels since it was last drawn keeps its drawn value (sensor and codec noise), and every frame shares one
    palette, so the static background repeats exactly and Pillow writes it as transparent runs.
    `keep`: RGB colours given exact palette entries (overlay colours), which a palette built over footage would merge.
    `overlays`: one RGBA image per frame (straight alpha), laid over `images` after the hold. The hold then sees the
    footage alone and every overlay pixel is exact in every frame; held over drawn frames instead, a box edge stays
    behind wherever the footage now under it is within `hold` of the edge's colour. With overlays, footage pixels
    (alpha under half) map to footage entries only: a green vehicle no box is on would otherwise take the nearest
    reserved tag green and show bright speckles. Those entries are built from what footage pixels show (zone fill
    included, opaque overlay left out) by octree, not median cut: median cut splits the many greys first, so a small
    strongly coloured vehicle would get no entry and, barred from the tag colours, show grey."""
    from PIL import Image

    if overlays is not None and len(overlays) != len(images):
        raise ValueError(f"{len(overlays)} overlays for {len(images)} frames")

    def palette_image(entries):
        p = Image.new("P", (1, 1))
        p.putpalette(entries)
        return p

    held, frames = images[0].astype(np.int16), []
    for img in images:
        cur = img.astype(np.int16)
        moved = np.abs(cur - held).max(axis=2) >= hold
        held[moved] = cur[moved]
        frames.append(held.astype(np.uint8))
    step = max(1, len(frames) // 8)
    if overlays is None:
        sample, method = np.concatenate(frames[::step], axis=0), Image.Quantize.MEDIANCUT
    else:   # what footage pixels will show: zone fill blended in, opaque overlay left out
        sample = np.concatenate([_over(f, ov)[ov[..., 3] < 128] for f, ov in zip(frames[::step], overlays[::step])])
        sample, method = sample[:, None], Image.Quantize.FASTOCTREE
    keep = [tuple(int(v) for v in c) for c in dict.fromkeys(tuple(c) for c in keep)]
    base = Image.fromarray(sample).quantize(colors - len(keep), method=method)
    footage = base.getpalette()[: 3 * (colors - len(keep))]
    entries = [v for c in keep for v in c] + footage
    entries += [0] * (768 - len(entries))
    palette = palette_image(entries)
    if overlays is None:
        gif = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    else:
        under, gif = palette_image(footage), []
        for f, ov in zip(frames, overlays):
            drawn = Image.fromarray(_over(f, ov))
            idx = np.where(ov[..., 3] >= 128, np.asarray(drawn.quantize(palette=palette, dither=Image.Dither.NONE)),
                           np.asarray(drawn.quantize(palette=under, dither=Image.Dither.NONE)) + len(keep))
            gif.append(Image.fromarray(idx.astype(np.uint8), "P"))
            gif[-1].putpalette(entries)
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=round(1000 / fps), loop=0, optimize=True)


def draw_tag(img: np.ndarray, bx1: int, by1: int, tid: int, colour, top: int, note: str = ""):
    """Filled ID tag, readable on any footage, sitting on the box's visible top edge: by1, or `top`, the strip's first
    row, for a box cut by the crop. Where that edge is under the strip header (40 px at W wide), or too close below it
    for the tag to fit above, the tag goes just below the header, still at the box's left. `note` follows the ID."""
    text = f"#{tid} {note}" if note else f"#{tid}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 2)
    header = top + int(40 * img.shape[1] / W)                         # a narrower clip is scaled up to W after this
    ty = max(max(by1, top), header + th + 10)                         # the tag's bottom row
    cv2.rectangle(img, (bx1, ty - th - 10), (bx1 + tw + 8, ty), colour, -1)
    cv2.putText(img, text, (bx1 + 4, ty - 5), FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)


def _crop(img: np.ndarray, y1: int, y2: int) -> np.ndarray:
    strip = img[y1:y2].copy()
    if strip.shape[1] != W:                                           # e.g. a 1024 px clip: same header layout
        strip = cv2.resize(strip, (W, round(strip.shape[0] * W / strip.shape[1])))
    return strip


def _rgb(strips: list[np.ndarray], width: int) -> np.ndarray:
    frame = np.vstack(strips)
    h = int(frame.shape[0] * width / W)
    return cv2.cvtColor(cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def main():
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
    ap.add_argument("--filters", nargs="+", choices=["none", *FILTERS],
                    help="detection filter per tracker, same order as --trackers, applied before it (default: none)")
    ap.add_argument("--min-travel-px", nargs="+", type=float,
                    help="zone rule per tracker, same order as --trackers: an enter waits until the track has moved "
                         "this far (default: 0, off)")
    a = ap.parse_args()
    n = len(a.trackers)
    for flag, given, off in (("--filters", a.filters, "none for no filter"),
                             ("--min-travel-px", a.min_travel_px, "0 for off")):
        if given is not None and len(given) != n:
            ap.error(f"{flag}: {len(given)} given for {n} trackers; give one per tracker, {off}")
    filters, travel = a.filters or ["none"] * n, a.min_travel_px or [0] * n

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
        fixes = [filters[i]] * (filters[i] != "none") + [f"zone rule {travel[i]:g} px"] * bool(travel[i])
        label = " + ".join([(a.labels or a.trackers)[i], *fixes])
        runs.append({"label": label, "tracker": create(name, **params), "keep": FILTERS.get(filters[i]),
                     "counter": DebouncedZoneCounter(poly, min_travel_px=travel[i]), "ids": set(), "closed": 0})

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

    def draw(img, r, boxes, f, ts, footage):
        """One strip: what is drawn over the road (the schematic, or a blank canvas over footage), cropped, headed."""
        draw_zone(img, poly)
        for g in gt.get(f, []):                                       # the real vehicles
            gx1, gy1, gx2, gy2 = map(int, g.bbox)
            if footage:                                               # footage shows the vehicle: outline only
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), GREY, 1)
            else:
                cv2.rectangle(img, (gx1 + 4, gy1 + 10), (gx2 - 4, gy2), (95, 95, 95), -1)
        if a.occluder:                                                # drawn over the vehicles
            cv2.rectangle(img, (a.occluder[0], 300), (a.occluder[1], 500), (70, 74, 82), -1)
            cv2.putText(img, "pillar", (a.occluder[0] + 22, 292), FONT, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        held_first = sorted(boxes, key=lambda b: not r["counter"].held(b.track_id))  # a wider held tag never covers a vehicle's
        for b in held_first:
            bx1, by1, bx2, by2 = map(int, b.bbox)
            held = r["counter"].held(b.track_id)                      # this run's zone rule holds its enter back
            c = GREY if held else _palette(b.track_id % TAG_COLOURS if footage else b.track_id)
            note = "held" if held else ""
            cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 3)
            if footage:
                draw_tag(img, bx1, by1, b.track_id, c, y1, note)
            else:
                cv2.putText(img, f"#{b.track_id} {note}".rstrip(), (bx1, by1 - 8), FONT, 0.8, c, 2, cv2.LINE_AA)
        strip = _crop(img, y1, y2)
        if footage:
            cv2.putText(strip, f"{ts / 1000:5.1f} s", (W - 130, strip.shape[0] - 16), FONT, 0.8, TEXT, 2, cv2.LINE_AA)
        cv2.rectangle(strip, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(strip, r["label"], (16, 28), FONT, label_scale(r["label"]), TEXT, 2, cv2.LINE_AA)
        stats = f"IDs used {len(r['ids'])}    visits closed {r['closed']}"
        cv2.putText(strip, stats, (470, 28), FONT, 0.7, TEXT, 1, cv2.LINE_AA)
        for tid, secs in r["counter"].open_visits(ts)[:1]:
            col = (70, 80, 235) if secs > 12 else ZONE_C
            cv2.putText(strip, f"open visit #{tid}: {secs:4.1f} s", (930, 28), FONT, 0.7, col, 2, cv2.LINE_AA)
        return strip

    images, overlays = [], []     # over footage: images is the footage alone, overlays what is drawn over it
    for f, ts, ds in frames(dets):
        if f > f1:
            break
        strips = []
        phase = f0 if clip else 0                                     # over footage the window's first frame is drawn
        shown = f >= f0 and ((f - phase) % a.every == 0 or f == f1)
        bg = clip.frame(f) if clip and shown else None
        for r in runs:
            boxes = r["tracker"].update(f, ts, r["keep"](ds) if r["keep"] else ds)
            for b in boxes:
                exits = sum(1 for e in r["counter"].update(b) if e.kind == "exit")
                if f >= f0:                                           # counters start at the window
                    r["ids"].add(b.track_id)
                    r["closed"] += exits
            if f == last:                                             # end of stream: close what is still open
                r["closed"] += sum(1 for e in r["counter"].expire(ts + 10**9) if e.kind == "exit")
            if not shown:
                continue
            if bg is not None:                                        # drawn on black and on white: see overlay_rgba
                strips.append([draw(np.full_like(bg, v), r, boxes, f, ts, True) for v in (0, 255)])
                continue
            img = np.full((720, W, 3), BG, np.uint8)
            cv2.rectangle(img, (0, 330), (W, 470), ROAD, -1)
            for x in range(0, W, 80):
                cv2.line(img, (x, 400), (x + 40, 400), LINE, 2)
            strips.append(draw(img, r, boxes, f, ts, False))
        if not strips:
            continue
        if bg is None:
            images.append(_rgb(strips, a.width))
        else:
            images.append(_rgb([_crop(bg, y1, y2)] * len(strips), a.width))
            overlays.append(overlay_rgba(*(_rgb([s[i] for s in strips], a.width) for i in (0, 1))))

    images += [images[-1]] * (a.fps * 2)                              # hold the final frame
    overlays += overlays[-1:] * (a.fps * 2)
    if clip:
        overlay = [_palette(i) for i in range(TAG_COLOURS)] + [ZONE_C, TEXT, (0, 0, 0), GREY]
        save_fixed_camera_gif(images, a.out, a.fps, keep=[c[::-1] for c in overlay], overlays=overlays)  # BGR -> RGB
    else:
        import imageio.v2 as imageio                                  # the schematic GIF only

        imageio.mimsave(a.out, images, duration=1 / a.fps, loop=0)
    for r in runs:
        print(f"{r['label']}: IDs {len(r['ids'])}, visits closed {r['closed']}")
    print("wrote", a.out, f"({len(images)} frames)")


if __name__ == "__main__":
    main()
