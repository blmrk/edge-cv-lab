"""Second boxes: one vehicle counted twice because a second detection on it starts a second track, and the steps that
try to stop it, each against the same ByteTrack baseline.

python -m replay.secondbox --dets D --zone Z --truth T --gt G [--ignored I] [--redetected KEY=PATH] [--out-dir DIR] \
    [--tolerance-ms 2000]
    every step in STEPS, and each --redetected file run as the baseline step: the visits table (replay.phantoms'), the
    by-vehicle tables at min_travel_px 0 and 30, the per-detection table, the vehicles each step loses with their
    frame counts, every step's extras, and the baseline step's every enter from its tracks in memory (to set beside
    --tracks on the saved file). Runs in the edge image: the bytetrack tracker needs ultralytics. --out-dir gets each
    step's tracks as <key>.jsonl.
python -m replay.secondbox --tracks TR --gt G --zone Z [--min-travel-px 30]
    the by-vehicle split of saved tracks; runs on the host and creates no tracker.

Steps are alternatives, each one change against the baseline. contain080 and contain090 drop a detection lying 0.8 (0.9)
or more of its own area inside a strictly higher-scoring detection of its frame, the _same steps only when the two carry
the same detector class. They run on every detection as dumped, as ByteTrack sees them, in one pass, so a dropped box
still drops the boxes inside it; equal scores drop neither. The birth rows are replay.phantoms' own.

Per detection (classify). A step's detections are filtered, then masked: a detection centred in an ignored region still
reaches the tracker but is not scored. In each frame, detections and annotated boxes pair one to one, greedily by IoU
(0.5 or more; ties by score, then file order), and a paired detection is its vehicle's. An unpaired one at IoU 0.5 or
more with some box is a duplicate of its best box. Otherwise one with 0.8 or more of its own area inside annotated boxes
belongs to the one of those it overlaps most (a car, not the bus around it): part of that vehicle, or, when no detection
paired with the vehicle in that frame, its best box there is its only box and the rest are part. Anything else is
other, and a frame outside the annotated range is not judged. Second boxes are the duplicates and parts. losses() reads
vehicle-frames, never detections: lost (paired at the baseline, unpaired after), lost and nested in another annotated
box, only lost (no paired or only box after), and fit lost (still paired, IoU down 0.1 or more).

By vehicle (by_vehicle). Truth vehicles are the annotated tracks with an enter of their own, the counter at its defaults
whatever the tracks' counter is. An enter is put on the annotated box its track's box matches in the commit frame (best
IoU, 0.5 or more), else on the one holding 0.8 of it (part), else on none, and the enters are judged in (frame, best IoU
first, track id) order, never the counter's: it returns an enter with its exit. Each takes the first kind that applies:
not judged, no annotated box, no truth visit, part, found (the vehicle's first matching enter), duplicate (another
track's box in that frame matches the vehicle too), again (the same vehicle again, under a new ID or by the track that
found it entering again). Every judged enter not found is an extra, and missed are the truth vehicles no enter found.
extras_diff compares two runs' duplicate, part and again extras by kind, vehicle and time, never by track ID: every
tracker run restarts them.

What it cannot tell apart: a detection on a vehicle whose annotated box was dropped for an ignored region is other, and
a box on a static object under a passing vehicle (a lane dash, a bin) is part or only, as a second box is. The _same
steps read the detector's class, not the vehicle: a car box on a bus is kept inside the bus box. And most changed
detections are boxes no track uses: the enters are the result, the per-detection counts only say where boxes went.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import phantoms
from .between import _area, _inter
from .detections import read_detections
from .motformat import write_tracks
from .schema import read_tracks
from .straddle import _by_frame, mask
from .trackers.greedy_iou import iou
from .zones import DebouncedZoneCounter, run

MATCH, PART, FIT = 0.5, 0.8, 0.1  # IoU that matches an annotated box; own-area share that makes part of one; fit lost


def _inside(a, b) -> float:
    """The share of a's own area that lies inside b."""
    return _inter(a, b) / _area(a) if _area(a) > 0 else 0.0


def contained(dets, share: float, same_class: bool = False):
    """dets without each detection lying `share` or more of its own area inside a strictly higher-scoring detection of
    its frame (of its class, with same_class). One pass: a dropped detection still contains. Input order is kept."""
    by = _by_frame(dets)
    return [d for d in dets if not any(c.score > d.score and (not same_class or c.cls == d.cls)
                                       and _inside(d.bbox, c.bbox) >= share for c in by[d.frame])]


def _holder(box, frame_gt):
    """The annotated box holding PART or more of box's own area that overlaps it most (in a car nested in a bus, the
    car), or None."""
    return max((g for g in frame_gt if _inside(box, g.bbox) >= PART), key=lambda g: iou(box, g.bbox), default=None)


def classify(dets, gt) -> list[tuple[str, int | None, float]]:
    """(label, vehicle id, IoU with that vehicle's box) per detection, in input order. dets: already masked."""
    g_at, d_at = _by_frame(gt), defaultdict(list)
    for i, d in enumerate(dets):
        d_at[d.frame].append(i)
    lo, hi = (min(g.frame for g in gt), max(g.frame for g in gt)) if gt else (0, -1)
    out = [("not judged", None, 0.0)] * len(dets)
    for f, idx in d_at.items():
        if not lo <= f <= hi:
            continue
        frame_gt, paired, used, parts = g_at[f], {}, set(), defaultdict(list)
        pairs = sorted(((iou(dets[i].bbox, g.bbox), dets[i].score, -i, -k, i, g) for i in idx
                        for k, g in enumerate(frame_gt) if iou(dets[i].bbox, g.bbox) >= MATCH),
                       key=lambda p: p[:4], reverse=True)  # IoU, then score, then file order
        for *_, i, g in pairs:
            if i not in paired and g not in used:
                paired[i] = g
                used.add(g)
        for i in idx:
            b = dets[i].bbox
            best = max(frame_gt, key=lambda g: iou(b, g.bbox), default=None)
            if i in paired:
                out[i] = ("vehicle", paired[i].track_id, iou(b, paired[i].bbox))
            elif best is not None and iou(b, best.bbox) >= MATCH:
                out[i] = ("duplicate", best.track_id, iou(b, best.bbox))
            elif (h := _holder(b, frame_gt)) is not None:
                parts[h].append(i)
            else:
                out[i] = ("other", None, 0.0)
        for h, ii in parts.items():
            ii.sort(key=lambda i: (-iou(dets[i].bbox, h.bbox), -dets[i].score, i))
            for n, i in enumerate(ii):
                out[i] = ("only" if n == 0 and h not in used else "part", h.track_id, iou(dets[i].bbox, h.bbox))
    return out


def cover(dets, labels) -> tuple[dict, set]:
    """({(frame, vehicle): IoU of its paired detection}, {(frame, vehicle) whose box has an "only" detection})."""
    paired = {(d.frame, v): i for d, (label, v, i) in zip(dets, labels) if label == "vehicle"}
    return paired, {(d.frame, v) for d, (label, v, _) in zip(dets, labels) if label == "only"}


def nested(gt) -> set[tuple[int, int]]:
    """(frame, vehicle) for every annotated box lying PART or more inside another annotated box of its frame."""
    return {(g.frame, g.track_id) for boxes in _by_frame(gt).values() for g in boxes
            if any(o is not g and _inside(g.bbox, o.bbox) >= PART for o in boxes)}


def losses(base_cover, step_cover, nested_vf) -> dict[str, set]:
    """Vehicle-frames a step loses against the baseline, as sets of (frame, vehicle): no detection is matched by
    identity, so a vehicle-frame is lost only when no detection pairs with it any more."""
    (base, base_only), (step, step_only) = base_cover, step_cover
    lost = set(base) - set(step)
    return {"lost": lost, "lost_nested": lost & nested_vf, "only_lost": base_only - set(step) - step_only,
            "fit_lost": {k for k in set(base) & set(step) if base[k] - step[k] >= FIT}}


def by_vehicle(tracks, gt, poly, **counter_kw) -> tuple[list[dict], list[int]]:
    """(one row per debounced enter, the truth vehicles no enter found). Truth vehicles: the annotated tracks with an
    enter of their own, counter at its defaults whatever counter_kw is; counter_kw goes to the tracks' counter alone.
    Rows: track_id, frame, ts_ms, kind, vehicle, iou, in (frame, -iou, track_id) order. Reads no detections."""
    gt = sorted(gt, key=lambda b: (b.frame, b.track_id))
    truth = {e.track_id for e in run(DebouncedZoneCounter(poly), gt) if e.kind == "enter"}
    lo, hi = (gt[0].frame, gt[-1].frame) if gt else (0, -1)
    tracks = sorted(tracks, key=lambda b: (b.frame, b.track_id))
    g_at, t_at = _by_frame(gt), _by_frame(tracks)

    def on(box, frame):  # the annotated box `box` matches: its best, at IoU MATCH or more
        g = max(g_at[frame], key=lambda g: iou(box, g.bbox), default=None)
        return g if g is not None and iou(box, g.bbox) >= MATCH else None

    rows = []
    for e in (e for e in run(DebouncedZoneCounter(poly, **counter_kw), tracks) if e.kind == "enter"):
        box = next(t.bbox for t in t_at[e.frame] if t.track_id == e.track_id)  # the commit frame's box
        g, part = on(box, e.frame), False
        if g is None:
            g = _holder(box, g_at[e.frame])
            part = g is not None
        rows.append({"track_id": e.track_id, "frame": e.frame, "ts_ms": e.ts_ms, "part": part,
                     "vehicle": g.track_id if g else None, "iou": iou(box, g.bbox) if g else 0.0})
    rows.sort(key=lambda r: (r["frame"], -r["iou"], r["track_id"]))
    found = set()
    for r in rows:
        v, part = r["vehicle"], r.pop("part")
        if not lo <= r["frame"] <= hi:
            r["kind"] = "not judged"
        elif v is None:
            r["kind"] = "no annotated box"
        elif v not in truth:
            r["kind"] = "no truth visit"
        elif part:
            r["kind"] = "part"
        elif v not in found:
            r["kind"] = "found"
            found.add(v)
        elif any(t.track_id != r["track_id"] and (o := on(t.bbox, r["frame"])) and o.track_id == v
                 for t in t_at[r["frame"]]):
            r["kind"] = "duplicate"
        else:
            r["kind"] = "again"
    return rows, sorted(truth - found)


SECOND = ("duplicate", "part", "again")


def extras_diff(base_rows, step_rows, tol_ms: int = 2000) -> tuple[list[dict], list[dict]]:
    """(baseline extras the step removes, step extras that are new), over the kinds in SECOND. Two extras are the same
    when kind and vehicle agree and they are tol_ms or less apart; track ids never count, as each tracker run restarts
    them."""
    def unmatched(rows, others):
        return [r for r in rows if r["kind"] in SECOND and not any(
            o["kind"] == r["kind"] and o["vehicle"] == r["vehicle"] and abs(o["ts_ms"] - r["ts_ms"]) <= tol_ms
            for o in others)]
    return unmatched(base_rows, step_rows), unmatched(step_rows, base_rows)


STEPS = [  # (key, step, bytetrack params, detection filter, counter params), as replay.phantoms.STEPS
    phantoms.STEPS[0],
    ("contain080", "dropped: 0.8 inside a higher-scoring box", {}, lambda ds: contained(ds, 0.8), {}),
    ("contain090", "dropped: 0.9 inside a higher-scoring box", {}, lambda ds: contained(ds, 0.9), {}),
    ("contain080_same", "dropped: 0.8 inside a higher-scoring box of its class", {},
     lambda ds: contained(ds, 0.8, same_class=True), {}),
    ("contain090_same", "dropped: 0.9 inside a higher-scoring box of its class", {},
     lambda ds: contained(ds, 0.9, same_class=True), {}),
    phantoms.STEPS[3],
    phantoms.STEPS[4],
]
KINDS = ("found", "duplicate", "part", "again", "no truth visit", "no annotated box", "not judged")
LABELS = ("vehicle", "duplicate", "part", "only", "other", "not judged")


def _items(rows) -> str:
    return ", ".join(f"{r['track_id']} on {r['vehicle']}" if r["vehicle"] is not None else str(r["track_id"])
                     for r in rows)


def _kind(rows, kind) -> str:
    hit = [r for r in rows if r["kind"] == kind]
    return f"{len(hit)}: {_items(hit)}" if hit else "0"


def _extras(rows) -> list[dict]:
    return [r for r in rows if r["kind"] not in ("found", "not judged")]


def _by_kind(rows) -> str:
    return "; ".join(f"{k} {_items([r for r in rows if r['kind'] == k])}" for k in SECOND
                     if any(r["kind"] == k for r in rows)) or "0"


def _table(title, cols, rows) -> None:
    print(f"\n{title}\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def _per_vehicle(vf) -> str:
    """'7: 2, 8: 1': each vehicle in a set of (frame, vehicle), with its number of frames."""
    n = Counter(v for _, v in vf)
    return ", ".join(f"{v}: {n[v]}" for v in sorted(n)) or "0"


def _enter_rows(rows) -> list[list]:
    return [[r["track_id"], r["frame"], r["ts_ms"], r["kind"], r["vehicle"], round(r["iou"], 2)] for r in rows]


def _tracks_mode(a, gt, poly) -> None:
    rows, missed = by_vehicle(list(read_tracks(a.tracks)), gt, poly, min_travel_px=a.min_travel_px)
    found = {r["vehicle"] for r in rows if r["kind"] == "found"}
    print(f"truth vehicles {len(found) + len(missed)} | enters {len(rows)} | min_travel_px {a.min_travel_px:g}")
    for k in KINDS:
        print(f"{k} {_kind(rows, k)}")
    print(f"extra {len(_extras(rows))} | missed {missed}")
    _table("enters, in judging order", ["track id", "frame", "ts ms", "kind", "vehicle", "iou"], _enter_rows(rows))


def _dets_mode(a, gt, poly) -> None:
    every, truth = read_detections(a.dets), json.load(open(a.truth))["enters_ms"]
    regions = json.load(open(a.ignored))["regions_xyxy"] if a.ignored else []
    queue = (read_detections(phantoms.FX / "queue.dets.jsonl"),
             [tuple(p) for p in json.load(open(phantoms.FX / "zone.json"))["polygon"]],
             json.load(open(phantoms.FX / "queue.truth.json"))["enters_ms"])
    rows, tracks = phantoms.table(every, poly, truth, queue, a.tolerance_ms, steps=STEPS)  # tracked: every detection
    keys = [s[0] for s in STEPS]
    scored = {key: None if keep is None and params else mask(keep(every) if keep else every, regions)
              for key, _, params, keep, _ in STEPS}  # None: a tracker setting, the detections are the baseline's
    for spec in a.redetected:
        key, path = spec.split("=", 1)
        redets = read_detections(path)
        (row,), t = phantoms.table(redets, poly, truth, None, a.tolerance_ms, steps=[phantoms.STEPS[0]])
        rows.append({**row, "step": f"baseline step on {path}", "queue_matched": "n/a", "queue_false": "n/a"})
        keys.append(key)
        tracks[key], scored[key] = t["baseline"], mask(redets, regions)
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        for key in keys:
            write_tracks(tracks[key], Path(a.out_dir) / f"{key}.jsonl")

    cols = [c for c in rows[0] if c != "step"]
    _table(f"visits (against --truth, tolerance {a.tolerance_ms} ms)",
           ["key", "step"] + [c.replace("_visits", "").replace("_", " ") for c in cols],
           [[k, r["step"], *(r[c] for c in cols)] for k, r in zip(keys, rows)])

    split = {px: {k: by_vehicle(tracks[k], gt, poly, min_travel_px=px) for k in keys} for px in (0, 30)}
    for px, by in split.items():
        base = by["baseline"][0]
        truth_n = len({r["vehicle"] for r in base if r["kind"] == "found"}) + len(by["baseline"][1])
        out = []
        for k, (vrows, missed) in by.items():
            removed, new = extras_diff(base, vrows, a.tolerance_ms)
            out.append([k, sum(r["kind"] == "found" for r in vrows), missed, len(_extras(vrows)),
                        *(_kind(vrows, kind) for kind in KINDS[1:]), _by_kind(removed), _by_kind(new)])
        _table(f"by vehicle, min_travel_px {px}: {truth_n} truth vehicles; removed and new against baseline at {px} px",
               ["key", "found", "missed", "extra", *KINDS[1:], "removed", "new"], out)

    base_dets = scored["baseline"]
    base_labels = classify(base_dets, gt)
    base_n, base_cover, nest = Counter(x[0] for x in base_labels), cover(base_dets, base_labels), nested(gt)
    out, lost_by = [], {}
    for k in keys:
        if scored[k] is None:
            out.append([k, "unchanged (tracker setting)"] + [""] * 11)
            continue
        labels = classify(scored[k], gt)
        n, lost = Counter(x[0] for x in labels), losses(base_cover, cover(scored[k], labels), nest)
        lost_by[k] = lost
        out.append([k, *(n[x] for x in LABELS), f"{n['duplicate'] - base_n['duplicate']:+d}",
                    f"{n['part'] - base_n['part']:+d}", *(len(v) for v in lost.values())])
    _table(f"per detection (masked after the filter; nested annotated vehicle-frames {len(nest)})",
           ["key", *LABELS, "duplicate change", "part change", "lost", "lost nested", "only lost", "fit lost"], out)
    _table("vehicle-frames lost against the baseline, by vehicle (vehicle: frames)",
           ["key", *(name.replace("_", " ") for name in lost_by["baseline"])],
           [[k, *(_per_vehicle(vf) for vf in lost.values())] for k, lost in lost_by.items()])

    for px, by in split.items():
        _table(f"extras, min_travel_px {px}", ["key", "track id", "frame", "ts ms", "kind", "vehicle", "iou"],
               [[k, *r] for k, (vrows, _) in by.items() for r in _enter_rows(_extras(vrows))])
    for px, by in split.items():  # every baseline enter, found too, from the tracks in memory: to set beside --tracks
        _table(f"baseline enters, in judging order, min_travel_px {px}",
               ["track id", "frame", "ts ms", "kind", "vehicle", "iou"], _enter_rows(by["baseline"][0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dets", help="detections JSONL: run every step (edge image)")
    mode.add_argument("--tracks", help="saved tracks JSONL: the by-vehicle split alone (host, no tracker)")
    ap.add_argument("--gt", required=True, help="annotated boxes JSONL (scripts/detrac_to_gt.py, mtid_to_gt.py)")
    ap.add_argument("--zone", required=True)
    ap.add_argument("--truth", help="with --dets: truth.json (enters_ms) for the visits table")
    ap.add_argument("--ignored", help='with --dets: {"regions_xyxy": [...]}; detections centred there are not scored')
    ap.add_argument("--redetected", action="append", default=[], metavar="KEY=PATH",
                    help="with --dets: another detection file, run as the baseline step and listed last as KEY")
    ap.add_argument("--out-dir", help="with --dets: write each step's tracks here as <key>.jsonl")
    ap.add_argument("--tolerance-ms", type=int, default=2000)
    ap.add_argument("--min-travel-px", type=float, default=0, help="with --tracks: the counter's zone rule (0: off)")
    a = ap.parse_args()
    if a.dets and not a.truth:
        ap.error("--dets needs --truth")
    keys = [r.split("=", 1)[0] for r in a.redetected]
    if any("=" not in r for r in a.redetected) or len(set(keys)) < len(keys) or set(keys) & {s[0] for s in STEPS}:
        ap.error("--redetected takes KEY=PATH, each KEY new and none a step's own key")
    gt, poly = list(read_tracks(a.gt)), [tuple(p) for p in json.load(open(a.zone))["polygon"]]
    (_dets_mode if a.dets else _tracks_mode)(a, gt, poly)


if __name__ == "__main__":
    main()
