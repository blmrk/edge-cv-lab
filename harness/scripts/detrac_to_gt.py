"""UA-DETRAC XML annotations -> harness ground truth.

python scripts/detrac_to_gt.py --xml ../media/UA-DETRAC/DETRAC-Train-Annotations-XML/MVI_20011.xml --out runs/MVI_20011 [--zone zone.json]

Writes:
  <out>.gt.jsonl        ground-truth tracks in the harness TrackBox schema (for --gt in replay.compare)
  <out>.ignored.json    the sequence's ignored regions, so you can keep your zone away from them
  <out>.truth.json      only with --zone: visit truth derived by running the debounced counter over the
                        annotated tracks (perfect IDs). Good for scoring trackers; not a substitute for
                        hand labels if you are evaluating the counter itself.

Format, as given in the dataset documentation (not yet checked against a downloaded file):
  <sequence name="MVI_20011">
    <ignored_region><box left=".." top=".." width=".." height=".."/></ignored_region>
    <frame num="1"><target_list><target id="1"><box left=".." top=".." width=".." height=".."/>
      <attribute vehicle_type="car" .../></target></target_list></frame>
UA-DETRAC is 25 fps, 960x540. Frame numbers are 1-based; the harness is 0-based.
Licence: UA-DETRAC is for research use. Publish metrics and a citation, not frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay.schema import TrackBox  # noqa: E402

FPS = 25


def _box(el) -> tuple[float, float, float, float]:
    l, t = float(el.get("left")), float(el.get("top"))
    return (l, t, l + float(el.get("width")), t + float(el.get("height")))


def _inside(b, region) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def convert(xml_path: Path, fps: float = FPS):
    root = ET.parse(xml_path).getroot()
    ignored = [_box(b) for b in root.findall("./ignored_region/box")]
    tracks: list[TrackBox] = []
    dropped = 0
    for fr in root.findall("frame"):
        f = int(fr.get("num")) - 1
        for tg in fr.findall("./target_list/target"):
            box_el = tg.find("box")
            if box_el is None:
                continue
            b = _box(box_el)
            if any(_inside(b, r) for r in ignored):
                dropped += 1
                continue
            attr = tg.find("attribute")
            cls = attr.get("vehicle_type", "car") if attr is not None else "car"
            tracks.append(TrackBox(f, int(f * 1000 / fps), int(tg.get("id")), b, 1.0, cls))
    tracks.sort(key=lambda b: (b.frame, b.track_id))
    return tracks, ignored, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True, help="output stem, e.g. runs/MVI_20011")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--zone", help="zone.json; also emit <out>.truth.json from the annotated tracks")
    a = ap.parse_args()

    tracks, ignored, dropped = convert(Path(a.xml), a.fps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.gt.jsonl", "w") as fh:
        for t in tracks:
            fh.write(json.dumps({"frame": t.frame, "ts_ms": t.ts_ms, "track_id": t.track_id,
                                 "bbox": [round(v, 1) for v in t.bbox], "score": 1.0, "cls": t.cls}) + "\n")
    Path(f"{out}.ignored.json").write_text(json.dumps({"regions_xyxy": ignored}))
    n_ids = len({t.track_id for t in tracks})
    print(f"{out}.gt.jsonl: {len(tracks)} boxes, {n_ids} tracks, {dropped} boxes dropped in ignored regions")

    if a.zone:
        from replay.zones import DebouncedZoneCounter, run
        poly = [tuple(p) for p in json.load(open(a.zone))["polygon"]]
        enters = [e.ts_ms for e in run(DebouncedZoneCounter(poly), tracks) if e.kind == "enter"]
        Path(f"{out}.truth.json").write_text(json.dumps(
            {"source": str(a.xml), "derived_from": "annotated tracks + DebouncedZoneCounter defaults",
             "expected_visits": len(enters), "enters_ms": sorted(enters)}, indent=2))
        print(f"{out}.truth.json: {len(enters)} visits")


if __name__ == "__main__":
    main()
