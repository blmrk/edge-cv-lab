# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""HOTA, IDF1 and ID switches from TrackEval for replay tracks against ground-truth tracks (e.g. mtid_to_gt.py output).

TrackEval is not a dependency. Clone it once, at the pinned commit (MIT licence); it needs numpy and scipy:
  pip install -e ".[trackeval]"
  git clone https://github.com/JonathonLuiten/TrackEval ../.cache/TrackEval && git -C ../.cache/TrackEval checkout 12c8791

python scripts/trackeval_run.py --gt runs/mtid.gt.jsonl --tracks bytetrack=runs/bytetrack.jsonl greedy_iou_5=runs/greedy_iou_5.jsonl

Vehicles are scored as MOTChallenge class 1 with TrackEval's pedestrian preprocessing off: the class only picks which
rows count, so this runs its 2D box metrics unchanged on vehicles. HOTA, DetA and AssA are means over TrackEval's
localisation thresholds (IoU 0.05 to 0.95); IDF1, MOTA and IDSW are at IoU 0.5. Only the ground truth's frame range
(first to last annotated frame) is scored: tracks past it would count as false positives in frames nobody annotated.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay.motformat import export_gt, export_tracks  # noqa: E402
from replay.schema import read_tracks  # noqa: E402

TRACKEVAL = Path(__file__).resolve().parents[2] / ".cache" / "TrackEval"
COLS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]


def _trackeval(path: Path):
    if not (path / "trackeval").is_dir():
        raise SystemExit(f"TrackEval not found at {path}: clone it first (see this script's docstring)")
    import numpy as np

    # ponytail: TrackEval 12c8791 still uses np.int / np.float / np.bool, removed in numpy 1.24. They were aliases of
    # the builtins, so restoring them is exact; drop this once TrackEval is patched upstream.
    for name, builtin in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, builtin)
    sys.path.insert(0, str(path))
    import trackeval
    return trackeval


def evaluate(gt, runs: dict, trackeval_dir: Path = TRACKEVAL) -> dict:
    """runs: {name: [TrackBox]}; returns {name: {HOTA, DetA, AssA, IDF1, MOTA, IDSW}} over the one sequence."""
    te = _trackeval(Path(trackeval_dir))
    import numpy as np

    seq = "clip"
    first, last = min(b.frame for b in gt), max(b.frame for b in gt)
    runs = {name: [b for b in boxes if first <= b.frame <= last] for name, boxes in runs.items()}
    length = last + 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "gt" / seq / "gt").mkdir(parents=True)
        export_gt(gt, root / "gt" / seq / "gt" / "gt.txt")
        for name, boxes in runs.items():
            (root / "trackers" / name / "data").mkdir(parents=True)
            export_tracks(boxes, root / "trackers" / name / "data" / f"{seq}.txt")
        quiet = {"PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False,
                 "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "LOG_ON_ERROR": None}
        dataset = te.datasets.MotChallenge2DBox({
            "GT_FOLDER": str(root / "gt"), "TRACKERS_FOLDER": str(root / "trackers"), "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {seq: length}, "DO_PREPROC": False, "TRACKERS_TO_EVAL": list(runs), "PRINT_CONFIG": False})
        res, _ = te.Evaluator(quiet).evaluate([dataset], [te.metrics.HOTA(), te.metrics.CLEAR(), te.metrics.Identity()])
    out = {}
    for name in runs:
        r = res["MotChallenge2DBox"][name]["COMBINED_SEQ"]["pedestrian"]
        out[name] = {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "DetA": float(np.mean(r["HOTA"]["DetA"])),
                     "AssA": float(np.mean(r["HOTA"]["AssA"])), "IDF1": float(r["Identity"]["IDF1"]),
                     "MOTA": float(r["CLEAR"]["MOTA"]), "IDSW": int(r["CLEAR"]["IDSW"])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", required=True, help="ground-truth tracks JSONL")
    ap.add_argument("--tracks", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--trackeval", default=str(TRACKEVAL), help="TrackEval clone")
    a = ap.parse_args()
    runs = {}
    for spec in a.tracks:
        name, sep, path = spec.rpartition("=")  # last "=": tracker names like greedy_iou:max_age=5 hold one
        if not sep:
            ap.error(f"--tracks takes NAME=PATH, got {spec!r}")
        runs[name] = list(read_tracks(path))
    res = evaluate(list(read_tracks(a.gt)), runs, Path(a.trackeval))
    print("| tracker | " + " | ".join(COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for name, r in res.items():
        print(f"| {name} | " + " | ".join(str(r[c]) if c == "IDSW" else f"{r[c]:.3f}" for c in COLS) + " |")


if __name__ == "__main__":
    main()
