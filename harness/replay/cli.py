"""python -m replay.cli --tracks fixtures/boundary_jitter.jsonl --zone fixtures/zone.json [--expected 1]"""
from __future__ import annotations

import argparse
import json

from .metrics import count_error, summarize
from .schema import read_tracks
from .zones import DebouncedZoneCounter, NaiveZoneCounter, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--zone", required=True, help='JSON file: {"polygon": [[x,y], ...]}')
    ap.add_argument("--expected", type=int, help="ground-truth number of zone visits")
    a = ap.parse_args()

    with open(a.zone) as fh:
        poly = [tuple(p) for p in json.load(fh)["polygon"]]
    boxes = sorted(read_tracks(a.tracks), key=lambda b: (b.frame, b.track_id))

    report = {}
    for name, counter in {
        "naive_centroid": NaiveZoneCounter(poly, "centroid"),
        "debounced_footpoint": DebouncedZoneCounter(poly),
    }.items():
        s = summarize(run(counter, boxes))
        if a.expected is not None:
            s.update(count_error(s, a.expected))
        report[name] = s
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
