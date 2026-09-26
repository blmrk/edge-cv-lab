import json
from pathlib import Path

from replay.ablation import ablate
from replay.schema import read_tracks

FX = Path(__file__).resolve().parent.parent / "fixtures"
POLY = [tuple(p) for p in json.loads((FX / "zone.json").read_text())["polygon"]]


def load(name):
    return sorted(read_tracks(FX / name), key=lambda b: (b.frame, b.track_id))


def test_boundary_jitter_starts_at_the_naive_count_and_ends_at_one_visit():
    rows = ablate(load("boundary_jitter.jsonl"), POLY)
    assert (rows[0]["enters"], rows[-1]["enters"]) == (66, 1)  # README headline: naive centroid 66, debounced 1


def test_footpoint_step_removes_the_shadow_false_visit():
    rows = ablate(load("shadow_expansion.jsonl"), POLY)
    assert [r["enters"] for r in rows[:2]] == [1, 0]  # the centroid drifts into the zone, the footpoint never does


def test_truth_adds_visit_matching():
    # the car idles on the edge for 10 s and drives in around frame 325 (make_fixtures.py): one visit near 10.8 s
    rows = ablate(load("boundary_jitter.jsonl"), POLY, truth_ms=[10833])
    assert (rows[0]["matched"], rows[0]["false_visits"]) == (1, 65)
    assert (rows[-1]["matched"], rows[-1]["false_visits"], rows[-1]["f1"]) == (1, 0, 1.0)
