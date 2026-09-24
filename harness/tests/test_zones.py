import json
from pathlib import Path

import pytest

from replay.metrics import summarize
from replay.schema import read_tracks
from replay.zones import DebouncedZoneCounter, NaiveZoneCounter, run

FX = Path(__file__).resolve().parent.parent / "fixtures"
POLY = [tuple(p) for p in json.loads((FX / "zone.json").read_text())["polygon"]]


def load(name):
    return list(read_tracks(FX / name))


def test_boundary_jitter_naive_overcounts():
    s = summarize(run(NaiveZoneCounter(POLY, "centroid"), load("boundary_jitter.jsonl")))
    assert s["enters"] > 10  # documents the bug: one car, dozens of "visits"


def test_boundary_jitter_debounced_counts_once():
    s = summarize(run(DebouncedZoneCounter(POLY), load("boundary_jitter.jsonl")))
    assert (s["enters"], s["exits"], s["net_balance"]) == (1, 1, 0)


def test_shadow_expansion_centroid_false_positive():
    s = summarize(run(NaiveZoneCounter(POLY, "centroid"), load("shadow_expansion.jsonl")))
    assert s["enters"] >= 1


def test_shadow_expansion_footpoint_ignores_adjacent_lane():
    s = summarize(run(DebouncedZoneCounter(POLY), load("shadow_expansion.jsonl")))
    assert s["enters"] == 0


def test_bad_record_fails_loud(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"frame": 1, "track_id": 1}\n')
    with pytest.raises(ValueError, match="bad track record"):
        list(read_tracks(p))


def test_mixed_traffic_matches_ground_truth():
    truth = json.loads((FX / "traffic.truth.json").read_text())["expected_visits"]
    boxes = load("traffic.jsonl")
    assert summarize(run(DebouncedZoneCounter(POLY), boxes))["enters"] == truth
    assert summarize(run(NaiveZoneCounter(POLY, "centroid"), boxes))["enters"] > 5 * truth


def test_parked_on_edge_then_real_visit_counts_once():
    from replay.schema import TrackBox
    # footpoint sits 4 px either side of the x=400 edge long enough to commit enter, dwell 1 s, and exit
    xs = list(range(300, 396, 4)) + [404] * 40 + [396] * 20
    xs += list(range(396, 600, 6)) + [600] * 60 + list(range(600, 900, 6))  # then drive in, stop 2 s, leave
    boxes = [TrackBox(f, f * 1000 // 30, 1, (x - 45, 340, x + 45, 400), 0.9, "car") for f, x in enumerate(xs)]
    s = summarize(run(DebouncedZoneCounter(POLY), boxes))
    assert (s["enters"], s["exits"]) == (1, 1)


def test_debounced_matches_ground_truth_on_seeded_scenes():
    from replay.synth import ZONE, generate_traffic
    wrong = {}
    for seed in range(11, 41):  # the sim replays seed 11, 12, 13, ...
        boxes, truth = generate_traffic(seed=seed)
        got = summarize(run(DebouncedZoneCounter(ZONE), boxes))["enters"]
        if got != truth:
            wrong[seed] = (got, truth)
    assert wrong == {}


def test_score_does_not_let_errors_cancel():
    from replay.score import match
    # one miss and one double count: the raw count is "perfect", the matching is not
    r = match(pred_ms=[1000, 1200, 30000], true_ms=[1000, 15000, 30000], tol_ms=2000)
    assert r["count_error_pct"] == 0.0
    assert (r["matched"], r["false_visits"], r["missed_visits"]) == (2, 1, 1)
