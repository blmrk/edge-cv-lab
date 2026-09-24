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


def test_score_does_not_let_errors_cancel():
    from replay.score import match
    # one miss and one double count: the raw count is "perfect", the matching is not
    r = match(pred_ms=[1000, 1200, 30000], true_ms=[1000, 15000, 30000], tol_ms=2000)
    assert r["count_error_pct"] == 0.0
    assert (r["matched"], r["false_visits"], r["missed_visits"]) == (2, 1, 1)
