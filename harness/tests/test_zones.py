# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
import json
from pathlib import Path

import pytest

from replay.metrics import summarize
from replay.schema import TrackBox, read_tracks
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


def test_explain_separates_a_parked_phantom_from_a_moving_vehicle():
    from replay.schema import TrackBox
    from replay.score import explain
    square = [(100, 100), (500, 100), (500, 500), (100, 500)]
    car = [TrackBox(f, f * 100, 1, (10 * f - 20, 260, 10 * f + 20, 300)) for f in range(60)]  # drives through
    dash = [TrackBox(f, f * 100, 2, (280, 280, 320, 300), 0.2) for f in range(200, 241)]  # never moves, inside
    boxes = sorted(car + dash, key=lambda b: (b.frame, b.track_id))

    rows, summary = explain(boxes, square, truth_ms=[1100], tol_ms=2000)

    assert [(r["track_id"], r["travel_px"]) for r in rows] == [(1, 590.0), (2, 0.0)]
    assert (rows[1]["w"], rows[1]["h"], rows[1]["score"]) == (40.0, 20.0, 0.2)
    assert (summary["static"], summary["moving"]) == (1, 1)
    assert summary["moving_vs_labels"]["matched"] == 1


def test_explain_scores_moving_enters_alone_so_a_phantom_cannot_take_a_label():
    # the phantom enters first, within 2 s of the car's label: time-order matching would hand it the label
    from replay.schema import TrackBox
    from replay.score import explain, match
    square = [(100, 100), (500, 100), (500, 500), (100, 500)]
    dash = [TrackBox(f, f * 100, 2, (280, 280, 320, 300), 0.2) for f in range(0, 40)]
    car = [TrackBox(f, f * 100, 1, (10 * (f - 5) - 20, 260, 10 * (f - 5) + 20, 300)) for f in range(5, 65)]
    boxes = sorted(car + dash, key=lambda b: (b.frame, b.track_id))
    rows, summary = explain(boxes, square, truth_ms=[1600], tol_ms=2000)

    assert match([r["ts_ms"] for r in rows], [1600], 2000)["matched"] == 1  # one of the two enters takes the label
    assert summary["moving_vs_labels"] == match([r["ts_ms"] for r in rows if r["track_id"] == 1], [1600], 2000)
    assert (summary["moving_vs_labels"]["matched"], summary["moving_vs_labels"]["false_visits"]) == (1, 0)


def test_min_travel_keeps_a_box_that_never_moved_from_entering():
    # a lane marking scored as a car: a small box jittering a pixel or two on the spot inside the zone for 5 s
    dash = [TrackBox(f, f * 1000 // 30, 1, (580 + f % 3, 380, 610 + f % 3, 400 + f % 2), 0.2, "car") for f in range(150)]
    assert summarize(run(DebouncedZoneCounter(POLY), dash))["enters"] == 1  # the default still counts it
    assert summarize(run(DebouncedZoneCounter(POLY, min_travel_px=20), dash))["enters"] == 0


def test_min_travel_still_counts_a_car_that_drives_in_and_stops():
    xs = list(range(300, 600, 10)) + [600] * 150  # drives in from outside, then stops in the zone until the end
    car = [TrackBox(f, f * 1000 // 30, 1, (x - 45, 340, x + 45, 400), 0.9, "car") for f, x in enumerate(xs)]
    got = run(DebouncedZoneCounter(POLY, min_travel_px=20), car)
    assert got == run(DebouncedZoneCounter(POLY), car)  # same events, same timestamps: it moved long before entering
    assert summarize(got)["enters"] == 1


def test_min_travel_never_counts_a_car_parked_in_the_zone_for_the_whole_clip():
    # the trade-off: its arrival was never seen, and a parked car looks exactly like a phantom
    parked = [TrackBox(f, f * 1000 // 30, 1, (555, 340, 645, 400), 0.9, "car") for f in range(150)]
    assert summarize(run(DebouncedZoneCounter(POLY), parked))["enters"] == 1
    assert summarize(run(DebouncedZoneCounter(POLY, min_travel_px=20), parked))["enters"] == 0


def test_min_travel_counts_a_parked_car_that_drives_off_from_when_it_moved():
    xs = [600] * 60 + list(range(605, 900, 5))  # parked for 2 s when its track starts, then leaves at 5 px a frame
    car = [TrackBox(f, f * 1000 // 30, 1, (x - 45, 340, x + 45, 400), 0.9, "car") for f, x in enumerate(xs)]
    enters = [e for e in run(DebouncedZoneCounter(POLY, min_travel_px=20), car) if e.kind == "enter"]
    assert [e.frame for e in enters] == [xs.index(620)]  # stamped once it had moved 20 px, not when the track started


def test_explain_passes_counter_settings_through():
    from replay.score import explain
    square = [(100, 100), (500, 100), (500, 500), (100, 500)]
    car = [TrackBox(f, f * 100, 1, (10 * f - 20, 260, 10 * f + 20, 300)) for f in range(60)]
    dash = [TrackBox(f, f * 100, 2, (280, 280, 320, 300), 0.2) for f in range(200, 241)]
    boxes = sorted(car + dash, key=lambda b: (b.frame, b.track_id))
    rows, _ = explain(boxes, square, truth_ms=[1100], min_travel_px=30)
    assert [r["track_id"] for r in rows] == [1]  # the dash never moved, so it never entered


def _returned_by(counter, boxes):
    """(frame of the box whose update() returned it, kind) per event; flush() at the last frame."""
    out = [(b.frame, e.kind) for b in boxes for e in counter.update(b)]
    return out + [(boxes[-1].frame, e.kind) for e in counter.flush()]


def _drive_through(stop_frames):
    xs = list(range(300, 600, 10)) + [600] * stop_frames + list(range(600, 900, 10))  # in, stop, out, 30 fps
    return [TrackBox(f, f * 1000 // 30, 1, (x - 45, 340, x + 45, 400), 0.9, "car") for f, x in enumerate(xs)]


def test_debounced_holds_each_enter_until_its_visit_closes_by_default():
    # a live consumer of the events learns about the visit only when it is over
    assert _returned_by(DebouncedZoneCounter(POLY), _drive_through(90)) == [(148, "enter"), (148, "exit")]


def test_enter_after_dwell_releases_the_enter_once_the_visit_has_lasted_min_dwell():
    car = _drive_through(90)
    got = _returned_by(DebouncedZoneCounter(POLY, enter_after_dwell=True), car)
    entered = run(DebouncedZoneCounter(POLY), car)[0]
    assert got == [(entered.frame + 30, "enter"), (148, "exit")]  # 30 frames = 1 s at 30 fps
    assert sorted(run(DebouncedZoneCounter(POLY, enter_after_dwell=True), car), key=lambda e: e.ts_ms) == \
        sorted(run(DebouncedZoneCounter(POLY), car), key=lambda e: e.ts_ms)  # same events, same stamps


def test_enter_after_dwell_still_drops_a_visit_shorter_than_min_dwell():
    # crosses the zone at 30 px a frame: inside for about half a second, then fully out
    quick = [TrackBox(f, f * 1000 // 30, 1, (x - 45, 340, x + 45, 400)) for f, x in enumerate(range(380, 1200, 30))]
    assert run(DebouncedZoneCounter(POLY), quick) == run(DebouncedZoneCounter(POLY, enter_after_dwell=True), quick) == []


def test_open_visits_seen_within_drops_a_visit_whose_track_went_quiet():
    # the car drives in, then its track stops; a second car keeps the stream going
    lost = [TrackBox(f, f * 100, 1, (x - 45, 340, x + 45, 400)) for f, x in enumerate(range(300, 610, 10))]
    other = [TrackBox(f, f * 100, 2, (-200, 340, -110, 400)) for f in range(31, 50)]  # outside the zone
    c = DebouncedZoneCounter(POLY)
    for b in lost + other[:5]:  # up to 3.5 s: track 1 last seen at 3.0 s
        c.update(b)
    assert [tid for tid, _ in c.open_visits(3500)] == [1]  # still open: lost_ms has not run out
    assert [tid for tid, _ in c.open_visits(3500, seen_within_ms=500)] == [1]  # seen 0.5 s ago: still counted
    assert c.open_visits(3600, seen_within_ms=500) == []  # 0.6 s since it was seen: no longer counted


def test_enter_after_dwell_with_no_min_dwell_returns_the_enter_from_the_commit_box():
    car = _drive_through(90)
    entered = run(DebouncedZoneCounter(POLY, min_dwell_ms=0), car)[0]
    got = _returned_by(DebouncedZoneCounter(POLY, min_dwell_ms=0, enter_after_dwell=True), car)
    assert got[0] == (entered.frame, "enter")


def test_enter_after_dwell_returns_enter_then_exit_when_dwell_is_proven_on_the_exit_box():
    car = _drive_through(90)
    enter, exit_ = run(DebouncedZoneCounter(POLY), car)
    dwell = exit_.ts_ms - enter.ts_ms  # dwell is proven on the very box that commits the exit
    got = _returned_by(DebouncedZoneCounter(POLY, min_dwell_ms=dwell, enter_after_dwell=True), car)
    assert got == [(exit_.frame, "enter"), (exit_.frame, "exit")]
