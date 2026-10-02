import json
import sys
import types
from dataclasses import asdict

import pytest

from replay import phantoms, secondbox, trackers
from replay.detections import Detection
from replay.schema import TrackBox
from replay.trackers import create
from replay.zones import DebouncedZoneCounter, run

BIG = (0, 0, 100, 100)  # 10000 px2


def _d(box, score, f=0, cls="car"):
    return Detection(f, f * 100, box, score, cls)


def _g(vid, box, f=0):
    return TrackBox(f, f * 100, vid, box, 1.0, "car")


def _labels(dets, gt):
    return [(label, vid) for label, vid, _ in secondbox.classify(dets, gt)]


# contained: a detection 0.8 (or 0.9) of its own area inside a strictly higher-scoring one in its frame is dropped

def test_a_lower_scoring_box_inside_a_higher_scoring_one_is_dropped_and_the_container_kept():
    big, small = _d(BIG, 0.9), _d((60, 0, 110, 40), 0.5)  # 1600 of its 2000 px2 inside: 0.8
    assert secondbox.contained([big, small], 0.8) == [big]


def test_a_large_low_scoring_box_around_a_small_high_scoring_one_is_kept():
    # inside BIG at IoU 0.16; the share is the small box's own area, never IoU or the container's area
    big, small = _d(BIG, 0.3), _d((10, 10, 50, 50), 0.9)
    assert secondbox.contained([big, small], 0.8) == [big, small]
    big, small = _d(BIG, 0.9), _d((10, 10, 50, 50), 0.3)  # the same pair, scores swapped: the small one goes
    assert secondbox.contained([big, small], 0.8) == [big]


@pytest.mark.parametrize("box, share, dropped", [
    ((20, 0, 120, 10), 0.8, True),   # 800 of 1000 px2 inside: exactly 0.80, and the rule is 0.8 or more
    ((21, 0, 121, 10), 0.8, False),  # 0.79
    ((15, 0, 115, 10), 0.9, False),  # 0.85, under 0.9
])
def test_the_share_cut_is_inclusive_on_integer_boxes(box, share, dropped):
    big, d = _d(BIG, 0.9), _d(box, 0.5)
    assert secondbox.contained([big, d], share) == ([big] if dropped else [big, d])


def test_equal_scores_drop_neither_box_even_wholly_inside():
    big, small = _d(BIG, 0.5), _d((10, 10, 50, 50), 0.5)
    assert secondbox.contained([big, small], 0.8) == [big, small]


def test_a_dropped_box_still_contains_so_a_chain_drops_every_link():
    a = _d(BIG, 0.9)
    b = _d((20, 0, 120, 100), 0.6)  # 0.8 inside a
    c = _d((100, 20, 120, 80), 0.3)  # wholly inside b, nothing inside a
    assert secondbox.contained([a, b, c], 0.8) == [a]


@pytest.mark.parametrize("same_class, kept", [(True, ["bus", "car"]), (False, ["bus"])])
def test_same_class_keeps_a_car_inside_a_bus_but_not_inside_another_car(same_class, kept):
    bus = _d((0, 0, 200, 100), 0.9, cls="bus")
    car = _d((10, 10, 60, 60), 0.5)
    inner = _d((20, 20, 40, 40), 0.3)  # inside the car and the bus
    out = secondbox.contained([bus, car, inner], 0.8, same_class=same_class)
    assert [d.cls for d in out] == kept and inner not in out


def test_only_its_own_frame_can_contain_a_box_and_the_input_order_is_kept():
    dets = [_d((200, 0, 260, 40), 0.3, f=1),
            _d((10, 10, 20, 20), 0.5, f=0),  # inside the frame-1 container below
            _d(BIG, 0.9, f=1),
            _d((30, 30, 40, 40), 0.2, f=1)]  # inside it in its own frame
    assert secondbox.contained(dets, 0.8) == dets[:3]
    assert secondbox.contained(dets[:3], 0.8) == dets[:3]
    assert secondbox.contained([], 0.8) == []


# classify: each detection against the annotated boxes of its frame

def test_the_vehicle_s_detection_is_the_higher_iou_one_not_the_higher_score():
    near, loose = _d((0, 0, 100, 90), 0.4), _d((0, 0, 100, 60), 0.9)  # IoU 0.9 and 0.6 with the annotated box
    out = secondbox.classify([near, loose], [_g(7, BIG)])
    assert [(label, vid, round(i, 2)) for label, vid, i in out] == [("vehicle", 7, 0.9), ("duplicate", 7, 0.6)]


@pytest.mark.parametrize("scores, labels", [
    ((0.4, 0.6), ["duplicate", "vehicle"]),  # equal IoU: the higher score
    ((0.5, 0.5), ["vehicle", "duplicate"]),  # equal IoU and score: the earlier line
])
def test_equal_iou_pairs_one_detection_by_score_then_file_order(scores, labels):
    top, bottom = _d((0, 0, 100, 80), scores[0]), _d((0, 20, 100, 100), scores[1])  # both IoU 0.8
    assert _labels([top, bottom], [_g(7, BIG)]) == [(label, 7) for label in labels]


def test_iou_exactly_0_5_is_paired():
    assert _labels([_d((0, 0, 10, 10), 0.5)], [_g(1, (0, 0, 20, 10))]) == [("vehicle", 1)]


def test_pairing_is_one_to_one_greedy_so_a_detection_pairs_with_its_second_best_box():
    # all 10 px tall: d1 overlaps A at IoU 0.7 and B at 0.6, d2 overlaps A at 0.9
    a, b = _g(1, (0, 0, 100, 10)), _g(2, (40, 0, 130, 10))
    d1, d2 = _d((30, 0, 100, 10), 0.5), _d((0, 0, 90, 10), 0.5)
    out = secondbox.classify([d1, d2], [a, b])
    assert [(label, vid, round(i, 2)) for label, vid, i in out] == [("vehicle", 2, 0.6), ("vehicle", 1, 0.9)]


def test_a_detection_pairs_once_so_its_second_box_stays_free_for_another_detection():
    # all 10 px tall: d1 overlaps A at IoU 0.7 and B at 0.6, d2 overlaps B at 0.56 and A at 0.2
    a, b = _g(1, (0, 0, 100, 10)), _g(2, (40, 0, 130, 10))
    d1, d2 = _d((30, 0, 100, 10), 0.5), _d((75, 0, 125, 10), 0.5)
    out = secondbox.classify([d1, d2], [a, b])
    assert [(label, vid, round(i, 2)) for label, vid, i in out] == [("vehicle", 1, 0.7), ("vehicle", 2, 0.56)]


def test_an_unpaired_detection_at_iou_exactly_0_5_is_a_duplicate():
    assert _labels([_d((0, 0, 20, 10), 0.9), _d((0, 0, 10, 10), 0.5)], [_g(1, (0, 0, 20, 10))]) == [
        ("vehicle", 1), ("duplicate", 1)]


def test_a_duplicate_is_of_its_best_box_not_the_first_it_matches():
    a, b = _g(1, (0, 0, 100, 10)), _g(2, (20, 0, 120, 10))
    dets = [_d(a.bbox, 0.9), _d(b.bbox, 0.9), _d((15, 0, 115, 10), 0.3)]  # the third: IoU 0.9 with b, 0.74 with a
    assert _labels(dets, [a, b]) == [("vehicle", 1), ("vehicle", 2), ("duplicate", 2)]


def test_only_turns_on_a_paired_detection_not_on_any_detection_at_iou_0_5():
    a, b = _g(1, (0, 0, 100, 10)), _g(2, (20, 0, 120, 10))  # b's detection is at IoU 0.67 with a, but paired with b
    assert _labels([_d(b.bbox, 0.9), _d((0, 0, 10, 10), 0.5)], [a, b]) == [("vehicle", 2), ("only", 1)]


PAIRED = (0, 0, 100, 95)  # IoU 0.95 with BIG


@pytest.mark.parametrize("why, boxes, labels", [
    ("0.8 of it inside, and the vehicle has its detection", [PAIRED, (20, 0, 120, 10)], ["vehicle", "part"]),
    ("0.8 of it inside, and no detection is paired with the vehicle", [(20, 0, 120, 10)], ["only"]),
    ("0.79 of it inside", [PAIRED, (21, 0, 121, 10)], ["vehicle", "other"]),
    ("wholly inside, at IoU 0.1", [PAIRED, (0, 0, 10, 100)], ["vehicle", "part"]),
])
def test_a_box_mostly_inside_an_annotated_box_is_part_of_it_on_its_own_area(why, boxes, labels):
    out = _labels([_d(b, 0.5) for b in boxes], [_g(5, BIG)])
    assert [label for label, _ in out] == labels, why
    assert [vid for label, vid in out if label != "other"] == [5] * (len(labels) - labels.count("other")), why


def test_a_part_box_listed_before_the_paired_detection_is_still_part():
    # the 0.9 score also rules out a single pass in score order
    assert _labels([_d((20, 0, 120, 10), 0.9), _d(PAIRED, 0.5)], [_g(5, BIG)]) == [("part", 5), ("vehicle", 5)]


def test_a_box_inside_a_car_and_the_bus_around_it_belongs_to_the_car():
    bus, car = _g(1, (0, 0, 400, 200)), _g(2, (100, 50, 200, 150))
    beside = _g(3, (60, 50, 120, 100))  # overlaps the box most (IoU 0.38), but holds only 0.6 of it
    d = _d((90, 50, 140, 100), 0.5)     # 0.8 inside the car (IoU 0.19), wholly inside the bus
    out = secondbox.classify([d], [bus, beside, car])
    assert [(label, vid, round(i, 2)) for label, vid, i in out] == [("only", 2, 0.19)]  # its IoU, not the 0.8 share


@pytest.mark.parametrize("boxes, labels", [
    ([((0, 0, 50, 50), 0.9), ((0, 0, 100, 40), 0.3)], ["part", "only"]),  # IoU 0.25 and 0.4
    ([((0, 0, 100, 40), 0.3), ((0, 0, 40, 100), 0.6)], ["part", "only"]),  # IoU 0.4 both: the higher score
    ([((0, 0, 100, 40), 0.5), ((0, 0, 40, 100), 0.5)], ["only", "part"]),  # and the score too: the earlier line
])
def test_with_no_paired_detection_one_box_on_the_vehicle_is_only_and_the_rest_part(boxes, labels):
    assert _labels([_d(b, s) for b, s in boxes], [_g(4, BIG)]) == [(label, 4) for label in labels]


def test_outside_the_annotated_range_is_not_judged_and_an_empty_frame_inside_it_is_other():
    gt = [_g(1, BIG, f=0), _g(1, BIG, f=4)]
    dets = [_d(BIG, 0.5, f=2), _d(BIG, 0.5, f=6), _d(BIG, 0.5, f=4)]
    assert secondbox.classify(dets, gt) == [("other", None, 0.0), ("not judged", None, 0.0), ("vehicle", 1, 1.0)]


def test_the_same_annotated_vehicle_in_another_frame_makes_no_duplicate():
    gt = [_g(3, BIG, f=0), _g(3, BIG, f=1)]
    assert _labels([_d(BIG, 0.5, f=0), _d(BIG, 0.5, f=1)], gt) == [("vehicle", 3), ("vehicle", 3)]


def test_losses_compare_vehicle_frames_not_detections():
    bus, car = (0, 0, 400, 200), (100, 50, 200, 150)  # the car lies inside the bus: nested
    gt = [_g(1, BIG, f=0), _g(3, bus, f=1), _g(2, car, f=1)] + [_g(1, BIG, f=f) for f in (2, 3, 4, 5, 6)]
    base = [_d((0, 0, 100, 90), 0.5, f=0),     # IoU 0.9, gone after
            _d((100, 50, 200, 140), 0.5, f=1),  # IoU 0.9 with the nested car, gone after
            _d((0, 0, 100, 90), 0.5, f=2),     # IoU 0.9, then another detection at 0.75
            _d((0, 0, 100, 90), 0.5, f=3),     # IoU 0.9, then another detection at 0.85
            _d((0, 0, 10, 100), 0.5, f=4),     # only: IoU 0.1, wholly inside, gone after
            _d((0, 0, 100, 75), 0.5, f=6)]     # IoU 0.75, then another detection at 0.9
    step = [_d((0, 0, 100, 75), 0.5, f=2), _d((0, 0, 100, 85), 0.5, f=3),
            _d((0, 0, 100, 90), 0.5, f=5),     # paired only after the step: a gain, never lost
            _d((0, 0, 100, 90), 0.5, f=6)]     # IoU rising 0.75 to 0.9: not fit lost

    base_cover = secondbox.cover(base, secondbox.classify(base, gt))
    step_cover = secondbox.cover(step, secondbox.classify(step, gt))
    out = secondbox.losses(base_cover, step_cover, secondbox.nested(gt))

    assert secondbox.nested(gt) == {(1, 2)}
    assert sorted(base_cover[0]) == [(0, 1), (1, 2), (2, 1), (3, 1), (6, 1)] and base_cover[1] == {(4, 1)}
    assert out == {"lost": {(0, 1), (1, 2)}, "lost_nested": {(1, 2)}, "only_lost": {(4, 1)}, "fit_lost": {(2, 1)}}


def test_a_nested_vehicle_frame_never_paired_at_the_baseline_is_not_lost_nested():
    bus, car = (0, 0, 400, 200), (100, 50, 200, 150)  # the car lies inside the bus: nested
    gt = [_g(3, bus, f=0), _g(2, car, f=0)]
    dets = [_d(bus, 0.5, f=0)]  # pairs with the bus; the car is never paired
    c = secondbox.cover(dets, secondbox.classify(dets, gt))
    assert secondbox.nested(gt) == {(0, 2)} and set(c[0]) == {(0, 3)}
    assert secondbox.losses(c, c, secondbox.nested(gt))["lost_nested"] == set()


def test_losses_count_only_falls_and_vehicle_frames_left_with_nothing():
    gt = [_g(1, BIG, f=f) for f in range(5)]
    base = [_d((0, 0, 100, 60), 0.5, f=0),                                 # paired 0.6, then 0.75: a rise
            _d((0, 0, 10, 100), 0.5, f=1),                                 # only, then paired
            _d((0, 0, 10, 100), 0.5, f=2),                                 # only, then only
            _d((0, 0, 100, 90), 0.5, f=3),                                 # paired 0.9, then 0.9 and a duplicate
            _d((0, 0, 100, 90), 0.5, f=4), _d((0, 0, 10, 100), 0.4, f=4)]  # paired and a part, then nothing
    step = [_d((0, 0, 100, 75), 0.5, f=0), _d((0, 0, 100, 90), 0.5, f=1), _d((0, 0, 10, 100), 0.5, f=2),
            _d((0, 0, 100, 90), 0.5, f=3), _d((0, 0, 100, 60), 0.4, f=3)]

    base_cover = secondbox.cover(base, secondbox.classify(base, gt))
    step_cover = secondbox.cover(step, secondbox.classify(step, gt))
    out = secondbox.losses(base_cover, step_cover, secondbox.nested(gt))

    assert base_cover == ({(0, 1): 0.6, (3, 1): 0.9, (4, 1): 0.9}, {(1, 1), (2, 1)})
    assert step_cover == ({(0, 1): 0.75, (1, 1): 0.9, (3, 1): 0.9}, {(2, 1)})
    assert out == {"lost": {(4, 1)}, "lost_nested": set(), "only_lost": set(), "fit_lost": set()}


def test_an_annotated_box_exactly_0_8_inside_another_is_nested_and_0_79_is_not():
    gt = [_g(1, (0, 0, 100, 100), f=0), _g(2, (20, 0, 120, 10), f=0),  # 2: 800 of its 1000 px2 inside 1, share 0.8
          _g(3, (0, 0, 100, 100), f=1), _g(4, (21, 0, 121, 10), f=1)]  # 4: 790 of 1000, share 0.79
    assert secondbox.nested(gt) == {(0, 2)}


# by_vehicle: each debounced enter put down to the annotated vehicle its box is on

ZONE = [(0, 200), (3000, 200), (3000, 800), (0, 800)]
F = range(80)


def _pass(tid, x, frames, f0=0, w=80, h=60):
    """A box driving down 10 px a frame, footpoint y 150 + 10 (f - f0): on frames f0..79 its enter commits at f0 + 10
    (10 px inside from f0 + 6, five frames). h=48 keeps the footpoint at IoU 0.8 with the 60 px box."""
    ys = {f: 150 + 10 * (f - f0) for f in frames}
    return [TrackBox(f, f * 100, tid, (x, ys[f] - h, x + w, ys[f]), 1.0, "car") for f in frames]


def _kinds(rows):
    out = {}
    for r in rows:
        out.setdefault(r["kind"], []).append(r["track_id"])
    return {k: sorted(v) for k, v in out.items()}


def test_by_vehicle_gives_each_enter_the_first_kind_that_applies():
    gt = (_pass(101, 100, range(71)) + _pass(102, 400, range(71)) + _pass(103, 700, range(71), w=200, h=150)
          + _pass(104, 1000, range(9, 13))  # four frames inside: no visit of its own
          + _pass(105, 1900, range(71)))   # no track at all: missed. Annotated range 0-70
    tracks = (_pass(1, 100, F) + _pass(2, 100, range(8, 80), h=48)        # on 101 at 10, and at 12 beside track 1
              + _pass(3, 400, range(21)) + _pass(4, 400, range(18, 80), h=48)  # on 102 at 10, and at 22 after 3 ended
              + _pass(5, 720, F) + _pass(6, 700, range(12, 80), w=200, h=150)  # inside bus 103 at 10, the bus at 16
              + _pass(7, 1000, F) + _pass(10, 1020, F, w=40, h=30)           # on 104, and inside it
              + _pass(8, 1300, F)                                            # on nothing annotated
              + _pass(9, 1600, range(75, 80), f0=65))                        # enters at 79, after the annotations

    rows, missed = secondbox.by_vehicle(tracks, gt, ZONE)

    assert _kinds(rows) == {"found": [1, 3, 6], "duplicate": [2], "again": [4], "part": [5],
                            "no truth visit": [7, 10], "no annotated box": [8], "not judged": [9]}
    assert missed == [105]


def test_enters_are_judged_in_commit_order_not_the_order_the_counter_returns_them():
    gt, a, b = _pass(7, 100, F), _pass(1, 100, F), _pass(2, 100, range(12, 31), h=48)  # b enters at 16, leaves at 30
    in_time = sorted(a + b, key=lambda t: (t.frame, t.track_id))
    assert [e.track_id for e in run(DebouncedZoneCounter(ZONE), in_time) if e.kind == "enter"] == [2, 1]

    rows, missed = secondbox.by_vehicle(a + b, gt, ZONE)

    assert _kinds(rows) == {"found": [1], "duplicate": [2]} and missed == []


@pytest.mark.parametrize("close, loose", [(2, 1), (1, 2)])
def test_of_two_enters_in_one_frame_the_higher_iou_is_found_whatever_the_track_id(close, loose):
    tracks = _pass(close, 100, F, h=54) + _pass(loose, 100, F, h=42)  # IoU 0.9 and 0.7, one footpoint
    rows, _ = secondbox.by_vehicle(tracks, _pass(7, 100, F), ZONE)
    assert [r["frame"] for r in rows] == [10, 10]
    assert _kinds(rows) == {"found": [close], "duplicate": [loose]}


@pytest.mark.parametrize("unannotated, kind", [
    (range(6, 10), "found"),            # annotated in the commit frame alone, of the five-frame streak
    (range(10, 11), "no annotated box"),  # annotated in the four streak frames before it alone
])
def test_an_enter_is_judged_on_its_commit_frame_not_its_streak(unannotated, kind):
    gt = _pass(7, 100, [f for f in F if f not in unannotated])  # still a visit of its own, later
    rows, _ = secondbox.by_vehicle(_pass(1, 100, F), gt, ZONE)
    assert [(r["frame"], r["kind"]) for r in rows] == [(10, kind)]


@pytest.mark.parametrize("last", [20, 21])  # 1's last box 2 frames, then 1 frame, before 2's enter at 22
def test_a_new_id_on_a_vehicle_whose_other_track_ended_one_or_two_frames_before_is_again(last):
    tracks = _pass(1, 100, range(last + 1)) + _pass(2, 100, range(18, 80), h=48)  # 2 enters at 22
    rows, _ = secondbox.by_vehicle(tracks, _pass(7, 100, F), ZONE)
    assert [(r["track_id"], r["frame"], r["kind"]) for r in rows] == [(1, 10, "found"), (2, 22, "again")]


@pytest.mark.parametrize("other", [_pass(3, 120, F),                 # its best box is vehicle 8, at IoU 0.6 with 7
                                   _pass(3, 110, F, w=40, h=30)])    # a part of 7, matching no box
def test_another_track_whose_best_box_is_not_the_vehicle_leaves_a_new_id_again(other):
    gt = _pass(7, 100, F) + _pass(8, 120, F)
    tracks = _pass(1, 100, range(21)) + _pass(2, 100, range(18, 80), h=48) + other
    rows, _ = secondbox.by_vehicle(tracks, gt, ZONE)
    assert [(r["track_id"], r["kind"]) for r in rows if r["track_id"] == 2] == [(2, "again")]


def test_the_track_that_found_a_vehicle_entering_it_again_under_its_own_id_is_again():
    ys = [150 + 10 * i for i in range(30)] + [440 - 10 * i for i in range(40)] + [50 + 10 * i for i in range(60)]
    path = [(f, (100, y - 60, 180, y)) for f, y in enumerate(ys)]  # in, back out over the top edge, in again
    track = [TrackBox(f, f * 100, 1, box, 1.0, "car") for f, box in path]
    rows, missed = secondbox.by_vehicle(track, [TrackBox(f, f * 100, 7, box, 1.0, "car") for f, box in path], ZONE)
    assert [(r["track_id"], r["frame"], r["kind"], r["vehicle"]) for r in rows] == [
        (1, 10, "found", 7), (1, 90, "again", 7)]  # an extra, by the kind the same vehicle again takes
    assert missed == []


def test_a_part_enter_is_never_found_and_leaves_the_vehicle_to_a_later_matching_enter():
    bus = _pass(7, 700, F, w=200, h=150)
    tracks = _pass(1, 720, F) + _pass(2, 700, range(12, 80), w=200, h=150)  # inside the bus at 10 (IoU 0.16); it at 16
    rows, missed = secondbox.by_vehicle(tracks, bus, ZONE)
    assert [(r["track_id"], r["kind"], r["vehicle"], round(r["iou"], 2)) for r in rows] == [
        (1, "part", 7, 0.16), (2, "found", 7, 1.0)]  # a part row carries its IoU with the box that holds it
    assert missed == []


@pytest.mark.parametrize("finder, part_frames, enter", [
    (F, range(12, 80), 16),          # the finder still on the bus: duplicate if part were skipped
    (range(21), range(18, 80), 22),  # the finder gone: again if part were skipped
])
def test_a_part_enter_after_its_vehicle_is_found_is_still_part(finder, part_frames, enter):
    bus = _pass(7, 700, F, w=200, h=150)
    tracks = _pass(2, 700, finder, w=200, h=150) + _pass(1, 720, part_frames)
    rows, missed = secondbox.by_vehicle(tracks, bus, ZONE)
    assert [(r["track_id"], r["frame"], r["kind"]) for r in rows] == [(2, 10, "found"), (1, enter, "part")]
    assert missed == []


def test_a_part_enter_inside_a_car_nested_in_a_bus_is_on_the_car():
    bus = _pass(7, 600, F, w=400, h=200)  # holds all of the track box: IoU 0.06
    car = _pass(8, 730, F, w=200, h=150)  # inside the bus; holds 70 of the box's 80 px width (0.875): IoU 0.14
    rows, missed = secondbox.by_vehicle(_pass(1, 720, F), bus + car, ZONE)
    assert [(r["kind"], r["vehicle"], round(r["iou"], 2)) for r in rows] == [("part", 8, 0.14)]
    assert missed == [7, 8]


def test_a_vehicle_with_only_a_part_enter_is_missed():
    rows, missed = secondbox.by_vehicle(_pass(1, 720, F), _pass(7, 700, F, w=200, h=150), ZONE)
    assert [r["kind"] for r in rows] == ["part"] and missed == [7]


def test_an_enter_at_iou_exactly_0_5_matches():
    rows, missed = secondbox.by_vehicle(_pass(1, 100, F), _pass(7, 100, F, h=30), ZONE)
    assert [(r["kind"], r["vehicle"], r["iou"]) for r in rows] == [("found", 7, 0.5)] and missed == []


@pytest.mark.parametrize("gt_frames, tracks, frame", [
    (range(0, 31), _pass(1, 100, range(26, 80)), 30),  # the enter commits on the last annotated frame
    (range(10, 80), _pass(1, 100, F), 10),             # on the first
])
def test_an_enter_on_either_end_of_the_annotated_range_is_judged(gt_frames, tracks, frame):
    rows, missed = secondbox.by_vehicle(tracks, _pass(7, 100, gt_frames), ZONE)
    assert [(r["frame"], r["kind"]) for r in rows] == [(frame, "found")] and missed == []


def test_by_vehicle_sorts_its_inputs():
    gt = _pass(7, 100, F) + _pass(8, 400, F)
    tracks = _pass(1, 100, F) + _pass(2, 400, range(21)) + _pass(3, 400, range(18, 80), h=48)
    want = secondbox.by_vehicle(tracks, gt, ZONE)
    assert [r["kind"] for r in want[0]] == ["found", "found", "again"]
    assert secondbox.by_vehicle(tracks[::-1], gt, ZONE) == want and secondbox.by_vehicle(tracks, gt[::-1], ZONE) == want


def test_the_zone_rule_changes_the_enters_never_the_truth_vehicles():
    standing = [TrackBox(f, f * 100, 7, (100, 440, 180, 500), 1.0, "car") for f in range(41)]  # in the zone from 0
    tracks = [TrackBox(b.frame, b.ts_ms, 1, b.bbox, 0.9, "car") for b in standing]
    assert secondbox.by_vehicle(tracks, standing, ZONE)[1] == []
    assert secondbox.by_vehicle(tracks, standing, ZONE, min_travel_px=30) == ([], [7])


def test_an_enter_belongs_to_its_best_box_even_when_a_weaker_match_has_a_visit():
    track = _pass(1, 100, F, w=100)
    gt = _pass(11, 145, F, w=55) + _pass(12, 100, range(9, 13), w=60)  # IoU 0.55 with a visit; 0.6, four frames only
    rows, missed = secondbox.by_vehicle(track, gt, ZONE)
    assert [(r["kind"], r["vehicle"], round(r["iou"], 2)) for r in rows] == [("no truth visit", 12, 0.6)]
    assert missed == [11]


def _row(tid, ts_ms, kind, vehicle=7):
    return {"track_id": tid, "frame": ts_ms // 100, "ts_ms": ts_ms, "kind": kind, "vehicle": vehicle, "iou": 0.6}


@pytest.mark.parametrize("why, step_extra, changed", [
    ("same kind and vehicle, 2000 ms apart, another track id", _row(9, 12000, "duplicate"), False),
    ("2001 ms apart", _row(9, 12001, "duplicate"), True),
    ("same time and vehicle, another kind", _row(9, 10000, "part"), True),
    ("same kind and time, another vehicle", _row(9, 10000, "duplicate", vehicle=8), True),
    ("again is compared like the others", _row(9, 10000, "again"), True),
])
def test_extras_diff_matches_extras_by_kind_vehicle_and_time_never_track_id(why, step_extra, changed):
    base = [_row(1, 1000, "found"), _row(5, 10000, "duplicate")]
    removed, new = secondbox.extras_diff(base, [_row(3, 1000, "found"), step_extra])
    assert (removed, new) == (([base[1]], [step_extra]) if changed else ([], [])), why


@pytest.mark.parametrize("why, step_row", [
    ("a found enter is no second box", _row(9, 10000, "found")),
    ("nor is an enter with no annotated box", _row(9, 10000, "no annotated box", vehicle=None)),
])
def test_extras_diff_compares_only_duplicate_part_and_again(why, step_row):
    base = [_row(1, 1000, "found"), _row(5, 10000, "duplicate")]
    assert secondbox.extras_diff(base, [_row(3, 1000, "found"), step_row]) == ([base[1]], []), why


# wiring

def test_steps_are_the_declared_grid_and_share_the_phantom_rows_without_changing_them():
    assert [s[0] for s in secondbox.STEPS] == ["baseline", "contain080", "contain090", "contain080_same",
                                               "contain090_same", "birth040", "birth050"]
    assert secondbox.STEPS[0] is phantoms.STEPS[0]
    assert secondbox.STEPS[5] is phantoms.STEPS[3] and secondbox.STEPS[6] is phantoms.STEPS[4]
    assert [s[0] for s in phantoms.STEPS] == ["baseline", "travel30", "floor030", "birth040", "birth050", "area1024",
                                              "mask025", "mask010"]
    inside = {0.85: (15, 0, 115, 10), 0.95: (5, 0, 105, 10)}  # a car this share inside a higher-scoring box
    dropped = {key: {(share, cls) for share, box in inside.items() for cls in ("bus", "car")
                     if keep([_d(BIG, 0.9, cls=cls), _d(box, 0.5)]) == [_d(BIG, 0.9, cls=cls)]}
               for key, _, params, keep, counter in secondbox.STEPS[1:5] if params == counter == {}}
    assert dropped == {"contain080": {(0.85, "bus"), (0.85, "car"), (0.95, "bus"), (0.95, "car")},
                       "contain090": {(0.95, "bus"), (0.95, "car")},
                       "contain080_same": {(0.85, "car"), (0.95, "car")},
                       "contain090_same": {(0.95, "car")}}


def _write(tmp_path, name, boxes=None, obj=None):
    path = tmp_path / name
    path.write_text(json.dumps(obj) if boxes is None else "".join(json.dumps(asdict(b)) + "\n" for b in boxes))
    return str(path)


def _table(out, title):
    """{key: cells} and the keys in order, for the markdown table printed under the line starting with `title`."""
    lines = out.splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith(title)) + 3  # title, header, separator
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return {r[0]: r for r in rows}, [r[0] for r in rows]


def test_main_dets_tracks_every_detection_scores_the_masked_ones_and_runs_each_redetection_as_the_baseline(
        tmp_path, monkeypatch, capsys):
    calls, seen = [], []

    def fake_create(name, **params):  # bytetrack needs ultralytics: groundplane stands in, and records its input
        calls.append((name, params))
        tracker = create("groundplane")
        return types.SimpleNamespace(update=lambda f, ts, ds: seen.extend(ds) or tracker.update(f, ts, ds))

    monkeypatch.setattr(phantoms, "create", fake_create)
    car = [Detection(b.frame, b.ts_ms, b.bbox, 0.9, "car") for b in _pass(0, 100, F)]
    ignored = Detection(5, 500, (930, 30, 970, 70), 0.9, "car")  # centred in the ignored region
    unannotated = [Detection(b.frame, b.ts_ms, b.bbox, 0.9, "car") for b in _pass(0, 1300, F)]  # only in redets
    redets = sorted([d for d in car if d.frame != 20] + unannotated, key=lambda d: d.frame)
    files = {"--dets": _write(tmp_path, "dets.jsonl", sorted(car + [ignored], key=lambda d: d.frame)),
             "--redetected": "nms050=" + _write(tmp_path, "redets.jsonl", redets),
             # vehicle 9 under the ignored box, one frame, outside the zone: only an unmasked set would pair it
             "--gt": _write(tmp_path, "gt.jsonl", _pass(7, 100, F) + [TrackBox(5, 500, 9, ignored.bbox, 1.0, "car")]),
             "--zone": _write(tmp_path, "zone.json", obj={"polygon": ZONE}),
             "--truth": _write(tmp_path, "truth.json", obj={"enters_ms": [1000]}),
             "--ignored": _write(tmp_path, "ignored.json", obj={"regions_xyxy": [[900, 0, 1000, 100]]}),
             "--out-dir": str(tmp_path / "out")}
    monkeypatch.setattr(sys, "argv", ["secondbox.py", *(x for kv in files.items() for x in kv)])

    secondbox.main()

    out = capsys.readouterr().out
    assert ignored in seen
    visits, keys = _table(out, "visits")
    assert keys == [s[0] for s in secondbox.STEPS] + ["nms050"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(k + ".jsonl" for k in keys)
    assert [visits[k][-2:] for k in ("birth040", "birth050", "nms050")] == [["n/a", "n/a"]] * 3
    assert "n/a" not in visits["baseline"]
    assert [visits[k][2] for k in ("baseline", "nms050")] == ["1", "2"]  # enters: nms050 is tracked on redets
    by, _ = _table(out, "by vehicle, min_travel_px 0")
    assert [by[k][3] for k in ("baseline", "nms050")] == ["0", "1"]  # extra: the by-vehicle split reads its tracks
    assert ("bytetrack", {"new_track_thresh": 0.4}) in calls and ("bytetrack", {"new_track_thresh": 0.5}) in calls
    per, _ = _table(out, "per detection")
    assert per["baseline"][1:7] == ["80", "0", "0", "0", "0", "0"]  # vehicle ... not judged: the ignored box is not one
    assert per["baseline"][7:] == ["+0", "+0", "0", "0", "0", "0"]  # the losses' reference is masked too
    assert per["nms050"][1] == "79" and per["nms050"][9] == "1"  # vehicle; lost, against --dets
    assert per["birth040"][1] == "unchanged (tracker setting)"
    enters, _ = _table(out, "baseline enters, in judging order, min_travel_px 0")  # found too, to set beside --tracks
    assert [r[1:] for r in enters.values()] == [["10", "1000", "found", "7", "1.0"]]


def test_main_dets_filters_every_detection_then_masks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(phantoms, "create", lambda name, **params: create("groundplane"))
    car = [Detection(b.frame, b.ts_ms, b.bbox, 0.9, "car") for b in _pass(0, 100, F)]
    # centre (950, 75): in the ignored region; holds car frame 0 (100, 90, 180, 150), centre (140, 120), wholly
    container = Detection(0, 0, (100, 0, 1800, 150), 0.95, "car")
    dets = _write(tmp_path, "dets.jsonl", sorted(car + [container], key=lambda d: d.frame))
    files = {"--dets": dets, "--redetected": "nms050=" + dets,  # nms050 on the same file: masked like the rest
             "--gt": _write(tmp_path, "gt.jsonl", _pass(7, 100, F)),
             "--zone": _write(tmp_path, "zone.json", obj={"polygon": ZONE}),
             "--truth": _write(tmp_path, "truth.json", obj={"enters_ms": [1000]}),
             "--ignored": _write(tmp_path, "ignored.json", obj={"regions_xyxy": [[900, 0, 1000, 100]]})}
    monkeypatch.setattr(sys, "argv", ["secondbox.py", *(x for kv in files.items() for x in kv)])

    secondbox.main()

    per, _ = _table(capsys.readouterr().out, "per detection")
    assert per["baseline"][1] == "80"
    for k in ("contain080", "contain090", "contain080_same", "contain090_same"):
        assert (per[k][1], per[k][9]) == ("79", "1"), k  # car frame 0 dropped by the masked container: lost
    assert [per[k][5] for k in ("baseline", "contain080", "contain090", "contain080_same", "contain090_same",
                                "nms050")] == ["0"] * 6  # other: the container is masked in every scored row


def test_main_dets_splits_each_step_by_vehicle_against_the_baseline_at_its_own_zone_rule(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(phantoms, "create", lambda name, **params: create("groundplane"))
    stand = [TrackBox(f, f * 100, 8, (1300, 440, 1380, 500), 1.0, "car") for f in F]  # in the zone from frame 0
    cars = _pass(7, 100, F) + stand + _pass(9, 2000, F)
    seconds = _pass(7, 100, F, h=48) + [TrackBox(f, f * 100, 8, (1300, 452, 1380, 500), 1.0, "car") for f in F]
    dets = sorted([Detection(b.frame, b.ts_ms, b.bbox, 0.9, "car") for b in cars]
                  + [Detection(b.frame, b.ts_ms, b.bbox, 0.5, "car") for b in seconds], key=lambda d: d.frame)
    redets = [d for d in dets if d.score == 0.9 and d.bbox[0] != 1300]       # no second box, no standing car
    files = {"--dets": _write(tmp_path, "dets.jsonl", dets),
             "--redetected": "nms050=" + _write(tmp_path, "redets.jsonl", redets),
             "--gt": _write(tmp_path, "gt.jsonl", _pass(7, 100, F) + stand
                            + _pass(9, 2000, range(9, 13))),  # 9: four frames, no truth visit: an enter, never found
             "--zone": _write(tmp_path, "zone.json", obj={"polygon": ZONE}),
             "--truth": _write(tmp_path, "truth.json", obj={"enters_ms": [1000]})}
    monkeypatch.setattr(sys, "argv", ["secondbox.py", *(x for kv in files.items() for x in kv)])

    secondbox.main()

    out = capsys.readouterr().out
    at0, _ = _table(out, "by vehicle, min_travel_px 0")
    at30, _ = _table(out, "by vehicle, min_travel_px 30")
    _, listed = _table(out, "extras, min_travel_px 0")
    # key, found, missed, extra, ..., removed, new
    assert [at0["baseline"][c] for c in (1, 2, 3, 10, 11)] == ["2", "[]", "3", "0", "0"]
    assert [at30["baseline"][c] for c in (1, 2, 3, 10, 11)] == ["1", "[8]", "2", "0", "0"]  # 8 never moves
    assert [at0["nms050"][c] for c in (1, 2, 3, 10, 11)] == ["1", "[8]", "1", "duplicate 5 on 8, 4 on 7", "0"]
    assert [at30["nms050"][c] for c in (10, 11)] == ["duplicate 4 on 7", "0"]
    assert "nms050" in listed
    assert at0["baseline"][4:10] == ["2: 5 on 8, 4 on 7", "0", "0", "1: 3 on 9", "0", "0"]  # duplicate ... not judged
    assert at0["nms050"][4:10] == ["0", "0", "0", "1: 2 on 9", "0", "0"]  # its own enters, not the baseline's
    extras0, _ = _table(out, "extras, min_travel_px 0")
    _, listed30 = _table(out, "extras, min_travel_px 30")
    assert listed.count("nms050") == 1 and extras0["nms050"][1:] == ["2", "10", "1000", "no truth visit", "9", "1.0"]
    assert listed30.count("baseline") == 2  # 30 px: 5 on 8 never enters
    assert [len(_table(out, f"baseline enters, in judging order, min_travel_px {px}")[1]) for px in (0, 30)] == [5, 3]


def _per_detection_clip(tmp_path, monkeypatch, capsys):
    """--dets on a clip whose nms050 file loses 3 vehicle-frames (1 nested), keeps 2 at a lower fit, drops the
    baseline's one duplicate and adds a part: every per-detection count differs from its neighbours."""
    monkeypatch.setattr(phantoms, "create", lambda name, **params: create("groundplane"))
    car = {b.frame: Detection(b.frame, b.ts_ms, b.bbox, 0.9, "car") for b in _pass(0, 100, F)}

    def at(f, dx1=0, dy1=0, dx2=0, dy2=0, score=0.9):
        x1, y1, x2, y2 = car[f].bbox
        return Detection(f, f * 100, (x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2), score, "car")

    nested_gt = TrackBox(60, 6000, 8, (110, 700, 150, 740), 1.0, "car")  # wholly inside vehicle 7 at frame 60
    base = sorted([*car.values(), at(30, 10, 0, 10, 0),  # a duplicate of 7
                   Detection(60, 6000, nested_gt.bbox, 0.8, "car")], key=lambda d: d.frame)  # pairs with 8
    step = sorted([d for f, d in car.items() if f not in (20, 21, 50, 51)]  # 7 lost in 20, 21; 8 lost (nested) in 60
                  + [at(50, dy2=-15), at(51, dy2=-15)]  # IoU 1.0 to 0.75: fit lost
                  + [at(40, 20, 10, -40, -30, score=0.5)], key=lambda d: d.frame)  # part of 7, which is paired
    files = {"--dets": _write(tmp_path, "dets.jsonl", base),
             "--redetected": "nms050=" + _write(tmp_path, "redets.jsonl", step),
             "--gt": _write(tmp_path, "gt.jsonl", _pass(7, 100, F) + [nested_gt]),
             "--zone": _write(tmp_path, "zone.json", obj={"polygon": ZONE}),
             "--truth": _write(tmp_path, "truth.json", obj={"enters_ms": [1000]})}
    monkeypatch.setattr(sys, "argv", ["secondbox.py", *(x for kv in files.items() for x in kv)])
    secondbox.main()
    return capsys.readouterr().out


def test_main_dets_prints_each_per_detection_cell_under_its_own_column(tmp_path, monkeypatch, capsys):
    out = _per_detection_clip(tmp_path, monkeypatch, capsys)
    per, _ = _table(out, "per detection")
    assert ("| key | vehicle | duplicate | part | only | other | not judged | duplicate change | part change | lost "
            "| lost nested | only lost | fit lost |") in out.splitlines()
    assert per["baseline"] == ["baseline", "81", "1", "0", "0", "0", "0", "+0", "+0", "0", "0", "0", "0"]
    assert per["nms050"] == ["nms050", "78", "0", "1", "0", "0", "0", "-1", "+1", "3", "1", "0", "2"]


def test_main_dets_names_the_vehicles_each_step_loses_with_their_frame_counts(tmp_path, monkeypatch, capsys):
    out = _per_detection_clip(tmp_path, monkeypatch, capsys)
    lost, keys = _table(out, "vehicle-frames lost against the baseline")
    assert "| key | lost | lost nested | only lost | fit lost |" in out.splitlines()
    assert lost["nms050"] == ["nms050", "7: 2, 8: 1", "8: 1", "0", "7: 2"]  # vehicle: its frames
    assert lost["baseline"] == ["baseline", "0", "0", "0", "0"]
    assert lost["contain080"] == ["contain080", "8: 1", "8: 1", "0", "0"]  # 8's box lies inside 7's car box
    assert "birth040" not in keys  # a tracker setting changes no detection


@pytest.mark.parametrize("values", [["runs/redets.jsonl"], ["baseline=runs/redets.jsonl"],  # no key; a step's own key
                                    ["nms050=runs/a.jsonl", "nms050=runs/b.jsonl"]])  # a repeated key
def test_main_rejects_a_redetected_value_that_is_not_a_new_key_and_a_path(values, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["secondbox.py", "--dets", "d.jsonl", "--truth", "t.json", "--gt", "g.jsonl",
                                      "--zone", "z.json", *(x for v in values for x in ("--redetected", v))])
    with pytest.raises(SystemExit):
        secondbox.main()


def test_main_tracks_splits_saved_tracks_by_vehicle_without_a_tracker(tmp_path, monkeypatch, capsys):
    for module in (phantoms, trackers):  # phantoms holds its own reference to create
        monkeypatch.setattr(module, "create", lambda *_, **__: pytest.fail("tracks mode created a tracker"))
    tracks = _pass(1, 100, F) + _pass(2, 100, range(12, 31), h=48)
    files = {"--tracks": _write(tmp_path, "tracks.jsonl", tracks),
             "--gt": _write(tmp_path, "gt.jsonl", _pass(7, 100, F) + _pass(8, 1900, F)),  # 8: no track, missed
             "--zone": _write(tmp_path, "zone.json", obj={"polygon": ZONE})}
    monkeypatch.setattr(sys, "argv", ["secondbox.py", *(x for kv in files.items() for x in kv)])

    secondbox.main()

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "truth vehicles 2 | enters 2 | min_travel_px 0"  # found 7 and missed 8, never found alone
    assert "found 1: 1 on 7" in out and "duplicate 1: 2 on 7" in out and "extra 1 | missed [8]" in out


def test_main_tracks_counts_every_judged_enter_not_found_as_extra_and_no_not_judged_one(tmp_path, monkeypatch, capsys):
    gt = _pass(7, 100, range(71)) + _pass(104, 1000, range(9, 13))  # 104: four frames inside, no visit. Range 0-70
    tracks = (_pass(1, 100, F) + _pass(5, 1000, F)                 # found on 7; on 104: no truth visit
              + _pass(8, 1300, F) + _pass(9, 1600, range(75, 80), f0=65))  # on nothing annotated; enters at 79
    files = {"--tracks": _write(tmp_path, "tracks.jsonl", tracks), "--gt": _write(tmp_path, "gt.jsonl", gt),
             "--zone": _write(tmp_path, "zone.json", obj={"polygon": ZONE})}
    monkeypatch.setattr(sys, "argv", ["secondbox.py", *(x for kv in files.items() for x in kv)])

    secondbox.main()

    out = capsys.readouterr().out.splitlines()
    assert "no truth visit 1: 5 on 104" in out and "no annotated box 1: 8" in out and "not judged 1: 9" in out
    assert "extra 2 | missed []" in out


def test_main_tracks_passes_min_travel_px_to_the_tracks_counter(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(phantoms, "create", lambda *_, **__: pytest.fail("tracks mode created a tracker"))
    standing = [TrackBox(f, f * 100, 7, (100, 440, 180, 500), 1.0, "car") for f in range(41)]  # in the zone from 0
    tracks = [TrackBox(b.frame, b.ts_ms, 1, b.bbox, 0.9, "car") for b in standing]
    args = ["--tracks", _write(tmp_path, "tracks.jsonl", tracks), "--gt", _write(tmp_path, "gt.jsonl", standing),
            "--zone", _write(tmp_path, "zone.json", obj={"polygon": ZONE})]
    out = {}
    for px in ("0", "30"):
        monkeypatch.setattr(sys, "argv", ["secondbox.py", *args, "--min-travel-px", px])
        secondbox.main()
        out[px] = capsys.readouterr().out.splitlines()
    assert "found 1: 1 on 7" in out["0"] and "extra 0 | missed []" in out["0"]
    assert "found 0" in out["30"] and "extra 0 | missed [7]" in out["30"]
