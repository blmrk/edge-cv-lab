import json
import sys

import pytest

from replay import between
from replay.detections import Detection
from replay.schema import TrackBox

LEFT, RIGHT = (100, 300, 200, 380), (210, 300, 310, 380)  # two cars side by side, 10 px apart
STRADDLE = (160, 305, 260, 375)                           # a weaker box across the gap, half on each


def _frame(*boxes, f=0):
    return [Detection(f, f * 20, b, s, "car") for b, s in boxes]


def test_a_weaker_box_across_two_side_by_side_vehicles_is_a_bridge_box():
    dets = _frame((LEFT, 0.9), (RIGHT, 0.8), (STRADDLE, 0.2))
    assert [d.bbox for d in dets if between.bridge(d, dets)] == [STRADDLE]  # and neither car is one


@pytest.mark.parametrize("why, near, weak", [  # each box fails exactly one of bridge()'s geometry rules
    ("touch: between them in x, but a lane further on, touching neither", [LEFT, RIGHT], (160, 500, 260, 580)),
    ("match: IoU 0.71 with LEFT, so it is a second box on LEFT", [LEFT, RIGHT], (100, 300, 240, 380)),
    ("rows: the two sit one behind the other, not side by side", [LEFT, (160, 200, 260, 280)], (130, 240, 230, 340)),
    ("pair: the two overlap each other at IoU 0.18", [LEFT, (170, 300, 270, 380)], (150, 310, 230, 370)),
    ("centre: it covers both and reaches far past them", [LEFT, RIGHT], (100, 300, 500, 380)),
])
def test_each_geometry_rule_on_its_own_rejects_a_box(why, near, weak):
    dets = _frame((near[0], 0.9), (near[1], 0.8), (weak, 0.2))
    assert not between.bridge(dets[2], dets), why


def test_a_bridge_box_must_score_below_both_vehicles():
    dets = _frame((LEFT, 0.9), (RIGHT, 0.3), (STRADDLE, 0.5))
    assert not between.bridge(dets[2], dets)


def test_outscored_false_drops_the_score_test():
    dets = _frame((LEFT, 0.9), (RIGHT, 0.3), (STRADDLE, 0.5))  # RIGHT scores below the box: no bridge by default
    assert between.bridge(dets[2], dets, outscored=False) == (dets[0], dets[1])


def test_enters_table_gives_each_enter_the_share_of_its_boxes_that_are_bridge_boxes():
    band = [(0, 290), (400, 290), (400, 400), (0, 400)]
    dets, tracks = [], []
    for f in range(60):                                    # two cars and a straddle box drive down into the band
        dy = 3 * f - 100
        boxes = [(LEFT, 0.9, 1), (RIGHT, 0.8, 2), (STRADDLE, 0.2, 3)]
        for (x1, y1, x2, y2), s, tid in boxes:
            b = (x1, y1 + dy, x2, y2 + dy)
            dets.append(Detection(f, f * 100, b, s, "car"))
            tracks.append(TrackBox(f, f * 100, tid, b, s, "car"))
    rows = {r["track_id"]: r for r in between.enters_table(dets, tracks, band)}
    assert (rows[1]["bridge_share"], rows[2]["bridge_share"], rows[3]["bridge_share"]) == (0.0, 0.0, 1.0)


def test_a_track_box_overlapping_no_detection_stands_for_none_even_when_its_frame_opens_with_a_bridge_box():
    band = [(0, 290), (400, 290), (400, 400), (0, 400)]
    dets, tracks = [], []
    for f in range(60):
        dy = 3 * f - 100
        for (x1, y1, x2, y2), s in ((STRADDLE, 0.2), (LEFT, 0.9), (RIGHT, 0.8)):  # the bridge box first in the frame
            dets.append(Detection(f, f * 100, (x1, y1 + dy, x2, y2 + dy), s, "car"))
        tracks.append(TrackBox(f, f * 100, 9, (330, 300 + dy, 390, 380 + dy), 0.9, "car"))  # beside them, touching none
    [row] = between.enters_table(dets, tracks, band)
    assert (row["track_id"], row["boxes"], row["bridge_boxes"]) == (9, 60, 0)


@pytest.mark.parametrize("bridged, frames, half_or_more", [
    (50, 101, 0),  # 50 of 101 boxes: 0.495, shown rounded as 0.5, is under half
    (30, 60, 1),   # 30 of 60 boxes: exactly half counts
])
def test_main_counts_share_half_or_more_on_the_exact_share_not_the_rounded_one(bridged, frames, half_or_more,
                                                                               tmp_path, monkeypatch, capsys):
    band = [(0, 290), (400, 290), (400, 400), (0, 400)]
    rows = []
    for f in range(frames):  # the box across is a bridge box only while both cars outscore it
        dy = 3 * f - 100
        for (x1, y1, x2, y2), s, tid in ((LEFT, 0.9, 1), (RIGHT, 0.8, 2), (STRADDLE, 0.2 if f < bridged else 0.95, 3)):
            rows.append({"frame": f, "ts_ms": f * 100, "bbox": [x1, y1 + dy, x2, y2 + dy], "score": s, "cls": "car",
                         "track_id": tid})
    for name, body in (("dets.jsonl", "".join(json.dumps(r) + "\n" for r in rows)),
                       ("zone.json", json.dumps({"polygon": band}))):
        (tmp_path / name).write_text(body)
    monkeypatch.setattr(sys, "argv", ["between.py", "--dets", str(tmp_path / "dets.jsonl"),
                                      "--tracks", str(tmp_path / "dets.jsonl"), "--zone", str(tmp_path / "zone.json")])

    between.main()

    out = capsys.readouterr().out
    assert f"| 3 | {frames} | {bridged} | 0.5 |" in out
    assert f"enters 3 | with any bridge box 1 | half or more {half_or_more}" in out
