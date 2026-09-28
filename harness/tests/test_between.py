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


def test_a_box_on_one_vehicle_or_between_vehicles_one_behind_the_other_is_not():
    on_left = (105, 302, 205, 382)                         # IoU 0.86 with LEFT: a second box on the same car
    dets = _frame((LEFT, 0.9), (RIGHT, 0.8), (on_left, 0.2))
    assert not between.bridge(dets[2], dets)
    behind = (100, 200, 200, 280)                          # same x as LEFT, a row further back: not side by side
    dets = _frame((LEFT, 0.9), (behind, 0.8), ((100, 250, 200, 330), 0.2))
    assert not any(between.bridge(d, dets) for d in dets)


def test_a_bridge_box_must_score_below_both_vehicles():
    dets = _frame((LEFT, 0.9), (RIGHT, 0.3), (STRADDLE, 0.5))
    assert not between.bridge(dets[2], dets)


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
