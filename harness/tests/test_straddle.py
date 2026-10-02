import json
import sys

import pytest

from replay import straddle
from replay.detections import Detection
from replay.schema import TrackBox

LEFT, RIGHT = (100, 300, 200, 380), (210, 300, 310, 380)  # two annotated cars side by side, 10 px apart
ACROSS = (160, 305, 260, 375)                             # a detection across the gap, half on each
BAND = [(0, 290), (400, 290), (400, 400), (0, 400)]


def _dets(*boxes, f=0, score=0.5):
    return [Detection(f, f * 40, b, score, "car") for b in boxes]


def _gt(*boxes, f=0):
    return [TrackBox(f, f * 40, tid, b, 1.0, "car") for tid, b in enumerate(boxes, 1)]


def test_a_detection_across_two_side_by_side_annotated_vehicles_is_a_straddle():
    dets, gt = _dets(LEFT, RIGHT, ACROSS, score=1.0), _gt(LEFT, RIGHT)  # scored 1.0 like the annotations: no matter
    assert straddle.straddles(dets, gt) == {0: [(dets[2], gt[0], gt[1])]}  # and neither car's own box is one


@pytest.mark.parametrize("why, gt, det", [  # each detection fails one rule
    ("touch: between them in x, but a lane further on, touching neither", [LEFT, RIGHT], (160, 500, 260, 580)),
    ("match: IoU 0.71 with LEFT, so it is a second box on LEFT", [LEFT, RIGHT], (100, 300, 240, 380)),
    ("match: a third annotated vehicle sits in the gap, and the box is on it", [LEFT, RIGHT, ACROSS], ACROSS),
    ("rows: their vertical ranges overlap by 35 px, under half of 80", [LEFT, (210, 345, 310, 425)],
     (160, 320, 260, 400)),
    ("pair: the two overlap each other at IoU 0.18", [LEFT, (170, 300, 270, 380)], (150, 310, 230, 370)),
    ("centre: it covers both and reaches far past them", [LEFT, RIGHT], (100, 300, 500, 380)),
])
def test_each_rule_on_its_own_rejects_a_detection(why, gt, det):
    assert straddle.straddles(_dets(det), _gt(*gt)) == {}, why


def test_mask_drops_detections_centred_in_an_ignored_region_edges_included():
    inside, on_edge, overhang, clear = (40, 40, 60, 60), (90, 90, 110, 110), (80, 80, 160, 160), (200, 200, 240, 240)
    kept = straddle.mask(_dets(inside, on_edge, overhang, clear), [(0, 0, 100, 100), (500, 500, 600, 600)])
    assert [d.bbox for d in kept] == [overhang, clear]  # overhang reaches into the region, but its centre is outside


def test_summary_counts_detections_unmatched_straddles_and_frames():
    dets = _dets(LEFT, RIGHT, ACROSS, f=0) + _dets(LEFT, (500, 100, 540, 130), f=1) + _dets(LEFT, RIGHT, f=2)
    gt = _gt(LEFT, RIGHT, f=0) + _gt(LEFT, RIGHT, f=1) + _gt(LEFT, RIGHT, f=2)
    assert straddle.summary(dets, gt) == {"detections": 7, "frames": 3, "unmatched": 2, "straddles": 1,
                                          "straddle_frames": 1, "straddle_share": 0.1429,
                                          "straddle_frame_share": 0.3333}


def _drive(right_annotated=range(60)):
    """Two cars and a box across them drive down into BAND, tracks 1, 2 and 3; the right car is annotated only in
    right_annotated frames, so the box across is a straddle only there."""
    dets, tracks, gt = [], [], []
    for f in range(60):
        dy = 3 * f - 100
        for tid, (x1, y1, x2, y2) in ((3, ACROSS), (1, LEFT), (2, RIGHT)):  # the box across first in each frame
            b = (x1, y1 + dy, x2, y2 + dy)
            dets.append(Detection(f, f * 100, b, 0.5, "car"))
            tracks.append(TrackBox(f, f * 100, tid, b, 0.5, "car"))
            if tid == 1 or (tid == 2 and f in right_annotated):
                gt.append(TrackBox(f, f * 100, tid, b, 1.0, "car"))
    return dets, tracks, gt


def test_enters_table_gives_each_enter_its_straddle_share_and_whether_it_entered_on_one():
    rows = {r["track_id"]: r for r in straddle.enters_table(*_drive(), BAND)}
    assert [(rows[t]["straddle_share"], rows[t]["straddle_at_enter"]) for t in (1, 2, 3)] == [
        (0.0, False), (0.0, False), (1.0, True)]
    assert rows[3]["boxes"] == rows[3]["straddle_boxes"] == 60


@pytest.mark.parametrize("annotated, at_enter", [(range(30), True), (range(30, 60), False)])
def test_straddle_at_enter_looks_at_the_enter_frame_alone(annotated, at_enter):
    row = next(r for r in straddle.enters_table(*_drive(annotated), BAND) if r["track_id"] == 3)
    assert row["frame"] < 30  # it enters in the first half, whichever half the right car is annotated in
    assert (row["straddle_boxes"], row["straddle_share"], row["straddle_at_enter"]) == (30, 0.5, at_enter)


@pytest.mark.parametrize("why, car", [
    ("overlaps the box across at IoU 0.2, more than any other kept box", (190, 340, 270, 440)),
    ("overlaps no kept box at all, and its frame opens with the box across", (330, 300, 390, 380)),
])
def test_a_track_box_whose_own_detection_was_masked_stands_for_no_detection(why, car):
    dets, tracks, gt = _drive()  # tracks are built from every detection, so track 4 runs on through the mask
    for f in range(60):
        b = (car[0], car[1] + 3 * f - 100, car[2], car[3] + 3 * f - 100)
        dets.append(Detection(f, f * 100, b, 0.5, "car"))
        tracks.append(TrackBox(f, f * 100, 4, b, 0.5, "car"))
    cx = (car[0] + car[2]) / 2
    kept = straddle.mask(dets, [(cx - 5, -1000, cx + 5, 1000)])  # an ignored column holding car 4's centre alone
    rows = {r["track_id"]: r for r in straddle.enters_table(kept, tracks, gt, BAND)}
    assert len(kept) == 180 and (rows[3]["straddle_boxes"], rows[3]["straddle_at_enter"]) == (60, True), why
    assert (rows[4]["boxes"], rows[4]["straddle_boxes"], rows[4]["straddle_at_enter"]) == (60, 0, False), why


def _files(tmp_path, dets, tracks, gt):
    def jsonl(name, boxes):
        rows = [{"frame": b.frame, "ts_ms": b.ts_ms, "bbox": list(b.bbox), "score": b.score, "cls": b.cls,
                 **({"track_id": b.track_id} if isinstance(b, TrackBox) else {})} for b in boxes]
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return str(tmp_path / name)

    (tmp_path / "ignored.json").write_text(json.dumps({"regions_xyxy": [[880, 0, 960, 60]]}))
    (tmp_path / "zone.json").write_text(json.dumps({"polygon": BAND}))
    return {"dets": jsonl("dets.jsonl", dets), "tracks": jsonl("tracks.jsonl", tracks), "gt": jsonl("gt.jsonl", gt),
            "ignored": str(tmp_path / "ignored.json"), "zone": str(tmp_path / "zone.json")}


def _main(monkeypatch, files, *keys):
    monkeypatch.setattr(sys, "argv", ["straddle.py", *(a for k in keys for a in (f"--{k}", files[k]))])
    straddle.main()


def test_main_masks_then_prints_the_summary_the_enters_with_a_straddle_and_the_totals(tmp_path, monkeypatch, capsys):
    dets, tracks, gt = _drive()
    dets.append(Detection(0, 0, (900, 10, 940, 50), 0.5, "car"))  # centred in the ignored region: dropped
    files = _files(tmp_path, dets, tracks, gt)

    _main(monkeypatch, files, "dets", "gt", "ignored", "tracks", "zone")

    out = capsys.readouterr().out
    assert "detections 181 | 1 centred in an ignored region, dropped | 180 kept in 60 frames" in out
    assert "unmatched 60 | straddles 60 (0.3333 of kept) in 60 frames (1.0 of frames)" in out
    assert "enters 3 | straddle at enter 1 | straddle share 0.5 or more 1" in out
    table = [line for line in out.splitlines() if line.startswith("| ") and not line.startswith("| enter")]
    assert len(table) == 1 and table[0].endswith("| 3 | 60 | 60 | 1.0 | True |")  # only the enter with a straddle


def test_main_without_tracks_prints_the_summary_alone_and_tracks_need_a_zone(tmp_path, monkeypatch, capsys):
    files = _files(tmp_path, *_drive())

    _main(monkeypatch, files, "dets", "gt", "ignored")

    out = capsys.readouterr().out
    assert "straddles 60" in out and "enters" not in out
    with pytest.raises(SystemExit):
        _main(monkeypatch, files, "dets", "gt", "ignored", "tracks")
