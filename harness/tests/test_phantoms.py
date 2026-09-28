import json
from pathlib import Path

from replay import phantoms
from replay.detections import Detection, read_detections
from replay.trackers import create

FX = Path(__file__).resolve().parent.parent / "fixtures"
DASH = (280, 280, 304, 296)  # a lane marking scored as a car: small, and it never moves


def _dash(frames, score=0.2):
    return [Detection(f, f * 100, DASH, score, "car") for f in frames]


def _car(frames, y=260):
    return [Detection(f, f * 100, (10 * f - 40, y, 10 * f + 40, y + 60), 0.9, "car") for f in frames]


def test_static_boxes_learns_a_box_that_sits_still_not_a_car_driving_past():
    dets = sorted(_dash(range(0, 100, 2)) + _car(range(100)), key=lambda d: d.frame)  # dash in half the frames
    static = phantoms.static_boxes(dets, min_share=0.25)
    assert static == [DASH]


def test_static_mask_never_learns_from_the_frames_it_filters():
    # a box standing still in the first half only is learned from the first half and so masked in the second, where
    # it is absent; the first half keeps it, because what masks the first half is learned from the second
    first_only = [Detection(f, f * 100, (500, 500, 540, 520), 0.3, "car") for f in range(50)]
    dets = sorted(first_only + _dash(range(100)), key=lambda d: d.frame)
    kept = phantoms.static_mask(dets, min_share=0.25)
    assert [d for d in kept if d.bbox == DASH] == []  # still in both halves: masked in both
    assert kept == first_only


def test_static_mask_keeps_a_stopped_car_over_a_masked_dash():
    # covers the dash (IoU 0.06), stopped too briefly (1 s) to be learned as static itself
    stopped = [Detection(f, f * 100, (250, 240, 350, 300), 0.8, "car") for f in range(45, 55)]
    dets = sorted(_dash(range(100)) + stopped, key=lambda d: d.frame)
    assert phantoms.static_mask(dets, min_share=0.25) == stopped


def test_min_area_drops_boxes_smaller_than_the_threshold():
    dets = _dash([0]) + _car([0])
    assert phantoms.min_area(dets, 32 * 32) == _car([0])  # the dash is 24 x 16 px


def test_table_has_a_row_per_step_and_counts_the_queue_cars_that_stop(monkeypatch):
    # bytetrack needs ultralytics; groundplane stands in, so this checks the table, not ByteTrack
    monkeypatch.setattr(phantoms, "create", lambda name, **params: create("groundplane"))
    dets = read_detections(FX / "queue.dets.jsonl")
    poly = [tuple(p) for p in json.loads((FX / "zone.json").read_text())["polygon"]]
    truth = json.loads((FX / "queue.truth.json").read_text())["enters_ms"]
    rows, tracks = phantoms.table(dets, poly, truth, queue=(dets, poly, truth))
    assert [r["step"] for r in rows] == [s[1] for s in phantoms.STEPS]
    assert set(tracks) == {s[0] for s in phantoms.STEPS}
    base = rows[0]
    assert (base["matched"], base["false_visits"], base["queue_matched"]) == (6, 0, 6)  # groundplane counts all six
    assert base["static"] == 0 and base["moving"] == base["enters"]  # the queue's cars all drive in
    assert all(r["queue_matched"] == "n/a" for r, s in zip(rows, phantoms.STEPS) if s[2])  # ByteTrack-only settings
