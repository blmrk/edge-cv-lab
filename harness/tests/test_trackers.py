import json
from pathlib import Path

from replay.detections import frames, read_detections
from replay.idmetrics import identity_report
from replay.motformat import export_dets, export_tracks, import_tracks
from replay.schema import read_tracks
from replay.score import match
from replay.trackers import available, create, run_tracker
from replay.zones import DebouncedZoneCounter, run

FX = Path(__file__).resolve().parent.parent / "fixtures"
POLY = [tuple(p) for p in json.loads((FX / "zone.json").read_text())["polygon"]]
DETS = read_detections(FX / "queue.dets.jsonl")
GT = list(read_tracks(FX / "queue.gt.jsonl"))
TRUTH = json.loads((FX / "queue.truth.json").read_text())["enters_ms"]


def visits(tracks):
    enters = [e.ts_ms for e in run(DebouncedZoneCounter(POLY), sorted(tracks, key=lambda b: (b.frame, b.track_id)))
              if e.kind == "enter"]
    return match(enters, TRUTH, 2000)


def test_registry_lists_builtins():
    assert {"greedy_iou", "groundplane"} <= set(available())


def test_iou_tracker_hands_identity_to_the_next_car():
    tracks = run_tracker(create("greedy_iou"), DETS)
    assert identity_report(GT, tracks)["id_transfers"] >= 3
    assert visits(tracks)["missed_visits"] >= 3          # separate customers merged into one visit


def test_short_buffer_trades_handover_for_fragmentation():
    tracks = run_tracker(create("greedy_iou", max_age=5), DETS)
    r = identity_report(GT, tracks)
    assert r["id_transfers"] == 0 and r["pred_ids"] >= 2 * r["gt_objects"]


def test_groundplane_keeps_one_id_per_car():
    tracks = run_tracker(create("groundplane"), DETS)
    r = identity_report(GT, tracks)
    assert (r["id_switches"], r["id_transfers"], r["pred_ids"]) == (0, 0, r["gt_objects"])
    assert visits(tracks)["f1"] == 1.0


def test_trackers_see_empty_frames():
    seen = [f for f, _, _ in frames(DETS)]
    assert seen == list(range(seen[0], seen[-1] + 1))


def test_mot_bridge_round_trip(tmp_path):
    tracks = run_tracker(create("groundplane"), DETS)
    export_dets(DETS, tmp_path / "det.txt")
    export_tracks(tracks, tmp_path / "res.txt")
    ts = {f: t for f, t, _ in frames(DETS)}
    back = import_tracks(tmp_path / "res.txt", ts)
    assert [(b.frame, b.track_id, b.ts_ms) for b in back] == \
           [(b.frame, b.track_id, b.ts_ms) for b in sorted(tracks, key=lambda b: (b.frame, b.track_id))]
    assert identity_report(GT, back)["id_switches"] == 0


def test_lost_track_inside_zone_is_closed_not_left_open():
    # With a short buffer every car changes ID as it pulls away, so its window ID just stops.
    from replay.zones import DebouncedZoneCounter, run
    tracks = sorted(run_tracker(create("greedy_iou", max_age=5), DETS), key=lambda b: (b.frame, b.track_id))
    never = [e for e in run(DebouncedZoneCounter(POLY, lost_ms=10**9), tracks) if e.kind == "exit"]
    closed = [e for e in run(DebouncedZoneCounter(POLY), tracks) if e.kind == "exit"]
    assert len(never) == 0 and len(closed) >= 5


def test_compare_reports_gt_coverage_next_to_identity_metrics(monkeypatch, capsys):
    # identity metrics only cover GT boxes some prediction overlaps, so the table must say how many that is
    import sys
    from replay import compare
    fx = Path(__file__).resolve().parent.parent / "fixtures"
    monkeypatch.setattr(sys, "argv", ["compare", "--dets", str(fx / "queue.dets.jsonl"), "--zone", str(fx / "zone.json"),
                                      "--truth", str(fx / "queue.truth.json"), "--gt", str(fx / "queue.gt.jsonl"),
                                      "--trackers", "groundplane"])
    compare.main()
    header, _, row = capsys.readouterr().out.splitlines()[:3]
    cols = [c.strip() for c in header.strip("|").split("|")]
    vals = dict(zip(cols, [c.strip() for c in row.strip("|").split("|")]))
    assert vals["gt objects"] == "6"  # the queue fixture has six cars
    assert 0 < float(vals["gt boxes matched pct"]) <= 100
