"""scripts/detrac_to_gt.py on a synthetic UA-DETRAC XML. No real sequence is committed (research licence)."""
import importlib.util
import json
import sys
from pathlib import Path

from replay.schema import read_tracks
from replay.zones import DebouncedZoneCounter, run

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "detrac_to_gt.py"
_spec = importlib.util.spec_from_file_location("detrac_to_gt", SCRIPT)
d2g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d2g)

# Frames out of order, targets out of order inside a frame. Ignored region (800, 0, 960, 100) in xyxy.
# id 3 is centred inside it (dropped); id 4 overlaps it but is centred outside (kept); id 6 has no box.
XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_TEST">
  <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
  <ignored_region>
    <box left="800" top="0" width="160" height="100"/>
  </ignored_region>
  <frame density="1" num="26">
    <target_list>
      <target id="2"><box left="300" top="200" width="50" height="40"/><attribute vehicle_type="van"/></target>
    </target_list>
  </frame>
  <frame density="5" num="1">
    <target_list>
      <target id="5"><box left="10" top="20" width="30" height="40"/><attribute orientation="18.5"/></target>
      <target id="1"><box left="100.5" top="200.25" width="50" height="40"/><attribute vehicle_type="bus"/></target>
      <target id="3"><box left="820" top="10" width="40" height="40"/><attribute vehicle_type="car"/></target>
      <target id="4"><box left="770" top="10" width="40" height="40"/></target>
      <target id="6"><attribute vehicle_type="car"/></target>
    </target_list>
  </frame>
  <frame density="1" num="4">
    <target_list>
      <target id="2"><box left="250" top="200" width="50" height="40"/><attribute vehicle_type="van"/></target>
    </target_list>
  </frame>
</sequence>
"""


def _convert(tmp_path, xml=XML, **kw):
    p = tmp_path / "seq.xml"
    p.write_text(xml)
    return d2g.convert(p, **kw)


def _main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["detrac_to_gt.py", *args])
    d2g.main()


def test_frame_num_becomes_zero_based_with_ts_at_25_fps(tmp_path):
    tracks, _, _ = _convert(tmp_path)
    assert [(t.frame, t.ts_ms) for t in tracks if t.track_id == 2] == [(3, 120), (25, 1000)]
    assert {t.frame for t in tracks if t.track_id == 1} == {0}


def test_box_left_top_width_height_becomes_xyxy(tmp_path):
    tracks, _, _ = _convert(tmp_path)
    assert [t.bbox for t in tracks if t.track_id == 1] == [(100.5, 200.25, 150.5, 240.25)]


def test_target_without_box_is_skipped(tmp_path):
    tracks, _, _ = _convert(tmp_path)
    assert 6 not in {t.track_id for t in tracks}


def test_box_centred_in_ignored_region_is_dropped_and_counted(tmp_path):
    tracks, ignored, dropped = _convert(tmp_path)
    ids = {t.track_id for t in tracks}
    assert 3 not in ids and 4 in ids  # centre decides, not overlap
    assert dropped == 1
    assert ignored == [(800.0, 0.0, 960.0, 100.0)]


def test_vehicle_type_becomes_cls_defaulting_to_car(tmp_path):
    tracks, _, _ = _convert(tmp_path)
    assert {t.track_id: t.cls for t in tracks} == {1: "bus", 2: "van", 4: "car", 5: "car"}


def test_output_sorted_by_frame_then_track_id(tmp_path):
    tracks, _, _ = _convert(tmp_path)
    assert [(t.frame, t.track_id) for t in tracks] == [(0, 1), (0, 4), (0, 5), (3, 2), (25, 2)]


def _crossing_xml(n_frames=80):
    # one car, footpoint moving 10 px/frame along y=140 through the zone below
    frames = "".join(
        f'<frame num="{n}"><target_list><target id="7"><box left="{10 * (n - 1)}" top="100" width="40" height="40"/>'
        f'<attribute vehicle_type="car"/></target></target_list></frame>' for n in range(1, n_frames + 1))
    return f'<sequence name="MVI_CROSS"><ignored_region></ignored_region>{frames}</sequence>'


def test_main_with_zone_writes_truth_from_debounced_counter(tmp_path, monkeypatch):
    xml = tmp_path / "cross.xml"
    xml.write_text(_crossing_xml())
    zone = tmp_path / "zone.json"
    poly = [[200, 0], [600, 0], [600, 300], [200, 300]]
    zone.write_text(json.dumps({"polygon": poly}))
    out = tmp_path / "runs" / "MVI_CROSS"

    _main(monkeypatch, "--xml", str(xml), "--out", str(out), "--zone", str(zone))

    tracks, _, _ = d2g.convert(xml)
    enters = [e.ts_ms for e in run(DebouncedZoneCounter([tuple(p) for p in poly]), tracks) if e.kind == "enter"]
    truth = json.loads(Path(f"{out}.truth.json").read_text())
    assert len(enters) == 1  # the fixture really produces a visit, so the comparison below is not 0 == 0
    assert (truth["expected_visits"], truth["enters_ms"]) == (len(enters), enters)
    assert len(list(read_tracks(f"{out}.gt.jsonl"))) == 80
