"""scripts/mtid_to_gt.py on synthetic MTID annotations. The real annotations stay local (media/ is gitignored)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from replay.schema import read_tracks

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mtid_to_gt.py"
_spec = importlib.util.spec_from_file_location("mtid_to_gt", SCRIPT)
m2g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m2g)

HEADER = "RGB mask file;RGB file;Object ID;Annotation tag;"


def _row(mtid_frame, oid, tag, pts):
    name = f"seq3-infra_{mtid_frame:07d}"
    return f"./rgbMasks/{name}.png;{name}.jpg;{oid};{tag};;0;" + " ".join(f"{v:g}" for v in pts) + " "


def _rect(x, y, w=40, h=30):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _write(root, folders):
    """folders: {"0": [row, ...]} -> root/<folder>/annotations.csv, like the Infrastructure folder."""
    for name, rows in folders.items():
        (root / name).mkdir(parents=True)
        (root / name / "annotations.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    return root


# Object ID 200 (car): MTID frames 1-3, then 5-6 after one unannotated frame, then 40-41 after 33.
CAR = [_row(f, 200, "Car", _rect(100 + 2 * f, 300)) for f in (1, 2, 3, 5, 6, 40, 41)]
CYCLIST = [_row(f, 201, "Cyclist", _rect(500, 100, 10, 20)) for f in (1, 2, 3)]
VAN = [_row(3, 202, "Van", [232, 409, 298, 360, 305, 361, 137, 366, 201, 428])]


def _convert(tmp_path, **kw):
    return m2g.convert(_write(tmp_path / "Infrastructure", {"0": CAR + CYCLIST + VAN}), **kw)


def _ids_by_frame(tracks, cls):
    return {t.frame: t.track_id for t in tracks if t.cls == cls}


def test_mtid_frame_becomes_zero_based_with_ts_at_30_fps(tmp_path):
    tracks, _ = _convert(tmp_path)
    assert [(t.frame, t.ts_ms) for t in tracks if t.cls == "car"] == [
        (0, 0), (1, 33), (2, 66), (4, 133), (5, 166), (39, 1300), (40, 1333)]


def test_polygon_becomes_xyxy_bbox(tmp_path):
    tracks, _ = _convert(tmp_path)
    assert [(t.frame, t.bbox, t.cls) for t in tracks if t.cls == "van"] == [(2, (137.0, 360.0, 305.0, 428.0), "van")]


def test_cyclist_excluded_by_default_and_kept_on_request(tmp_path):
    tracks, stats = _convert(tmp_path)
    assert "cyclist" not in {t.cls for t in tracks}
    assert stats["excluded_boxes"] == 3
    kept, _ = m2g.convert(tmp_path / "Infrastructure", exclude=())
    assert sum(t.cls == "cyclist" for t in kept) == 3


def test_reuse_after_long_gap_is_a_new_identity_short_gap_keeps_it(tmp_path):
    tracks, stats = _convert(tmp_path, gap_frames=5)
    ids = _ids_by_frame(tracks, "car")
    assert ids[0] == ids[1] == ids[2] == ids[4] == ids[5]  # one unannotated frame: same vehicle
    assert ids[39] == ids[40] != ids[5]  # 33 unannotated frames: the ID was reused
    assert (stats["reused_ids"], stats["gap_splits"]) == (1, 1)


def test_default_gap_splits_on_any_unannotated_frame(tmp_path):
    tracks, stats = _convert(tmp_path)
    ids = _ids_by_frame(tracks, "car")
    assert len({ids[0], ids[4], ids[39]}) == 3
    assert stats["gap_splits"] == 2
    assert stats["gaps"] == [1, 33]


def test_identities_are_unique_across_object_ids_and_output_sorted(tmp_path):
    tracks, _ = _convert(tmp_path)
    assert [(t.frame, t.track_id) for t in tracks] == sorted((t.frame, t.track_id) for t in tracks)
    assert len({t.track_id for t in tracks}) == 4  # car split in three + van


def test_folders_are_merged_and_empty_csv_is_fine(tmp_path):
    root = _write(tmp_path / "Infrastructure", {
        "0": [_row(1, 200, "Car", _rect(100, 300))],
        "1000": [_row(2, 200, "Car", _rect(102, 300))],
        "3100": [],
    })
    tracks, _ = m2g.convert(root)
    assert [(t.frame, t.track_id) for t in tracks] == [(0, 1), (1, 1)]


def test_bad_rows_fail_loud(tmp_path):
    odd = _write(tmp_path / "odd", {"0": [_row(1, 200, "Car", [1, 2, 3])]})
    with pytest.raises(ValueError, match="annotations.csv:2"):
        m2g.convert(odd)
    dup = _write(tmp_path / "dup", {"0": [_row(1, 200, "Car", _rect(0, 0))], "1": [_row(1, 200, "Car", _rect(0, 0))]})
    with pytest.raises(ValueError, match="twice"):
        m2g.convert(dup)


def test_main_writes_gt_jsonl_and_prints_stats(tmp_path, monkeypatch, capsys):
    root = _write(tmp_path / "Infrastructure", {"0": CAR + CYCLIST + VAN})
    out = tmp_path / "runs" / "mtid"
    monkeypatch.setattr(sys, "argv", ["mtid_to_gt.py", "--annotations", str(root), "--out", str(out),
                                      "--gap-frames", "5"])

    m2g.main()

    rows = list(read_tracks(f"{out}.gt.jsonl"))
    assert len(rows) == 8 and len({r.track_id for r in rows}) == 3
    printed = capsys.readouterr().out
    assert "8 boxes, 3 identities" in printed
    assert "1 of 3 Object IDs reused" in printed
    assert "3 cyclist boxes excluded" in printed
