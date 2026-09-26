"""Ground truth in MOTChallenge format, and scripts/trackeval_run.py against a local TrackEval clone.

The TrackEval tests skip when the clone is absent (CI): it is an external tool, cloned per docs/case-study-tracking.md.
"""
import importlib.util
from pathlib import Path

import pytest

from replay.motformat import export_gt
from replay.schema import TrackBox

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "trackeval_run.py"
TRACKEVAL = Path(__file__).resolve().parents[2] / ".cache" / "TrackEval"


def test_export_gt_writes_one_based_rows_marked_for_evaluation(tmp_path):
    gt = [TrackBox(0, 0, 7, (10.0, 20.0, 50.0, 80.0)), TrackBox(1, 33, 7, (12.0, 20.0, 52.0, 80.0))]

    export_gt(gt, tmp_path / "gt.txt")

    # frame (1-based), id, left, top, width, height, consider=1, class=1, visibility=1
    assert (tmp_path / "gt.txt").read_text().splitlines() == [
        "1,7,10.00,20.00,40.00,60.00,1,1,1", "2,7,12.00,20.00,40.00,60.00,1,1,1"]


def _two_cars(frames=10, switch_at=None):
    """Car 1 drives right along y=100, car 2 along y=400. switch_at: car 1 gets a new ID from that frame on."""
    out = []
    for f in range(frames):
        tid = 3 if switch_at is not None and f >= switch_at else 1
        out.append(TrackBox(f, f * 33, tid, (100.0 + 10 * f, 100.0, 180.0 + 10 * f, 160.0)))
        out.append(TrackBox(f, f * 33, 2, (600.0 - 10 * f, 400.0, 680.0 - 10 * f, 460.0)))
    return out


@pytest.mark.skipif(not TRACKEVAL.exists(), reason="TrackEval not cloned into .cache/TrackEval")
def test_trackeval_scores_perfect_tracks_one_and_counts_a_switch():
    spec = importlib.util.spec_from_file_location("trackeval_run", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    res = mod.evaluate(_two_cars(), {"perfect": _two_cars(), "switched": _two_cars(switch_at=5)}, TRACKEVAL)

    assert res["perfect"]["HOTA"] == pytest.approx(1.0)
    assert res["perfect"]["IDF1"] == pytest.approx(1.0)
    assert res["perfect"]["IDSW"] == 0
    assert res["switched"]["IDSW"] == 1
    assert res["switched"]["IDF1"] == pytest.approx(0.75)  # car 1 keeps its best ID for 5 of 10 frames
