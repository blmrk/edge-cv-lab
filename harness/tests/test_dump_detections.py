"""dump_detections.main() with ultralytics and cv2 stubbed, so frame numbering and --fps rules run in CI."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dump_detections.py"


class _Boxes:
    def __init__(self):
        self.xyxy, self.conf, self.cls = [[[1.0, 2.0, 3.0, 4.0]]], [[0.9]], [[2]]


class _Tensor(list):
    def tolist(self):
        return list(self)


def _result(path):
    b = _Boxes()
    return types.SimpleNamespace(path=str(path), boxes=types.SimpleNamespace(
        xyxy=_Tensor(b.xyxy[0]), conf=_Tensor(b.conf[0]), cls=_Tensor(b.cls[0])))


def run(monkeypatch, tmp_path, source, paths, fps=None, video_fps=25.0, calls=None, extra=()):
    """Runs main() as if ultralytics yielded one result per entry in `paths`; returns (frame, ts_ms) rows.
    `calls` collects the keyword arguments each predict() call received."""
    def predict(*_, **kw):
        (calls if calls is not None else []).append(kw)
        return iter(_result(p) for p in paths)

    model = types.SimpleNamespace(names={2: "car"}, predict=predict)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda *_: model))
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        CAP_PROP_FPS=5, VideoCapture=lambda *_: types.SimpleNamespace(get=lambda *_: video_fps)))
    spec = importlib.util.spec_from_file_location("dump_detections", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "dets.jsonl"
    argv = ["dump_detections.py", "--video", str(source), "--out", str(out)] + (["--fps", str(fps)] if fps else [])
    argv += list(extra)
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    return [(d["frame"], d["ts_ms"]) for d in map(json.loads, out.read_text().splitlines())]


def frames_dir(tmp_path, *names):
    d = tmp_path / "frames"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"")
    return d


def test_frame_directory_keeps_the_gap_left_by_an_unreadable_frame(monkeypatch, tmp_path):
    d = frames_dir(tmp_path, "img00001.jpg", "img00002.jpg", "img00004.jpg")  # img00003 failed to load
    rows = run(monkeypatch, tmp_path, d, [d / "img00001.jpg", d / "img00002.jpg", d / "img00004.jpg"], fps=25)
    assert rows == [(0, 0), (1, 40), (3, 120)]


@pytest.mark.parametrize("name", ["clip.mp4", "clip.wmv", "clip.ts"])  # any format ultralytics reads as video
def test_video_file_is_numbered_in_read_order_at_its_own_fps(monkeypatch, tmp_path, name):
    (tmp_path / name).write_bytes(b"")
    rows = run(monkeypatch, tmp_path, tmp_path / name, [tmp_path / name] * 3)
    assert rows == [(0, 0), (1, 40), (2, 80)]


@pytest.mark.parametrize("source, path", [
    ("rtsp://mediamtx:8554/cam1", "rtsp___mediamtx_8554_cam1"),  # the lab's own camera stream
    ("0", "0"),  # webcam index
])
def test_stream_is_numbered_in_read_order(monkeypatch, tmp_path, source, path):
    assert run(monkeypatch, tmp_path, source, [path] * 3) == [(0, 0), (1, 40), (2, 80)]


def test_frame_directory_without_fps_exits(monkeypatch, tmp_path):
    d = frames_dir(tmp_path, "img00001.jpg")
    with pytest.raises(SystemExit, match="--fps"):
        run(monkeypatch, tmp_path, d, [d / "img00001.jpg"])


def test_frame_numbered_zero_exits_instead_of_writing_frame_minus_one(monkeypatch, tmp_path):
    d = frames_dir(tmp_path, "frame_000000.jpg")
    with pytest.raises(SystemExit, match="start at 1"):
        run(monkeypatch, tmp_path, d, [d / "frame_000000.jpg"], fps=25)


def test_image_name_without_a_number_exits(monkeypatch, tmp_path):
    d = frames_dir(tmp_path, "frame.jpg")
    with pytest.raises(SystemExit, match="frame number"):
        run(monkeypatch, tmp_path, d, [d / "frame.jpg"], fps=25)


def test_nms_is_class_agnostic_unless_per_class_is_asked_for(monkeypatch, tmp_path):
    # per-class NMS keeps a car box and a truck box on the same vehicle; class-agnostic NMS keeps the better one
    (tmp_path / "clip.mp4").write_bytes(b"")
    calls = []
    run(monkeypatch, tmp_path, tmp_path / "clip.mp4", [], calls=calls)
    run(monkeypatch, tmp_path, tmp_path / "clip.mp4", [], calls=calls, extra=["--per-class-nms"])
    assert [c.get("agnostic_nms") for c in calls] == [True, False]


def test_dump_tracks_nms_is_class_agnostic_unless_per_class_is_asked_for(monkeypatch, tmp_path):
    calls = []
    model = types.SimpleNamespace(names={2: "car"}, track=lambda *_, **kw: calls.append(kw) or iter(()))
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda *_: model))
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        CAP_PROP_FPS=5, VideoCapture=lambda *_: types.SimpleNamespace(get=lambda *_: 30.0)))
    spec = importlib.util.spec_from_file_location("dump_tracks", SCRIPT.parent / "dump_tracks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for extra in ([], ["--per-class-nms"]):
        monkeypatch.setattr(sys, "argv", ["dump_tracks.py", "--video", "clip.mp4", "--out", str(tmp_path / "t.jsonl"),
                                          *extra])
        mod.main()
    assert [c.get("agnostic_nms") for c in calls] == [True, False]
