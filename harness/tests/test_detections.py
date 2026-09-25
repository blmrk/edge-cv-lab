from pathlib import Path

import pytest

from replay.detections import frame_from_filename, is_frame_source


@pytest.mark.parametrize("path, frame", [
    ("img00001.jpg", 0),  # UA-DETRAC: 1-based file numbers, like the XML frame num
    ("000123.jpg", 122),  # MOT style
    ("/data/MVI_20011/img00457.jpg", 456),
    (Path("seq_2/img00010.png"), 9),  # digits in the directory do not count
    ("MVI_20011_img00001.jpg", 0),  # nor digits earlier in the stem
])
def test_frame_from_filename_is_trailing_number_minus_one(path, frame):
    assert frame_from_filename(path) == frame


@pytest.mark.parametrize("path", ["frame.jpg", "img00001_left.jpg", "clip"])
def test_frame_from_filename_without_trailing_digits_is_none(path):
    assert frame_from_filename(path) is None


def test_frame_sources_need_fps(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "img00001.jpg").write_bytes(b"")
    (tmp_path / "list.txt").write_text("frames/img00001.jpg\n")
    assert is_frame_source(tmp_path / "frames")  # directory
    assert is_frame_source(str(tmp_path / "frames" / "*.jpg"))  # glob
    assert is_frame_source(tmp_path / "list.txt")  # list of frames
    assert is_frame_source(tmp_path / "frames" / "IMG00001.JPG")  # single image


@pytest.mark.parametrize("src", ["clip.mp4", "clip.wmv", "clip.ts", "CLIP.MOV", "missing.mp4",
                                 "rtsp://mediamtx:8554/cam1", "rtsp://cam/live?channel=1", "0"])
def test_videos_streams_and_webcams_are_not_frame_sources(src):
    assert not is_frame_source(src)
