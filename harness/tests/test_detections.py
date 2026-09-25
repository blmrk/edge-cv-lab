from pathlib import Path

import pytest

from replay.detections import frame_from_filename, is_video_file


@pytest.mark.parametrize("path, frame", [
    ("img00001.jpg", 0),  # UA-DETRAC: 1-based file numbers, like the XML frame num
    ("000123.jpg", 122),  # MOT style
    ("/data/MVI_20011/img00457.jpg", 456),
    (Path("seq_2/img00010.png"), 9),  # digits in the directory or mid-stem do not count
])
def test_frame_from_filename_is_trailing_number_minus_one(path, frame):
    assert frame_from_filename(path) == frame


@pytest.mark.parametrize("path", ["frame.jpg", "img00001_left.jpg", "clip"])
def test_frame_from_filename_without_trailing_digits_is_none(path):
    assert frame_from_filename(path) is None


@pytest.mark.parametrize("name", ["clip.mp4", "clip.avi", "clip.mov", "clip.mkv", "clip.m4v", "clip.webm",
                                  "clip.mpg", "clip.mpeg", "CLIP.MP4"])
def test_existing_video_file_is_video(tmp_path, name):
    (tmp_path / name).write_bytes(b"")
    assert is_video_file(tmp_path / name)


def test_frame_sources_are_not_video(tmp_path):
    # each of these needs --fps: images carry no frame rate
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "img00001.jpg").write_bytes(b"")
    (tmp_path / "list.txt").write_text("frames/img00001.jpg\n")
    assert not is_video_file(tmp_path / "frames")  # directory
    assert not is_video_file(str(tmp_path / "frames" / "*.jpg"))  # glob
    assert not is_video_file(tmp_path / "list.txt")  # list of frames
    assert not is_video_file(tmp_path / "frames" / "img00001.jpg")  # single image
    assert not is_video_file(tmp_path / "missing.mp4")  # not a file
