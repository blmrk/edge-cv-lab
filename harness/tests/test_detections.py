from pathlib import Path

import pytest

from replay.detections import frame_from_filename, frame_window, is_frame_source


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


def test_frame_window_without_a_window_is_the_whole_stream():
    assert frame_window(3, 3198, 30.0) == (3, 3198)


def test_frame_window_is_start_inclusive_end_exclusive_in_video_time():
    # 20 s to 33 s of a 30 fps clip: frames 600..989, frame f shown at f / fps seconds
    assert frame_window(0, 3198, 30.0, start=20, seconds=13) == (600, 989)
    assert frame_window(0, 249, 25.0, start=2.5, seconds=0.2) == (63, 67)   # 2.52 s .. 2.68 s
    assert frame_window(0, 3198, 30.0, start=0.1, seconds=0.1) == (3, 5)      # no float drift at 0.1 * 30


def test_frame_window_clamps_to_the_stream():
    assert frame_window(0, 3198, 30.0, start=100, seconds=10) == (3000, 3198)  # runs past the end
    assert frame_window(40, 3198, 30.0, start=1, seconds=2) == (40, 89)         # starts before the first frame
    assert frame_window(0, 3198, 30.0, start=5) == (150, 3198)                  # no length: to the end


@pytest.mark.parametrize("start, seconds", [(-1, 5), (0, 0), (0, -2), (107, 5), (2.01, 0.01)])
def test_frame_window_rejects_empty_or_negative_windows(start, seconds):
    with pytest.raises(ValueError):
        frame_window(0, 3198, 30.0, start=start, seconds=seconds)
