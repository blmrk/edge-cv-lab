from pathlib import Path

import pytest

from replay.detections import frame_from_filename


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
