# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
from replay.gauge import OccupancyGauge
from replay.schema import TrackBox
from replay.zones import DebouncedZoneCounter

SQUARE = [(100, 100), (500, 100), (500, 500), (100, 500)]


def _parked_after_driving_in(counter, frames=40):
    """Footpoint drives in from x=0 at 10 px a frame, 100 ms a frame; inside from frame 11, committed at frame 15."""
    for f in range(frames):
        counter.update(TrackBox(f, f * 100, 1, (10 * f - 20, 270, 10 * f + 20, 300)))


def test_off_by_default_publishes_nothing():
    c = DebouncedZoneCounter(SQUARE)
    _parked_after_driving_in(c)
    g = OccupancyGauge(c, seen_within_ms=0)
    assert [g.sample(t) for t in (3900, 5000, 6000)] == [None, None, None]


def test_samples_at_most_once_per_interval():
    c = DebouncedZoneCounter(SQUARE)
    _parked_after_driving_in(c)
    g = OccupancyGauge(c, seen_within_ms=500, every_ms=1000)
    assert [g.sample(t) for t in (3900, 4000, 4899, 4900)] == [1, None, None, 0]  # 4900: last seen 1 s ago


def test_drops_a_visit_whose_track_went_quiet_without_any_new_box():
    # the edge samples on frames with no detections too: the gauge must not wait for the next box to arrive
    c = DebouncedZoneCounter(SQUARE)
    _parked_after_driving_in(c)  # last box at 3.9 s
    g = OccupancyGauge(c, seen_within_ms=500, every_ms=100)
    assert (g.sample(4400), g.sample(4500)) == (1, 0)
