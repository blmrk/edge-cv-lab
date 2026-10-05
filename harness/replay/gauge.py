# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""Live occupancy for a device to publish: a DebouncedZoneCounter's committed-inside visits whose track was seen within
seen_within_ms, sampled at most once per every_ms. Events cannot give this live (docs/case-study-balance.md): the counter
holds each enter until its visit closes, and learns a track is gone only lost_ms after it was last seen.

Off when seen_within_ms is 0, the services' default: sample() then returns None and the device publishes nothing.
Measured as the closest live view on the real clip with seen_within_ms=500 and the counter's min_travel_px=30.
"""
from __future__ import annotations


class OccupancyGauge:
    def __init__(self, counter, seen_within_ms: int, every_ms: int = 1000):
        self.counter, self.seen_within_ms, self.every_ms = counter, seen_within_ms, every_ms
        self._next_ms: int | None = None

    def sample(self, now_ms: int) -> int | None:
        """Vehicles in the zone now, or None when off or not yet due. Takes the current time, not a box, so call it on
        frames with no detections too."""
        if self.seen_within_ms <= 0 or (self._next_ms is not None and now_ms < self._next_ms):
            return None
        self._next_ms = now_ms + self.every_ms
        return len(self.counter.open_visits(now_ms, seen_within_ms=self.seen_within_ms))
