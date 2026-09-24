"""Exactly-once check for the sim: what the running lab stored vs what each seeded scene must produce."""
from __future__ import annotations

from collections import Counter

from .synth import ZONE, by_frame, generate_traffic
from .zones import DebouncedZoneCounter, NaiveZoneCounter


def expected_events(seed: int) -> Counter:
    """(counter, kind) -> events the sim publishes for one seed. Mirrors services/sim/src/main.py."""
    boxes, _ = generate_traffic(seed=seed)
    counters = {"naive": NaiveZoneCounter(ZONE, "centroid"), "debounced": DebouncedZoneCounter(ZONE)}
    n: Counter = Counter()
    for _, bucket in by_frame(boxes):
        for name, counter in counters.items():
            for b in bucket:
                for ev in counter.update(b):
                    n[(name, ev.kind)] += 1
    return n


def reconcile(expected: Counter, stored: Counter, distinct: Counter) -> dict:
    """distinct: stored rows counted once per payload. Lost/extra compare unique payloads with the scene,
    so a redelivery stored under a new event_id is a duplicate and cannot hide a lost event."""
    keys = set(expected) | set(stored)
    return {"expected": sum(expected.values()), "stored": sum(stored.values()),
            "lost": sum(max(expected[k] - distinct[k], 0) for k in keys),
            "extra": sum(max(distinct[k] - expected[k], 0) for k in keys),
            "duplicates": sum(stored[k] - distinct[k] for k in keys)}


def finished_seeds(truth: set, seen: set) -> list:
    """The sim plays seeds in order, so a seed has finished if its ground truth arrived or a later seed has
    events. Seeds in that range with no rows at all were lost outright and still count as finished."""
    if not truth and not seen:
        return []
    return sorted(truth | set(range(min(truth | seen), max(seen, default=0))))
