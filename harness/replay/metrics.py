from __future__ import annotations

from collections import Counter

from .zones import ZoneEvent


def summarize(events: list[ZoneEvent]) -> dict:
    kinds = Counter(e.kind for e in events)
    per_track = Counter(e.track_id for e in events if e.kind == "enter")
    return {
        "enters": kinds["enter"],
        "exits": kinds["exit"],
        "net_balance": kinds["enter"] - kinds["exit"],
        "tracks_with_multiple_enters": sum(1 for v in per_track.values() if v > 1),
    }


def count_error(summary: dict, expected_visits: int) -> dict:
    return {
        "expected_visits": expected_visits,
        "enter_error": summary["enters"] - expected_visits,
        "enter_error_pct": round(100 * (summary["enters"] - expected_visits) / max(expected_visits, 1), 1),
    }
