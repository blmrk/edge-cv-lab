"""Did every sim event reach Postgres exactly once, and how late? Needs the lab running (`make up`).

Every finished scene is compared with an offline replay of the same seed. A scene has finished when its
ground truth arrives (the sim publishes it as each seed ends) or a later seed has events. Exits 1 on any
lost, extra or duplicated event, or a finished scene whose ground truth never arrived.

    python scripts/check_delivery.py [--window NAME START END ...]    # START/END: ISO-8601 UTC
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from replay.delivery import expected_events, finished_seeds, reconcile

ROOT = Path(__file__).resolve().parents[2]


def sql(query: str) -> list[list[str]]:
    out = subprocess.run(["docker", "compose", "exec", "-T", "postgres", "psql", "-U", "postgres", "lab",
                          "-tA", "-F", "|", "-c", query], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line.split("|") for line in out.stdout.splitlines() if line]


def utc(s: str) -> str:
    # rejects anything that is not a timestamp before it reaches SQL; 'Z' spelled out for Python 3.10
    return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="sim-01")
    ap.add_argument("--window", nargs=3, action="append", default=[], metavar=("NAME", "START", "END"),
                    help="report delivery lag for events received in [START, END)")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.device):
        sys.exit(f"bad --device {a.device!r}")
    dev = f"device_id = '{a.device}'"

    truth = {int(s) for (s,) in sql(f"SELECT seed FROM ground_truth WHERE {dev}")}
    rows = sql(f"SELECT track_id / 1000, counter, kind, count(*), count(DISTINCT (track_id, ts_ms)) "
               f"FROM zone_events WHERE {dev} GROUP BY 1, 2, 3")
    seeds = finished_seeds(truth, {int(r[0]) for r in rows})
    if not seeds:
        sys.exit("no finished scenes yet: wait for the first seed to end")
    stored, distinct = Counter(), Counter()
    for seed, counter, kind, n, d in rows:
        if int(seed) in seeds:
            stored[(int(seed), counter, kind)] += int(n)
            distinct[(int(seed), counter, kind)] += int(d)
    expected = Counter({(s, c, k): n for s in seeds for (c, k), n in expected_events(s).items()})
    r = reconcile(expected, stored, distinct)
    no_truth = [s for s in seeds if s not in truth]
    print(f"finished scenes: {len(seeds)} (seeds {seeds[0]}-{seeds[-1]}) | events expected {r['expected']} "
          f"| stored {r['stored']} | lost {r['lost']} | extra {r['extra']} | duplicates {r['duplicates']}")
    if no_truth:
        print(f"finished scenes whose ground truth never arrived: seeds {no_truth}")
    if r["extra"]:
        print("extra events: did the sim restart mid-run? It replays from its first seed. Check on a fresh `make up`.")

    if a.window:  # naive events are published the moment they happen; debounced enters are held until the visit ends
        print(f"\ndelivery lag of naive events\n{'window':<12} {'events':>7} {'median':>8} {'max':>8}")
    for name, start, end in a.window:
        n, p50, mx = sql(f"SELECT count(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY lag), max(lag) FROM "
                         f"(SELECT extract(epoch FROM received_at) - ts_ms / 1000.0 AS lag FROM zone_events "
                         f"WHERE {dev} AND counter = 'naive' "
                         f"AND received_at >= '{utc(start)}' AND received_at < '{utc(end)}') t")[0]
        fmt = lambda v: f"{float(v):.2f} s" if v else "-"
        print(f"{name:<12} {n:>7} {fmt(p50):>8} {fmt(mx):>8}")
    sys.exit(1 if r["lost"] or r["extra"] or r["duplicates"] or no_truth else 0)


if __name__ == "__main__":
    main()
