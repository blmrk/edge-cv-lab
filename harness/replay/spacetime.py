"""Space-time diagram: position along the lane vs time, coloured by predicted track ID.
An identity handover shows up as one colour continuing onto the next vehicle's line.

python -m replay.spacetime --tracks iou=runs/iou.jsonl ground=runs/ground.jsonl --out spacetime.png [--band 440 560]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from .schema import read_tracks


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", nargs=2, type=float, help="x-range with no detections (occluder), shaded")
    a = ap.parse_args()

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(len(a.tracks), 1, figsize=(12, 3.4 * len(a.tracks)), sharex=True, squeeze=False)
    for ax, spec in zip(axes[:, 0], a.tracks):
        name, path = spec.split("=", 1)
        by = defaultdict(list)
        for b in read_tracks(path):
            by[b.track_id].append((b.ts_ms / 1000, (b.bbox[0] + b.bbox[2]) / 2))
        for i, (tid, pts) in enumerate(sorted(by.items())):
            ax.scatter(*zip(*pts), s=7, color=cmap(i % 20))
            ax.annotate(f"#{tid}", pts[0], fontsize=7, color=cmap(i % 20), xytext=(2, 4), textcoords="offset points")
        if a.band:
            ax.axhspan(*a.band, color="0.5", alpha=0.18, lw=0)
            ax.text(0.2, sum(a.band) / 2, "occluder: no detections", fontsize=8, va="center", color="0.3")
        ax.set_title(f"{name}: {len(by)} track IDs", loc="left", fontweight="bold")
        ax.set_ylabel("x along lane (px)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("seconds")
    fig.suptitle("Each line is one vehicle. Colour is the ID the tracker gave it.", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
