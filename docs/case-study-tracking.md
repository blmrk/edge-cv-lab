# Case study: counting vehicles once

> Status: template. Numbers marked `TBD` are filled from real runs. Do not publish estimates.

## Problem

Zone-based vehicle analytics report inflated visit counts and wrong dwell times. Single-frame detection accuracy is high, yet totals drift. The errors come from what happens *after* detection.

## Reproducing it without hardware

- Footage: `TBD` (source, licence, duration, resolution, camera angle)
- Ground truth: `TBD` visits, labelled with `tools/label.html` at 2x, `TBD` passes. Logged in `media/SOURCES.md`.
- Tracks: `dump_tracks.py` with `yolov8n`, run once per tracker and saved under `harness/runs/` (gitignored; regenerate each with `dump_tracks.py` as in Reproduce below)
- Zone: polygon in `TBD.json`, screenshot below

## Baseline

| Metric | Naive counter + ByteTrack |
|---|---|
| Enter events vs ground truth | TBD |
| Precision / recall / F1 of visits (`replay.score`, 2 s tolerance) | TBD |
| Tracks with more than one enter | TBD |
| Net balance (enters minus exits) | TBD |
| ID switches / IDF1 / HOTA (TrackEval) | TBD |

Add a GIF of the worst offender here. One clip of a box flickering on a boundary explains more than a table.

## Fixes, one at a time

| Change | Enter error | Notes |
|---|---|---|
| Baseline | TBD | |
| + footpoint anchor | TBD | |
| + hysteresis (5 in / 8 out) | TBD | |
| + edge margin (10 px) | TBD | parked on the zone edge |
| + min dwell 1 s, cooldown 1.5 s | TBD | |
| + tracker swap (see docs/trackers.md): `replay.compare` row per tracker | TBD | identity handover impact |

Report each step separately. An ablation is more convincing than one before/after pair.

## Synthetic confirmation

| Fixture | Truth | Naive | Debounced |
|---|---|---|---|
| Boundary jitter, 10 s | 1 | 66 | 1 |
| Shadow into adjacent lane | 0 | 1 | 0 |

Reproduce: `cd harness && python -m replay.cli --tracks fixtures/boundary_jitter.jsonl --zone fixtures/zone.json --expected 1`,
then the same with `--tracks fixtures/shadow_expansion.jsonl --expected 0`.

## What did not work

TBD. Keep this section. Parameters that over-suppressed real short visits, cases where footpoint was worse, and so on.

## Limitations

Single camera angle, daytime footage, small ground-truth set, CPU-only model.

## Reproduce

```bash
pip install -e "harness[dev,video]"
python harness/scripts/dump_tracks.py --video media/sample.mp4 --out harness/runs/bytetrack.jsonl
python -m replay.score --tracks harness/runs/bytetrack.jsonl --zone zone.json --truth truth.json
```
