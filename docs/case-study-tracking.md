# Case study: counting vehicles once

> Status: in progress. Numbers marked `TBD` are filled from real runs. Do not publish estimates.

## Problem

Zone-based vehicle analytics report inflated visit counts and wrong dwell times. Single-frame detection accuracy is high, yet totals drift. The errors come from what happens *after* detection.

## Reproducing it without hardware

- Footage: MTID (Multi-View Traffic Intersection Dataset) infrastructure camera: a fixed, pole-mounted camera looking
  obliquely down on a signalised intersection, daytime, overcast. CC BY 4.0, M. B. Jensen, A. Møgelmose, T. B. Moeslund
  (Aalborg University). 1024x640 at 30 fps, 3199 frames, 106.63 s (`ffprobe media/sample.mp4`). Source, build command and
  citation in `media/SOURCES.md`.
- Ground truth: 14 visits in the zone drawn in `tools/label.html` (`harness/zone.json`). Two independent visual passes over
  0.5 s contact sheets found 14 and 12 (pass-to-pass difference 2); a third pass derived from the dataset's annotated tracks
  found 20 for frames 1-3099, counting vehicles that run along the zone edge. A visit counts when at least two of the three
  passes include it. Clip logged in `media/SOURCES.md`.
- Detections and tracks: `yolov8n` on CPU inside the edge image. `dump_detections.py` feeds the harness trackers and
  `dump_tracks.py` gives ByteTrack's own tracks, saved under `harness/runs/` (gitignored; the Reproduce commands regenerate them).
  The baseline and the steps up to the tracker swap use the detector's default per-class non-maximum suppression
  (`--per-class-nms`); the last step switches it to class-agnostic.
- Zone: polygon in `harness/zone.json`, drawn in `tools/label.html` over the crossing lanes; the edge uses the same polygon
  (`ZONE_POLYGON` in `docker-compose.yml`)

## Baseline

| Metric | Naive counter + ByteTrack | Debounced counter + ByteTrack |
|---|---|---|
| Enter events vs ground truth (14) | 78 (+457.1%) | 39 (+178.6%) |
| Precision / recall / F1 of visits (`replay.score`, 2 s tolerance) | 0.167 / 0.929 / 0.283 | 0.359 / 1.0 / 0.528 |
| Tracks with more than one enter | 0 | 0 |
| Net balance (enters minus exits) | 66 | 3 |
| ID switches (`replay.compare --gt`) / IDF1 / HOTA (TrackEval) | 110 / TBD / TBD | 110 / TBD / TBD |

ID switches are counted against ground-truth tracks built from the dataset's own annotations (`mtid_to_gt.py`), on the
20.5% of ground-truth boxes a ByteTrack box overlaps at IoU 0.5 or more. Coverage is low: much of the annotated traffic runs
along the top of the frame, partly under the burned-in timestamp band, where few detections land.

No track re-enters the zone, so every extra visit is a new track ID for a vehicle already counted: on this footage the
tracker, not the zone logic, drives the error. The debounced counter still removes most of the naive counter's excess.

Add a GIF of the worst offender here. One clip of a box flickering on a boundary explains more than a table.

## Fixes, one at a time

| Change | Enter error | Notes |
|---|---|---|
| Baseline | +457.1% (78 vs 14) | naive centroid counter on ByteTrack tracks |
| + footpoint anchor | +600.0% (98) | worse on this view (see What did not work) |
| + hysteresis (5 in / 8 out) | +392.9% (69) | |
| + edge margin (10 px) | +364.3% (65) | parked on the zone edge |
| + min dwell 1 s, cooldown 1.5 s | +178.6% (39) | largest single step: fragment tracks shorter than a second no longer count |
| + tracker swap (see docs/trackers.md): `replay.compare` row per tracker | table below | best here: `greedy_iou:max_age=5`, 21 vs 14 |
| + class-agnostic NMS (detector) | +85.7% (26) | `greedy_iou:max_age=5`: F1 0.686 to 0.7, missed visits 2 to 0, ID switches 368 to 68 |

Report each step separately. An ablation is more convincing than one before/after pair.

The counter rows above are ByteTrack tracks, steps added cumulatively (`replay.ablation`, command in Reproduce):

| step | enters | enter error pct | matched | false visits | missed visits | f1 |
|---|---|---|---|---|---|---|
| baseline (naive, centroid) | 78 | +457.1% | 13 | 65 | 1 | 0.283 |
| + footpoint anchor | 98 | +600.0% | 14 | 84 | 0 | 0.25 |
| + hysteresis (5 in / 8 out) | 69 | +392.9% | 14 | 55 | 0 | 0.337 |
| + edge margin (10 px) | 65 | +364.3% | 14 | 51 | 0 | 0.354 |
| + min dwell 1 s, cooldown 1.5 s | 39 | +178.6% | 14 | 25 | 0 | 0.528 |

The same steps on `greedy_iou:max_age=5` tracks, the best tracker below, which splits vehicles into 383 track IDs:

| step | enters | enter error pct | matched | false visits | missed visits | f1 |
|---|---|---|---|---|---|---|
| baseline (naive, centroid) | 171 | +1121.4% | 14 | 157 | 0 | 0.151 |
| + footpoint anchor | 194 | +1285.7% | 14 | 180 | 0 | 0.135 |
| + hysteresis (5 in / 8 out) | 83 | +492.9% | 14 | 69 | 0 | 0.289 |
| + edge margin (10 px) | 79 | +464.3% | 14 | 65 | 0 | 0.301 |
| + min dwell 1 s, cooldown 1.5 s | 21 | +50.0% | 12 | 9 | 2 | 0.686 |

Its many fragments are short, so the minimum dwell removes most of them; that is why it ends up best.

Tracker swap on the same detections, debounced counter, identity against the annotated tracks (`replay.compare --gt`,
commands in Reproduce):

| tracker | labelled | predicted | missed visits | false visits | f1 | id switches | id transfers | pred ids | gt objects | gt boxes matched pct |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy_iou:max_age=5 | 14 | 21 | 2 | 9 | 0.686 | 368 | 0 | 383 | 65 | 18.2 |
| greedy_iou | 14 | 31 | 1 | 18 | 0.578 | 347 | 3 | 273 | 65 | 18.2 |
| groundplane | 14 | 38 | 0 | 24 | 0.538 | 349 | 40 | 148 | 65 | 18.2 |
| bytetrack | 14 | 39 | 0 | 25 | 0.528 | 110 | 0 | 194 | 65 | 20.5 |

The two rankings disagree. ByteTrack keeps identities best (a third of the switches) yet counts visits worst; the
short-buffer IoU tracker counts best only because its fragments are too short to pass the minimum dwell. Counting
accuracy here comes from the counter's rules absorbing fragmentation, not from better tracking.

### Detector fix: class-agnostic NMS

Non-maximum suppression runs per class by default, so one vehicle scored as both car and truck keeps two boxes and a
tracker follows each. With class-agnostic NMS the detector keeps the higher-scoring box: 20151 boxes instead of 21902 over the
clip (`wc -l`), everything else unchanged. Same table on the new detections:

| tracker | labelled | predicted | missed visits | false visits | f1 | id switches | id transfers | pred ids | gt objects | gt boxes matched pct |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy_iou:max_age=5 | 14 | 26 | 0 | 12 | 0.7 | 68 | 0 | 272 | 65 | 18.2 |
| greedy_iou | 14 | 31 | 0 | 17 | 0.622 | 46 | 1 | 194 | 65 | 18.2 |
| groundplane | 14 | 32 | 0 | 18 | 0.609 | 44 | 22 | 114 | 65 | 18.2 |
| bytetrack | 14 | 39 | 0 | 25 | 0.528 | 11 | 0 | 117 | 65 | 20.7 |

The NMS mode is the only change, so the drop comes from the cross-class overlapping boxes it removes: ID switches fall
from 368, 347, 349 and 110 to 68, 46, 44 and 11. Visit F1 rises for the three harness trackers and none of the four
misses a visit. The best tracker's count gets worse, 26 against 21, because a count nets false visits against
missed ones: 21 was 12 matched and 9 false with 2 missed, 26 is 14 matched and 12 false. ByteTrack's debounced count
does not move (39, F1 0.528); its naive count drops from 78 to 62 and its net balance from 66 to 50. The fix shows up
in identity far more than in counting, and the rankings still disagree: ByteTrack switches least and counts worst.

## Synthetic confirmation

| Fixture | Truth | Naive | Debounced |
|---|---|---|---|
| Boundary jitter, 10 s | 1 | 66 | 1 |
| Shadow into adjacent lane | 0 | 1 | 0 |

Reproduce: `cd harness && python -m replay.cli --tracks fixtures/boundary_jitter.jsonl --zone fixtures/zone.json --expected 1`,
then the same with `--tracks fixtures/shadow_expansion.jsonl --expected 0`.

## What did not work

- The footpoint anchor made counts worse on this view: +600.0% against +457.1% for the centroid on ByteTrack tracks,
  and +1285.7% against +1121.4% on `greedy_iou:max_age=5`. On the synthetic shadow fixture it is what removes the false
  visit. Diagnosis: every extra enter is on the zone's two far edges, from traffic on the far road and in the lane just
  beyond them. On this oblique view a box's bottom-centre is not the vehicle's ground contact: it sits below the vehicle,
  towards the camera, so vehicles running just outside the far edges dip in, while the centroid sits too high. It is not
  box-height flicker, and not the duplicate boxes: with class-agnostic NMS the footpoint step is still worse, +435.7%
  against +342.9% on ByteTrack and +971.4% against +864.3% on `greedy_iou:max_age=5`. A ground-contact point from a
  road-plane mapping would fix the anchor properly; tuning the anchor height on 14 visits would fit noise.
- `groundplane`, the tracker that fixes the synthetic queue, does not help at this intersection: F1 0.538, and 0.56 with
  its lane direction set along the zone (`--param 'lane_dir=[0.85,0.53]'`); 0.609 and 0.596 with class-agnostic NMS,
  still behind the IoU trackers. One lane direction cannot describe turning
  traffic from several approaches, and its gates were tuned on the synthetic queue at 10 fps.

Keep this section. Parameters that over-suppressed real short visits, cases where footpoint was worse, and so on.

## Limitations

Single camera angle, daytime footage, small ground-truth set, CPU-only model.

## Reproduce

```bash
pip install -e "harness[dev]"   # scoring; detection runs in the edge image that `make up-video` builds
docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge \
  python scripts/dump_detections.py --video ../media/sample.mp4 --model /app/yolov8n.pt --per-class-nms --out runs/dets.jsonl
docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge \
  python scripts/dump_tracks.py --video ../media/sample.mp4 --model /app/yolov8n.pt --per-class-nms --tracker bytetrack.yaml --out runs/bytetrack.jsonl
# class-agnostic NMS, the scripts' default
docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge \
  python scripts/dump_detections.py --video ../media/sample.mp4 --model /app/yolov8n.pt --out runs/dets.agnostic.jsonl
docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge \
  python scripts/dump_tracks.py --video ../media/sample.mp4 --model /app/yolov8n.pt --tracker bytetrack.yaml --out runs/bytetrack.agnostic.jsonl
cd harness
python -m replay.score --tracks runs/bytetrack.jsonl --zone zone.json --truth truth.json
python -m replay.cli --tracks runs/bytetrack.jsonl --zone zone.json --expected 14
python -m replay.ablation --tracks runs/bytetrack.jsonl --zone zone.json --truth truth.json
python -m replay.track --dets runs/dets.jsonl --tracker greedy_iou --param max_age=5 --out runs/greedy_iou_5.jsonl
python -m replay.ablation --tracks runs/greedy_iou_5.jsonl --zone zone.json --truth truth.json
python scripts/mtid_to_gt.py --annotations ../media/candidates/mtid/annotations/Infrastructure --out runs/mtid
python -m replay.compare --dets runs/dets.jsonl --zone zone.json --truth truth.json --gt runs/mtid.gt.jsonl \
    --trackers greedy_iou:max_age=5 greedy_iou groundplane --tracks bytetrack=runs/bytetrack.jsonl
python -m replay.track --dets runs/dets.jsonl --tracker groundplane --param 'lane_dir=[0.85,0.53]' --out runs/groundplane_diag.jsonl
python -m replay.compare --dets runs/dets.jsonl --zone zone.json --truth truth.json --tracks groundplane_diagonal=runs/groundplane_diag.jsonl
# class-agnostic NMS
wc -l runs/dets.jsonl runs/dets.agnostic.jsonl
python -m replay.cli --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --expected 14
python -m replay.ablation --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --truth truth.json
python -m replay.track --dets runs/dets.agnostic.jsonl --tracker greedy_iou --param max_age=5 --out runs/greedy_iou_5.agnostic.jsonl
python -m replay.ablation --tracks runs/greedy_iou_5.agnostic.jsonl --zone zone.json --truth truth.json
python -m replay.compare --dets runs/dets.agnostic.jsonl --zone zone.json --truth truth.json --gt runs/mtid.gt.jsonl \
    --trackers greedy_iou:max_age=5 greedy_iou groundplane --tracks bytetrack=runs/bytetrack.agnostic.jsonl
python -m replay.track --dets runs/dets.agnostic.jsonl --tracker groundplane --param 'lane_dir=[0.85,0.53]' --out runs/groundplane_diag.agnostic.jsonl
python -m replay.compare --dets runs/dets.agnostic.jsonl --zone zone.json --truth truth.json --tracks groundplane_diagonal=runs/groundplane_diag.agnostic.jsonl
cd .. && make footage   # docs/footage/real-compare.gif
```
