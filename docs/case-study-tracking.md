# Case study: counting vehicles once

> Status: measured on one public clip. Every measured figure comes from a command in Reproduce; the labelling passes
> behind the ground truth are described under Ground truth.

## Problem

Zone-based vehicle analytics report inflated visit counts and wrong dwell times, even when the detector looks right frame by frame. The errors come from what happens *after* detection, and from weak detections that tracking turns into objects.

## Reproducing it without hardware

- Footage: MTID (Multi-View Traffic Intersection Dataset) infrastructure camera: a fixed, pole-mounted camera looking
  obliquely down on a signalised intersection, daytime, overcast. CC BY 4.0, M. B. Jensen, A. Møgelmose, T. B. Moeslund
  (Aalborg University). 1024x640 at 30 fps, 3199 frames, 106.63 s (`ffprobe media/sample.mp4`). Source, build command and
  citation in `media/SOURCES.md`.
- Ground truth: 14 visits in the zone drawn in `tools/label.html` (`harness/zone.json`). Two model-assisted visual passes
  (an AI model reading 0.5 s contact sheets, each pass run separately) found 14 and 12 (pass-to-pass difference 2); a third pass derived from the dataset's annotated tracks
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
| ID switches (`replay.compare --gt`) / IDF1 / HOTA (TrackEval) | 110 / 0.263 / 0.221 | 110 / 0.263 / 0.221 |

ID switches are counted against ground-truth tracks built from the dataset's own annotations (`mtid_to_gt.py`), on the
20.5% of ground-truth boxes a ByteTrack box overlaps at IoU 0.5 or more. Coverage is low: much of the annotated traffic runs
along the top of the frame, partly under the burned-in timestamp band, where few detections land. IDF1 and HOTA come
from TrackEval on the same ground truth (tables below); the counter does not change the tracks, so both columns match.

No track re-enters the zone. `replay.score --explain` lists each of the debounced counter's 39 enters with how far its
track's footpoint ever moved, and they fall in two groups with nothing between: 18 tracks moved at most 6 px, with boxes
22 to 34 px wide and 14 to 24 px tall at mean scores 0.17 to 0.31, and 21 moved at least 355 px. Drawn on a frame, the
first group sits on the white dashes of the bike lane that crosses the zone: the detector scores lane markings as cars
at low confidence, and ByteTrack keeps them as parked tracks (Ultralytics' defaults start a track at score 0.25 and keep
it on boxes down to 0.1). Scored on their own, the 21 moving enters match all 14 labelled visits with 7 false (F1 0.8,
against 0.528 with the phantoms). So 18 of ByteTrack's 25 extra visits are phantoms, and 7 are moving enters with no
labelled visit left to match: vehicles the labels do not count, or a second vehicle entering close to one that took
the label. Phantom boxes get their own case study, [case-study-phantoms.md](case-study-phantoms.md), where a zone rule
removes all 18; this one keeps them. The debounced counter still removes most of the naive counter's excess.

The GIF in the README (`docs/footage/real-compare.gif`, `make footage`) shows two of the trackers below over 13 s of the
clip from 20 s, IDs as coloured tags.

## Fixes, one at a time

| Change | Enter error | Notes |
|---|---|---|
| Baseline | +457.1% (78 vs 14) | naive centroid counter on ByteTrack tracks |
| + footpoint anchor | +600.0% (98) | worse on this view (see What did not work) |
| + hysteresis (5 in / 8 out) | +392.9% (69) | |
| + edge margin (10 px) | +364.3% (65) | parked on the zone edge |
| + min dwell 1 s, cooldown 1.5 s | +178.6% (39) | largest single step: fragment tracks shorter than a second no longer count |
| + tracker swap (see docs/trackers.md): `replay.compare` row per tracker | table below | best count here: `greedy_iou:max_age=5`, 21 vs 14, mostly from its 0.3 score floor (below) |
| + class-agnostic NMS (detector) | +85.7% (26) | `greedy_iou:max_age=5`: count 21 to 26, worse (see below), but F1 0.686 to 0.7, missed visits 2 to 0, ID switches 368 to 68 |

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

Its many fragments are short, so the minimum dwell removes most of them, and its 0.3 score floor keeps most lane-marking
phantoms out (below); that is why it ends up best.

Tracker swap on the same detections, debounced counter, identity against the annotated tracks (`replay.compare --gt`,
commands in Reproduce). The harness trackers drop boxes scored under 0.3 before association (`min_score`); ByteTrack
uses boxes down to 0.1:

| tracker | labelled | predicted | missed visits | false visits | f1 | id switches | id transfers | pred ids | gt objects | gt boxes matched pct |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy_iou:max_age=5 | 14 | 21 | 2 | 9 | 0.686 | 368 | 0 | 383 | 65 | 18.2 |
| greedy_iou | 14 | 31 | 1 | 18 | 0.578 | 347 | 3 | 273 | 65 | 18.2 |
| groundplane | 14 | 38 | 0 | 24 | 0.538 | 349 | 40 | 148 | 65 | 18.2 |
| bytetrack | 14 | 39 | 0 | 25 | 0.528 | 110 | 0 | 194 | 65 | 20.5 |

The same tracks scored by TrackEval (`scripts/trackeval_run.py`, TrackEval 12c8791; HOTA, DetA and AssA are means over
IoU 0.05 to 0.95, the rest at IoU 0.5; only the annotated frames, 1 to 3099, are scored):

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| greedy_iou:max_age=5 | 0.155 | 0.135 | 0.179 | 0.167 | -0.081 | 205 |
| greedy_iou | 0.161 | 0.135 | 0.193 | 0.178 | -0.080 | 184 |
| groundplane | 0.182 | 0.135 | 0.246 | 0.217 | -0.077 | 127 |
| bytetrack | 0.221 | 0.141 | 0.347 | 0.263 | -0.100 | 16 |

The absolute scores are low because detection is: DetA 0.135 to 0.141 against the annotated vehicles, and MOTA is
negative, so misses, false positives and switches together outnumber the ground-truth boxes. The two switch counts differ by definition. The harness counts a
switch whenever the per-frame greedy match of a ground-truth vehicle changes predicted ID, so two overlapping boxes on
one vehicle can alternate and count one each time; TrackEval's CLEAR matching gives the previous pairing priority, so
it counts fewer (16 against 110 for ByteTrack). Both rank ByteTrack first on identity.

The two rankings disagree, and `--explain` says why. ByteTrack keeps identities best yet counts worst because it keeps
the lane-marking phantoms: 18 of its 39 enters, against 4 of 21 for `greedy_iou:max_age=5`. That gap is the score
floor: 14090 of the 21902 detections score under 0.3, and with `min_score=0.1` the same tracker logs 121 enters, 95
of them static. On moving enters alone the two are close: F1 0.8 for ByteTrack (21 enters, 14 matched, 7 false)
against 0.774 (17 enters, 12 matched, 5 false, 2 missed). The tracker swap mostly swaps the score floor.

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
does not move (39, F1 0.528), and neither do its 18 phantom enters; its naive count drops from 78 to 62 and its net
balance from 66 to 50. TrackEval agrees on the direction:

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| greedy_iou:max_age=5 | 0.202 | 0.138 | 0.296 | 0.227 | -0.048 | 66 |
| greedy_iou | 0.206 | 0.138 | 0.309 | 0.238 | -0.046 | 44 |
| groundplane | 0.207 | 0.137 | 0.312 | 0.244 | -0.046 | 38 |
| bytetrack | 0.224 | 0.143 | 0.350 | 0.268 | -0.082 | 9 |

Its switch counts fall from 205, 184, 127 and 16 to 66, 44, 38 and 9, and IDF1 and HOTA rise for every tracker, most
for the harness trackers. The fix shows up in identity far more than in counting, and the rankings still disagree:
ByteTrack switches least and counts worst.

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
  box-height flicker, and the duplicate boxes are not why: removing them lowers both anchors' counts, and with
  class-agnostic NMS the footpoint step is still worse, +435.7% against +342.9% on ByteTrack and +971.4% against
  +864.3% on `greedy_iou:max_age=5`. A ground-contact point from a
  road-plane mapping would fix the anchor properly; tuning the anchor height on 14 visits would fit noise.
- `groundplane`, the tracker that fixes the synthetic queue, does not help at this intersection: F1 0.538, and 0.56 with
  its lane direction set along the zone (`--param 'lane_dir=[0.85,0.53]'`); 0.609 and 0.596 with class-agnostic NMS,
  still behind the IoU trackers. One lane direction cannot describe turning
  traffic from several approaches, and its gates were tuned on the synthetic queue at 10 fps.

## Limitations

- One clip: a single camera angle, daytime, 106.6 s, 14 labelled visits. Every figure here is from that clip.
- The visit ground truth rests on three passes, two of them by an AI model reading contact sheets; the third comes from
  the dataset's own annotations. They found 14, 12 and 20 visits, and a visit counts when two of three include it.
- Visits are matched on enter time alone (2 s tolerance), so an enter from the wrong object can take a labelled visit's
  match and hide a miss: at `min_score=0.1` the 1 s-buffer tracker scores recall 1.0 with its phantoms and 12 of 14
  on its moving enters alone. `--explain` scores the moving enters on their own for that reason.
- Identity metrics only see the annotated vehicles the detector finds: 18 to 21% of ground-truth boxes are matched, and
  DetA is 0.135 to 0.143.
- The harness trackers and ByteTrack use different score floors (0.3 and 0.1), so the tracker comparison mixes
  association with detection filtering.
- CPU-only `yolov8n` at `conf` 0.1; a larger model or a higher threshold would change the detections, and every number
  after them.

## Reproduce

```bash
setopt interactive_comments 2>/dev/null || true
pip install -e "harness[dev,trackeval]"   # scoring; detection runs in the edge image that `make up-video` builds
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
python -m replay.score --tracks runs/bytetrack.jsonl --zone zone.json --truth truth.json --explain
python -m replay.score --tracks runs/greedy_iou_5.jsonl --zone zone.json --truth truth.json --explain
python -c "from replay.detections import read_detections as r; d = list(r('runs/dets.jsonl')); print(len(d), sum(x.score < 0.3 for x in d))"
python -m replay.track --dets runs/dets.jsonl --tracker greedy_iou --param max_age=5 --param min_score=0.1 --out runs/greedy_iou_5_s01.jsonl
python -m replay.score --tracks runs/greedy_iou_5_s01.jsonl --zone zone.json --truth truth.json --explain
# TrackEval, cloned once at the pinned commit (MIT); the runner reads it from ../.cache/TrackEval
git clone https://github.com/JonathonLuiten/TrackEval ../.cache/TrackEval && git -C ../.cache/TrackEval checkout 12c8791
python -m replay.track --dets runs/dets.jsonl --tracker greedy_iou --out runs/greedy_iou.jsonl
python -m replay.track --dets runs/dets.jsonl --tracker groundplane --out runs/groundplane.jsonl
python scripts/trackeval_run.py --gt runs/mtid.gt.jsonl --tracks greedy_iou:max_age=5=runs/greedy_iou_5.jsonl \
    greedy_iou=runs/greedy_iou.jsonl groundplane=runs/groundplane.jsonl bytetrack=runs/bytetrack.jsonl
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
python -m replay.score --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --truth truth.json --explain
python -m replay.score --tracks runs/greedy_iou_5.agnostic.jsonl --zone zone.json --truth truth.json --explain
python -m replay.track --dets runs/dets.agnostic.jsonl --tracker greedy_iou --out runs/greedy_iou.agnostic.jsonl
python -m replay.track --dets runs/dets.agnostic.jsonl --tracker groundplane --out runs/groundplane.agnostic.jsonl
python scripts/trackeval_run.py --gt runs/mtid.gt.jsonl --tracks greedy_iou:max_age=5=runs/greedy_iou_5.agnostic.jsonl \
    greedy_iou=runs/greedy_iou.agnostic.jsonl groundplane=runs/groundplane.agnostic.jsonl bytetrack=runs/bytetrack.agnostic.jsonl
cd .. && make footage   # docs/footage/real-compare.gif
```
