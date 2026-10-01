# Case study: zone enter/exit balance

> Status: measured offline on two public clips, one with annotated vehicle tracks. Every measured figure comes from a
> command in Reproduce. The live lab was not run for this study; statements about the live services come from their code.

## Problem

A zone counter emits enters and exits. Enters minus exits, counted up to any moment, is the counter's occupancy: how
many vehicles it says are in the zone. Every vehicle that enters eventually leaves, so over a stretch of video the two
should reconcile, and at any moment the balance should match the vehicles actually inside. A dashboard that shows a
standing imbalance is saying one of three things: vehicles really are inside, some visits were never closed, or some
visits were never real. This study splits the imbalance into those parts, scores the counter's occupancy against the
truth, and asks what a live consumer of the events can know.

## Reproducing it without hardware

- Footage and tracks: the counting and phantom studies' MTID intersection clip (1024x640, 30 fps, 3199 frames, `ffprobe`) with its
  zone (`harness/zone.json`), ByteTrack's video-run tracks on the class-agnostic detections
  (`runs/bytetrack.agnostic.jsonl`) and `greedy_iou:max_age=5` on the same detections; plus the phantom study's
  expressway clip (Vecteezy 6434705, metrics only, logged in `media/SOURCES.md`) with ByteTrack replayed on its
  detections and the band zone fixed there (`runs/gap.zone.json`). See
  [case-study-phantoms.md](case-study-phantoms.md) for how each was made.
- True occupancy: the dataset's annotated tracks (`runs/mtid.gt.jsonl`, from `scripts/mtid_to_gt.py`), counted per frame
  as the annotated vehicles whose footpoint is inside the zone, over the annotated frames 0 to 3098. Over those 3099
  frames the zone holds 1189 vehicle-frames, so a counter that always says "empty" scores a mean error of 0.384. The
  expressway clip has no annotations: there, only the balance and its parts are measured.
- The tool: `replay.balance`. For each counter it gives the balance at the end of the clip, splits the visits still open
  by their track's last box (inside at end: seen within 3 s of the end with its anchor inside; lost inside: anchor
  inside but gone for longer; left: last seen outside the zone), and scores occupancy three ways, each as the mean
  absolute error per annotated frame against the truth:
  - after the fact: every event placed at its own timestamp, as a store of events sees it once the clip is over;
  - live: every event placed when the counter emitted it, as a dashboard sees it at that moment;
  - gauge: no events at all; the counter's own committed visits whose track was seen in the last 500 ms
    (`DebouncedZoneCounter.open_visits(now, seen_within_ms=500)`; 500 ms was set before measuring).
- Counters: the rows add one change at a time, from the naive counter to the debounced one, then the phantom study's zone
  rule (`min_travel_px=30`) and this study's `enter_after_dwell`. Both new settings default to off: the edge and the sim
  run without them.

## Baseline

The naive counter (centroid, one event per crossing) on ByteTrack's tracks: 62 enters, 12 exits, a balance of 50. 46 of
its open visits are tracks whose last box sat inside the zone more than 3 s before the clip ended: a track that ends
inside the zone is a vehicle the tracker lost or gave a new ID, and the naive counter has no rule that closes a visit
whose track goes silent. Its
occupancy only climbs: at the last annotated frame it says 46 vehicles are inside, where the annotations have none, and
its mean error is 22.074.

## Fixes, one at a time

MTID, ByteTrack tracks (`replay.balance --gt`):

| counter | enters | exits | balance | open inside at end | open lost inside | open left | occupancy mae | occupancy max error | live mae | gauge mae | end counted | end true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive, centroid | 62 | 12 | 50 | 4 | 46 | 0 | 22.074 | 46 | 22.074 | None | 46 | 0 |
| naive, footpoint | 75 | 25 | 50 | 3 | 47 | 0 | 23.215 | 47 | 23.215 | None | 47 | 0 |
| debounced, no lost-track close | 60 | 6 | 54 | 2 | 42 | 10 | 24.442 | 51 | 0.384 | 0.8 | 51 | 0 |
| debounced | 39 | 36 | 3 | 2 | 0 | 1 | 0.486 | 5 | 0.384 | 0.8 | 0 | 0 |
| debounced + zone rule 30 px | 21 | 20 | 1 | 0 | 0 | 1 | 0.118 | 2 | 0.384 | 0.18 | 0 | 0 |
| debounced + zone rule 30 px + enter after dwell | 21 | 20 | 1 | 0 | 0 | 1 | 0.118 | 2 | 0.507 | 0.18 | 0 | 0 |

"end counted" and "end true" are at frame 3098, the last annotated one; the balance and the open visits are at the end
of the clip, frame 3198. `None`: the naive counter keeps no visit state to ask.

The same on `greedy_iou:max_age=5` tracks, the counting study's best tracker:

| counter | enters | exits | balance | open inside at end | open lost inside | open left | occupancy mae | occupancy max error | live mae | gauge mae | end counted | end true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive, centroid | 135 | 16 | 119 | 5 | 114 | 0 | 54.912 | 114 | 54.912 | None | 114 | 0 |
| naive, footpoint | 150 | 24 | 126 | 5 | 121 | 0 | 59.748 | 121 | 59.748 | None | 121 | 0 |
| debounced, no lost-track close | 49 | 7 | 42 | 0 | 33 | 9 | 17.952 | 41 | 0.385 | 0.386 | 41 | 0 |
| debounced | 26 | 25 | 1 | 0 | 0 | 1 | 0.176 | 3 | 0.385 | 0.386 | 0 | 0 |
| debounced + zone rule 30 px | 22 | 21 | 1 | 0 | 0 | 1 | 0.132 | 2 | 0.385 | 0.171 | 0 | 0 |
| debounced + zone rule 30 px + enter after dwell | 22 | 21 | 1 | 0 | 0 | 1 | 0.132 | 2 | 0.499 | 0.171 | 0 | 0 |

(Its live mae is 0.385, not 0.384, because it has no box at all in 7 frames, which are left out of its scoring.)

And the expressway band, no annotations, so balance and open visits only:

| counter | enters | exits | balance | open inside at end | open lost inside | open left |
|---|---|---|---|---|---|---|
| naive, centroid | 94 | 34 | 60 | 9 | 51 | 0 |
| naive, footpoint | 109 | 52 | 57 | 12 | 45 | 0 |
| debounced, no lost-track close | 73 | 19 | 54 | 9 | 41 | 4 |
| debounced | 52 | 43 | 9 | 9 | 0 | 0 |
| debounced + zone rule 30 px | 47 | 40 | 7 | 7 | 0 | 0 |
| debounced + zone rule 30 px + enter after dwell | 47 | 40 | 7 | 7 | 0 | 0 |

**Closing lost tracks does most of the work.** Hysteresis and the dwell rule without the lost-track close still leave a
balance of 54 on ByteTrack (42 visits lost inside, 10 whose track ended outside the zone before the exit committed).
Closing a visit 3 s after its track was last seen (`lost_ms`) takes it to 3, and to 1 on `greedy_iou:max_age=5`. On the
expressway band it takes 54 to 9, and every one of the 9 is a track seen inside within 3 s of the end, which is what
vehicles still in the band look like; without annotations this clip cannot confirm them.

**What is left is phantoms.** ByteTrack's 3 open visits are tracks 779 and 808, two of the phantom study's static
lane-marking tracks, still "inside" when the clip ends, and track 769, a vehicle leaving in the last frames whose exit
had not committed yet. After the fact, the debounced counter's occupancy is further from the truth than saying "empty"
(0.486 against 0.384): each phantom is a visit the counter holds open while nothing is there. The zone rule removes them:
balance 1, error 0.118.

**Live, the events carry no occupancy.** The debounced counter holds each enter back until its visit closes and then
returns it together with its exit, stamped with the time the vehicle entered (a test in `harness/tests/test_zones.py`
pins this). So a consumer reading the events as they arrive never sees a vehicle that is still inside: every debounced
row that holds its enters scores 0.384 live, exactly what "always empty" scores. The lab's Grafana panel "Net balance: enters minus exits
(debounced)" is that consumer: it sums events in the order they arrive (`received_at`) and labels the result `in_zone`.
From the code, not from a run of the lab: each debounced enter arrives together with its exit, so the sum is back where
it was after every visit and never shows a vehicle that is still inside.

## What did not work

- **Releasing the enter once dwell is proven** (`enter_after_dwell`). It is what the counter's own comment claimed, and
  it changes when events are returned, not their timestamps, so every figure after the fact is unchanged. Live, the
  enter now arrives once the visit has lasted `min_dwell_ms` (1 s), but the exit arrives late: on ByteTrack's tracks 14
  of the 20 exits are emitted 3 s or more after their stamp, which is the lost-track rule closing visits whose track
  stopped being seen before its exit committed (8 frames outside the zone). The live error goes up, from 0.384 to 0.507:
  485 vehicle-frames short of the truth and 1085 beyond it. An event can only say a vehicle left once the counter is
  sure, and here it is mostly sure 3 s late.
- **The gauge without the zone rule.** Counting the counter's own visits seen in the last 500 ms does not wait for any
  event, but a phantom is seen in every frame: 0.8 on ByteTrack, worse than "always empty". With the zone rule it is 0.18
  on ByteTrack and 0.171 on `greedy_iou:max_age=5`, the closest live view to the truth here; the two fixes need each other.

## What the live services would need

Not changed in this study (the services keep every default). From the measurements above:

- Publish occupancy as a gauge from the counter's state (`open_visits(now, seen_within_ms=500)`) with the zone rule on,
  instead of deriving it from events; label the event sum as what it is, a count of closed visits.
- The edge calls `counter.update()` only for frames that have track boxes, so `expire()` runs only when some box
  arrives: a visit whose track is lost stays open until the next vehicle is detected. On these clips that never mattered
  (every ByteTrack frame has a box, and `greedy_iou:max_age=5` lacks one in 7 frames), but on a quiet camera it would.
  The gauge takes the current time, so it would not depend on boxes arriving.

## Limitations

- One annotated clip, 3199 frames at 30 fps. True occupancy uses the annotated boxes' footpoint, the same anchor as the counters: a
  vehicle the annotations miss, or one whose footpoint sits on the wrong side of the zone edge on this oblique view (see
  the counting study's "What did not work"), counts as an error either way.
- The live and gauge views are replayed offline from the saved tracks, not measured on the running lab or its dashboard.
- The expressway clip has no annotations, so its open visits are counted, not checked.
- The zone rule's 30 px, the 3 s `lost_ms` and the 500 ms gauge window were set before measuring; none was tuned here.

## Reproduce

```bash
setopt interactive_comments 2>/dev/null || true
# inputs: the phantom study's Reproduce makes runs/bytetrack.agnostic.jsonl, runs/greedy_iou_5.agnostic.jsonl,
# runs/gap.bytetrack.jsonl and runs/gap.zone.json; the counting study's makes runs/mtid.gt.jsonl
pip install -e "harness[dev]"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames media/sample.mp4
cd harness
python -m replay.balance --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --gt runs/mtid.gt.jsonl --explain
python -m replay.balance --tracks runs/greedy_iou_5.agnostic.jsonl --zone zone.json --gt runs/mtid.gt.jsonl
python -m replay.balance --tracks runs/gap.bytetrack.jsonl --zone runs/gap.zone.json
# frames that have any track box, per run
python -c "
from replay.schema import read_tracks
for path, n in (('runs/bytetrack.agnostic.jsonl', 3199), ('runs/greedy_iou_5.agnostic.jsonl', 3199), ('runs/gap.bytetrack.jsonl', 1500)):
    print(path, 'frames with a track box:', len({b.frame for b in read_tracks(path)}), 'of', n)"
# the truth's size, and why releasing the enter early does worse live: late exits
python -c "
import json
from replay import balance; from replay.schema import read_tracks; from replay.zones import DebouncedZoneCounter
poly = [tuple(p) for p in json.load(open('zone.json'))['polygon']]
truth = balance.true_occupancy(read_tracks('runs/mtid.gt.jsonl'), poly); lo, hi = min(truth), max(truth)
boxes = sorted(read_tracks('runs/bytetrack.agnostic.jsonl'), key=lambda b: (b.frame, b.track_id))
frame_ts = sorted({(b.frame, b.ts_ms) for b in boxes if lo <= b.frame <= hi})
print('annotated frames', len(frame_ts), '| vehicle-frames inside', sum(truth.get(f, 0) for f, _ in frame_ts), '| error of always 0:', round(sum(truth.get(f, 0) for f, _ in frame_ts) / len(frame_ts), 3))
for kw in (dict(min_travel_px=30), dict(min_travel_px=30, enter_after_dwell=True)):
    live = balance.emitted(DebouncedZoneCounter(poly, **kw), boxes)
    occ = balance.occupancy(live, frame_ts)
    over = sum(max(0, occ[f] - truth.get(f, 0)) for f, _ in frame_ts); under = sum(max(0, truth.get(f, 0) - occ[f]) for f, _ in frame_ts)
    lag = [at - e.ts_ms for at, e in live if e.kind == 'exit']
    print(kw, '| live over', over, 'under', under, '| exits', len(lag), 'emitted 3 s or more after their stamp', sum(x >= 3000 for x in lag))"
# the phantom study's --explain lists tracks 779 and 808 among the static enters
python -m replay.score --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --truth truth.json --explain
```
