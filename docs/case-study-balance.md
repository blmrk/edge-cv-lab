# Case study: zone enter/exit balance

> Status: measured offline on two public clips, one with annotated vehicle tracks. Every measured figure comes from a
> command in Reproduce. The live lab was run only to check the opt-in gauge (In the live services); other statements about
> the services come from their code.

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
its open visits are tracks whose last box sat inside the zone more than 3 s before the clip ended, and the naive counter
has no rule that closes a visit whose track goes silent. 36 of the 46 never moved 30 px: the phantom study's static
lane-marking tracks, dropped by the tracker. The other 10 are moving vehicles the tracker lost or gave a new ID inside the
zone. Its occupancy only climbs: at the last annotated frame it says 46 vehicles are inside, where the annotations have none, and
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
balance of 54 on ByteTrack: 42 visits lost inside (34 of them static tracks), 10 whose track ended outside the zone
before the exit committed, and 2 still inside at the end.
Closing a visit 3 s after its track was last seen (`lost_ms`) takes it to 3, and to 1 on `greedy_iou:max_age=5`. On the
expressway band it takes 54 to 9. Those 9 were all seen inside within 3 s of the end, but only 4 in the last frame; the
other 5 were last seen 1.18 to 2.6 s before it and are open only because `lost_ms` has not run out. The zone rule leaves
7. Without annotations this clip cannot say which are vehicles still in the band.

**What is left: phantoms, and a vehicle leaving.** ByteTrack's 3 open visits at the end of the clip are tracks 779 and
808, two of the phantom study's static lane-marking tracks, and track 769, a vehicle leaving in the last frames whose
exit had not committed yet. All three enter after frame 3098, the last annotated one, so they do not touch the occupancy
scores. Inside the annotated frames the debounced counter commits 16 other static tracks, each a visit held open while
nothing is there: after the fact its occupancy is 1415 vehicle-frames above the truth and 91 below, further from the
truth than saying "empty" (0.486 against 0.384). The zone rule removes the static enters: 146 above and 220 below, error
0.118, and a balance of 1 at the end, track 769.

**Live, the events carry no occupancy.** The debounced counter holds each enter back until its visit closes and then
returns it together with its exit, stamped with the time the vehicle entered (a test in `harness/tests/test_zones.py`
pins this). So a consumer reading the events as they arrive never sees a vehicle that is still inside: every debounced
row that holds its enters scores 0.384 live, exactly what "always empty" scores. The lab's Grafana panel "Net balance: enters minus exits
(debounced)" was that consumer: it sums events in the order they arrive (`received_at`) and labelled the result `in_zone`
(relabelled since; see In the live services).
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

## In the live services (opt-in)

The edge and the sim keep every default. Two flags, both 0 (off) unless set, turn on the pair measured above:

- `OCCUPANCY_GAUGE_MS=500` (edge and sim): the device publishes `{device_id, counter, in_zone, ts_ms}` to
  `occupancy/<device>` once a second, from `OccupancyGauge` (`harness/replay/gauge.py`, the gauge view above). QoS 0: a
  stale sample is not worth queueing through an outage, so samples sent while the uplink is down are dropped. Ingest
  stores them in `zone_occupancy` and Grafana plots them in "Vehicles in zone (gauge, opt-in)", at each sample's device
  time rather than its arrival. The edge samples on
  every frame, detections or not, so a track that goes quiet drops out without waiting for the next box.
- `MIN_TRAVEL_PX=30` (edge only): the zone rule, without which the gauge counts phantoms. The sim's synthetic traffic
  has none.

The panel that summed events is now "Enters minus exits, as received (debounced)", series `enters_minus_exits`. With
the default counter it is back where it was after every visit, so a step that stays means half of a visit arrived. It
is not a delivery check: enter and exit travel together, so a visit lost on the way, both messages, leaves it flat.
`make delivery` checks delivery.

Checked on the sim lab with the gauge on (commands in Reproduce): of seed 11's 131 samples, 130 reached the database,
each equal to the offline replay of the same scene through `OccupancyGauge` at the same timestamp; the missing one is
the scene's first, sent while the lab was starting. A QoS 0 sample is dropped unless the device is connected and ingest
has subscribed, and ingest subscribes only once it has reached Postgres. `make delivery` still finds every zone event
stored once (seeds 11 and 12: 610 expected, 610 stored, none lost, extra or duplicated).

The edge ran the gauge live on the MTID clip (`make up-video`: the clip looped through MediaMTX, YOLO and ByteTrack on
CPU), once with the zone rule and once without, about 290 s each, some 2.7 loops of the clip (commands in Reproduce).
On CPU it kept up with about half of the 30 fps stream: 15.5 and 14.7 frames a second at the end of each run
(`device_status`). The runs are not frame-aligned with the annotations, so they are compared by distribution with the
same clip's offline gauge (over all 3199 frames), not scored frame by frame:

| gauge | samples | mean vehicles in zone | share of samples above 0 | max |
|---|---|---|---|---|
| edge, live, `MIN_TRAVEL_PX=30` | 284 in 293.1 s | 0.426 | 0.356 | 3 |
| edge, live, `MIN_TRAVEL_PX=0` | 280 in 288.0 s | 0.825 | 0.464 | 10 |
| offline, `min_travel_px=30` | 3199 frames | 0.476 | 0.388 | 3 |
| offline, `min_travel_px=0` | 3199 frames | 1.135 | 0.477 | 10 |

The annotated mean is 0.384. The zone rule cuts the gauge's mean by half or more (live 0.825 to 0.426, offline 1.135 to
0.476) and takes its maximum from 10 to 3: without it, phantoms sitting in the zone count as vehicles inside. One run of
each, not repeated: live figures would vary with the frames the edge keeps up with and with which part of the clip the
partial third loop covers.

Still open: the edge calls `counter.update()` only for frames that have track boxes, so `expire()` runs only when some box
arrives, and a visit whose track is lost stays open, as events go, until the next vehicle is detected. On these clips
that never mattered (every ByteTrack frame has a box, and `greedy_iou:max_age=5` lacks one in 7 frames), but on a quiet
camera it would. The gauge reads the clock, so it does not depend on boxes arriving.

## Limitations

- One annotated clip: 3199 frames at 30 fps, 3099 of them annotated. True occupancy uses the annotated boxes' footpoint, the same anchor as the counters: a
  vehicle the annotations miss, or one whose footpoint sits on the wrong side of the zone edge on this oblique view (see
  the counting study's "What did not work"), counts as an error either way.
- The live and gauge views are scored offline from the saved tracks. On the running lab, the sim's gauge is checked
  against its offline replay and the edge's live gauge is compared by distribution only, not scored against the
  annotations; the dashboard itself is not measured.
- The expressway clip has no annotations, so its open visits are counted, not checked.
- The zone rule's 30 px, the 3 s `lost_ms` and the 500 ms gauge window were set before measuring; none was tuned here.

## Reproduce

```bash
setopt interactive_comments 2>/dev/null || true
# inputs: the phantom study's Reproduce makes runs/bytetrack.agnostic.jsonl, runs/greedy_iou_5.agnostic.jsonl,
# runs/gap.bytetrack.jsonl and runs/gap.zone.json; the counting study's makes runs/mtid.gt.jsonl
pip install -e "harness[dev]"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames media/sample.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames media/vecteezy-6434705.mp4
cd harness
python -m replay.balance --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --gt runs/mtid.gt.jsonl --explain
python -m replay.balance --tracks runs/greedy_iou_5.agnostic.jsonl --zone zone.json --gt runs/mtid.gt.jsonl
python -m replay.balance --tracks runs/gap.bytetrack.jsonl --zone runs/gap.zone.json --explain
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
# frame ranges, greedy_iou's box-less frames, and which open visits and enters are static (moved under 30 px)
python -c "
import json, math
from collections import defaultdict
from replay import balance; from replay.schema import read_tracks; from replay.zones import DebouncedZoneCounter, NaiveZoneCounter
poly = [tuple(p) for p in json.load(open('zone.json'))['polygon']]
truth = balance.true_occupancy(read_tracks('runs/mtid.gt.jsonl'), poly); lo, hi = min(truth), max(truth)
boxes = sorted(read_tracks('runs/bytetrack.agnostic.jsonl'), key=lambda b: (b.frame, b.track_id))
by = defaultdict(list)
for b in boxes: by[b.track_id].append(b.footpoint)
travel = {t: math.hypot(max(x for x, _ in p) - min(x for x, _ in p), max(y for _, y in p) - min(y for _, y in p)) for t, p in by.items()}
print('annotated frames', lo, 'to', hi, '| last track frame', boxes[-1].frame)
g = {b.frame for b in read_tracks('runs/greedy_iou_5.agnostic.jsonl')}
print('greedy_iou_5 frames with no box:', [(f, truth.get(f, 0)) for f in range(boxes[-1].frame + 1) if f not in g])
for name, c, anchor in (('naive, centroid', NaiveZoneCounter(poly, 'centroid'), 'centroid'), ('debounced, no lost-track close', DebouncedZoneCounter(poly, lost_ms=10**12), 'footpoint')):
    ev = [e for _, e in balance.emitted(c, boxes)]
    lost = [r['track_id'] for r in balance.open_visits(ev, boxes, poly, anchor) if r['reason'] == 'lost inside']
    print(name, '| lost inside', len(lost), '| static (moved < 30 px)', sum(travel[t] < 30 for t in lost))
frame_ts = sorted({(b.frame, b.ts_ms) for b in boxes if lo <= b.frame <= hi})
for name, kw in (('debounced', {}), ('debounced + zone rule 30 px', dict(min_travel_px=30))):
    ev = [e for _, e in balance.emitted(DebouncedZoneCounter(poly, **kw), boxes)]
    occ = balance.occupancy([(e.ts_ms, e) for e in ev], frame_ts)
    over = sum(max(0, occ[f] - truth.get(f, 0)) for f, _ in frame_ts); under = sum(max(0, truth.get(f, 0) - occ[f]) for f, _ in frame_ts)
    static_in = sorted(e.track_id for e in ev if e.kind == 'enter' and travel[e.track_id] < 30 and e.frame <= hi)
    print(name, '| after the fact: over', over, 'under', under, '| static enters inside the annotated frames', len(static_in))
    print('  enter frames of 769, 779, 808:', [(e.track_id, e.frame) for e in ev if e.kind == 'enter' and e.track_id in (769, 779, 808)])"
# the phantom study's --explain lists tracks 779 and 808 among the static enters
python -m replay.score --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --truth truth.json --explain
# the opt-in gauge on the sim lab, about 5 min (seed 11 is the first scene; GRAFANA_PORT=3001 when 3000 is taken)
cd .. && make down && OCCUPANCY_GAUGE_MS=500 GRAFANA_PORT=3001 make up
i=0; until docker compose logs sim | grep -q "seed 12 done"; do i=$((i+1)); [ $i -ge 60 ] && { echo TIMEOUT; break; }; sleep 10; done
docker compose exec -T postgres psql -U postgres lab -At -F, -c "select ts_ms, in_zone from zone_occupancy order by ts_ms" > harness/runs/gauge.csv
docker compose exec -T postgres psql -U postgres lab -At -F, -c "select ts_ms from zone_events where counter='debounced' order by ts_ms limit 1" > harness/runs/events.csv
cd harness && python -c "
from replay.gauge import OccupancyGauge; from replay.synth import ZONE, by_frame, generate_traffic; from replay.zones import DebouncedZoneCounter, run
live = [tuple(map(int, l.split(','))) for l in open('runs/gauge.csv').read().split()]
first_live = min(int(l.split(',')[0]) for l in open('runs/events.csv').read().split())
boxes, _ = generate_traffic(seed=11)
first = min(run(DebouncedZoneCounter(ZONE), sorted(boxes, key=lambda b: (b.frame, b.track_id))), key=lambda e: e.ts_ms)
wall0 = first_live - int(first.ts_ms / 2)  # the sim runs at SPEED 2
c = DebouncedZoneCounter(ZONE); g = OccupancyGauge(c, 500, every_ms=2000); off = {}
for f, bucket in by_frame(boxes):
    for b in bucket: c.update(b)
    if (n := g.sample(bucket[0].ts_ms)) is not None: off[wall0 + int(bucket[0].ts_ms / 2)] = n
mine = [(t, n) for t, n in live if t <= max(off)]
print('offline samples', len(off), '| live samples in the scene', len(mine), '| all equal at the same ts:', all(off.get(t) == n for t, n in mine))
print('offline samples missing live (ms after scene start):', sorted(t - wall0 for t in set(off) - {t for t, _ in mine}))" && cd ..
make delivery
make down
# the edge gauge live on the MTID clip (media/sample.mp4), with the zone rule and without, about 5 min each
for px in 30 0; do
  make down && OCCUPANCY_GAUGE_MS=500 MIN_TRAVEL_PX=$px GRAFANA_PORT=3001 make up-video
  i=0; until [ "$(docker compose --profile video exec -T postgres psql -U postgres lab -At -c "select count(*) from zone_occupancy where device_id='edge-01'" 2>/dev/null || echo 0)" -ge 280 ]; do i=$((i+1)); [ $i -ge 60 ] && { echo TIMEOUT; break; }; sleep 10; done
  docker compose --profile video exec -T postgres psql -U postgres lab -At -F ' | ' -c "select device_id, count(*) samples, round(extract(epoch from max(received_at)-min(received_at))::numeric,1) span_s, round(avg(in_zone)::numeric,3) mean, round(avg((in_zone>0)::int)::numeric,3) share_pos, max(in_zone) from zone_occupancy group by 1" -c "select device_id, state, fps from device_status"
done
make down
# the same clip's offline gauge, with and without the zone rule
cd harness && python -c "
import json
from replay import balance; from replay.schema import read_tracks; from replay.zones import DebouncedZoneCounter
poly = [tuple(p) for p in json.load(open('zone.json'))['polygon']]
boxes = sorted(read_tracks('runs/bytetrack.agnostic.jsonl'), key=lambda b: (b.frame, b.track_id))
for px in (0, 30):
    v = list(balance.gauge(DebouncedZoneCounter(poly, min_travel_px=px), boxes).values())
    print('offline gauge, min_travel_px', px, '| frames', len(v), '| mean', round(sum(v) / len(v), 3), '| share > 0', round(sum(x > 0 for x in v) / len(v), 3), '| max', max(v))" && cd ..
```
