# Case study: boxes straddling two vehicles side by side

> Status: baseline measured on one annotated public clip. The fix moved to second boxes: its design is declared under
> Fix, and none of its steps has been run. Every measured figure comes from a command in Reproduce.

## Problem

In dense traffic the detector sometimes puts one box across two vehicles side by side. The tracker carries that box
like any other, and the zone counter can log it as a visit: a vehicle that does not exist, counted next to the two that
do. The [phantom boxes study](case-study-phantoms.md#second-kind-a-box-between-two-vehicles-side-by-side) found such
enters on an expressway clip with no annotations: which box was a straddle came from visual passes, the rest of the
count could not be scored, and a fix could not be checked for the real vehicles it would remove. This study uses a clip
whose vehicles are annotated box by box, so both come from the annotations instead of visual passes: which visits are
real, from the annotated tracks, and which detections are straddles, by a fixed rule on the annotated boxes (Scorer,
below) that overcounts as well as undercounts (Limitations).

On this clip no enter turned out to sit on a box across two separate vehicles, so the fix moved to the second-box failure
behind 8 of the 9 extra enters, one vehicle counted on two tracks, and the straddle candidate is not pursued on this clip
(Fix).

## Reproducing it without hardware

- Clip: UA-DETRAC test sequence MVI_40714, 960x540 at 25 fps, 1180 frames, 47.2 s (`ffprobe -count_frames`). A fixed,
  elevated view down a divided road, daytime; the near-left carriageway is queued several vehicles abreast, buses among
  them. UA-DETRAC is CC BY-NC-SA 3.0, academic use only: this study publishes metrics and the citation, never a frame,
  crop or image. The clip, its annotations and every file derived from them stay in the gitignored `media/` and
  `harness/runs/`. Source in `media/SOURCES.md`; the check that the mp4 lines up with the annotations in
  [datasets.md](datasets.md).
- Annotated boxes: `scripts/detrac_to_gt.py` turns the sequence's XML into 33749 boxes in 62 tracks, after dropping 186
  boxes centred in its ignored regions (`runs/MVI_40714.gt.jsonl`, `runs/MVI_40714.ignored.json`).
- Zone: a polygon over the queued carriageway, its full width, with the top and bottom edges in the gaps between rows of
  standing vehicles and clear of the ignored regions, chosen from the frames and the annotations only, without reading
  any detector output (`runs/MVI_40714.zone.json`, written in Reproduce). The zone is busy: 8.775 annotated vehicles in
  it per frame on average and 12 at most (`replay.balance.true_occupancy`, footpoint inside).
- Visit truth: the debounced counter at its defaults, run over the annotated tracks, whose IDs are perfect
  (`detrac_to_gt.py --zone`): 27 visits, as enter times only (`runs/MVI_40714.truth.json`). 11 of them are stamped 160
  ms, as early as the counter's hysteresis allows: vehicles already standing in the zone when the clip starts.
- Detections: `yolov8n` on CPU in the edge image (Ultralytics 8.4.170), COCO car, bus and truck, class-agnostic NMS at
  `conf` 0.1, the dump script's defaults: 38485 boxes (`runs/MVI_40714.dets.jsonl`).
- Tracker and counter: ByteTrack, Ultralytics' `BYTETracker` at its `bytetrack.yaml` defaults (a track starts on a box
  scoring at least 0.25 and is carried on boxes above 0.1), replayed on the saved detections as in the phantom study
  (`--tracker bytetrack`): 139 track IDs, 28882 boxes. Then `DebouncedZoneCounter` at its defaults, and once more with
  the phantom study's zone rule (`min_travel_px=30`).
- Scorer: `replay.straddle`. A detection centred in an ignored region is dropped first: nothing there is annotated, so
  nothing there can be judged. A kept detection is unmatched when no annotated box in its frame overlaps it at IoU 0.5
  or more, and a straddle when it is unmatched and lies across two annotated boxes side by side as `replay.between`
  defines a bridge, minus its score test: the two boxes' vertical ranges overlap by at least half the shorter height,
  they overlap each other at IoU under 0.1, the detection covers at least 0.2 of its own area on each and matches
  neither, and its centre lies between theirs in x. The rule does not ask for two separate vehicles: 0.2 of the box on
  each is enough, so a box mostly on one vehicle passes, and two annotated boxes one lying mostly inside the other (a
  car in front of a bus) still pass as side by side at IoU under 0.1, so a box on that car alone can read as a straddle
  across both. An enter is a straddle at enter when its track's box in the enter frame stands for a straddle; a track
  box stands for the detection it overlaps most in its frame, and only at IoU above 0.5. Straddle share is the share
  of a track's boxes that stand for straddles. Visits are scored by `replay.score` on enter time alone, 2 s tolerance,
  as in the other studies.
- The scorer and its thresholds (match IoU 0.5, touch 0.2, mask by box centre, a track box standing for a detection
  only above IoU 0.5) and the zone rule's 30 px were fixed before any result on this clip was seen; none was changed
  after, except for one fix to the scorer's code: straddle share 0.5 or more had been counted on the share as rounded
  for display, and is now counted on the exact share. The clip's figure is the same either way, since no entering
  track's share is near half (`replay.straddle`'s enters table, Reproduce). The breakdowns of what the straddles are
  (Baseline) were looked at after the results, and the rule was kept. The zone was chosen from the frames and the
  annotations only, without reading any detector output, though the detections already existed when it was drawn.

The headline figure is straddle at enter: whether the box an enter was counted on stands for a straddle (Scorer,
above). Straddle share 0.5 or more is the secondary one.

## Baseline

### Visits

`replay.score`; the annotated tracks go through the same command as a ceiling:

| tracks | counter | enters | matched | false visits | missed visits | f1 |
|---|---|---|---|---|---|---|
| ByteTrack | naive, centroid | 72 | 27 | 45 | 0 | 0.545 |
| ByteTrack | debounced | 35 | 26 | 9 | 1 | 0.839 |
| ByteTrack | debounced + zone rule 30 px | 30 | 15 | 15 | 12 | 0.526 |
| annotated | naive, centroid | 36 | 27 | 9 | 0 | 0.857 |
| annotated | debounced | 27 | 27 | 0 | 0 | 1.0 |
| annotated | debounced + zone rule 30 px | 27 | 16 | 11 | 11 | 0.593 |

ByteTrack with the debounced counter logs 35 enters for 27 visits: 26 matched, 9 false, 1 missed. `--explain` finds 2
static tracks among them (453 and 388, which moved 4.3 and 7.2 px); the 33 moving enters alone match 26 with 7 false
and 1 missed (F1 0.867). The annotated tracks' debounced row is a consistency check, not a result: the truth is that
counter on those tracks. The zone-rule rows cannot be read by time. Under the rule even the annotated tracks score 16
matched, 11 false and 11 missed: a vehicle standing in the zone at the start is stamped once it has moved 30 px, not at
160 ms, so it no longer lines up with its truth enter. ByteTrack's zone-rule row carries the same shift; the split by
vehicle below scores it without that shift.

### Straddles

Of the 38485 detections, 3334 are centred in an ignored region and dropped; 35151 are kept, in all 1180 frames. 4349 of
those match no annotated box, and 568 are straddles (0.0162 of the kept), in 490 of the 1180 frames (0.4153).

| run | enters | straddle at enter | straddle share 0.5 or more | enters whose track holds any straddle |
|---|---|---|---|---|
| debounced | 35 | 1 | 0 | 3 |
| debounced + zone rule 30 px | 30 | 1 | 0 | 3 |

By the rule, 1 enter of 35 is counted on a straddle: track 750 at 42040 ms. That one enter is a second box on part of
one car, car 25, whose annotated box lies mostly inside a bus's, bus 30's. It is not a box across two vehicles, so on
this clip no enter sits on a box across two separate vehicles. No entering track spends half its boxes on straddles;
the other two entering tracks that hold any are 21 (1 of its 857 boxes) and 726 (6 of 85). The zone rule leaves track
750 in: it moves with the traffic (327.6 px).

Track 750's 17 straddles all lie across car 25 and bus 30. Each has 0.81 to 0.98 of its area inside car 25's annotated
box, at IoU 0.332 to 0.499 with it, and car 25's box lies 0.719 to 1.0 inside bus 30's, at IoU 0.062 to 0.097: the pair
passes as side by side, and a box on the car touches both. None of the 17 puts 0.2 of its area on each annotated box
outside the other. The detection its box stands for at the enter, at score 0.164, has 0.899 of its area inside car
25's box and all of it inside bus 30's, and covers 0.473 of car 25's box; car 25's own track, 656, is on car 25 in
that frame and enters in the same frame. Of its 48 boxes, 17 are these straddles, 22 sit on car 25 and 9 on neither.
ByteTrack reports a track from the frame after its birth: track 750's first box, at frame 1033, is a straddle across
the same pair scoring 0.425, and the detection it was born on, at frame 1032, is one too, scoring 0.38, above the 0.25
a track needs to start. Bus 30 is counted on its own track as well (228). This reading of track
750, like the breakdowns at the end of this section, was made after the results; the rule was not changed for it.

Straddles score low:

| detections | count | min | lower quartile | median | upper quartile | max | share under 0.25 |
|---|---|---|---|---|---|---|---|
| kept | 35151 | 0.1 | 0.246 | 0.465 | 0.707 | 0.943 | 0.255 |
| unmatched | 4349 | 0.1 | 0.125 | 0.163 | 0.254 | 0.761 | 0.741 |
| straddles | 568 | 0.1 | 0.12 | 0.142 | 0.171 | 0.644 | 0.91 |

0.91 of the straddles score under 0.25, against 0.255 of all kept detections, so most of them cannot start a track;
ByteTrack can still use them to carry a track it already has. Track 750 was born on one of the few that can start one.
Most of the straddles, and most of that 0.91, are one pair of buses (What the straddles are, below).

### Where the false enters come from

`replay.score` matches by time and cannot say which enter is which vehicle, and 11 truth enters share the timestamp 160
ms. So each enter is also put down to a vehicle: its track's box in the enter frame against the annotated boxes in that
frame, at IoU 0.5 or more. The first enter on an annotated vehicle with a truth visit is that visit; any other is extra.

| enter put down to | debounced | debounced + zone rule 30 px |
|---|---|---|
| first enter of a truth visit | 26 | 25 |
| straddle at enter, by the rule (track 750) | 1 | 1 |
| second box on a vehicle another track is on | 5 | 2 |
| same vehicle again under a new track ID | 1 | 0 |
| box on part of one vehicle (no annotated box at IoU 0.5) | 2 | 2 |
| annotated vehicle with no truth visit | 0 | 0 |
| truth visits with no enter | 1 | 2 |

Without the zone rule this agrees with the time match: 26 visits found, 1 missed, 9 extra. One of the 9 is the rule's
straddle at enter, track 750, and seven are a second box on one vehicle. Five enter while another track is on the same
annotated vehicle: tracks 9 and 543 on bus 27, 749 on bus 30, and 453 and 592 on cars 12 and 17. Two, tracks 550 and
726, match no annotated box but lie wholly inside one bus's annotated box, at IoU 0.3 and 0.35 with it: a box on part
of a bus. The last, track 388, is a standing car counted again under a new ID, with no other track on it. Put down by
vehicle alone, with no straddle row (a pass looked at after the results), track 750 joins 550 and 726: its track box
in the enter frame, not the detection it stands for, matches no annotated box at IoU 0.5, and 0.88 of it lies inside
car 25's, at IoU 0.46. Then 8 of the 9 extra enters are a second box on one vehicle and none is a box across two
vehicles; with the zone rule, all 5 extra enters are. The missed visit is car 34: a ByteTrack box covers it at IoU 0.5
in 19 of its 266 frames, on tracks 257 and 272, and neither enters.

With the zone rule, by vehicle: 25 visits found, 2 missed, 5 extra, against 26, 1 and 9 without it. The rule removes
four extra enters (9, 388, 453 and 592, the two static ones among them) and loses car 11: track 10 follows car 11 in all
746 of its frames but commits no enter under the rule, its documented trade-off: a vehicle standing in the zone counts
only once it has moved 30 px from where its track started, and track 10 gets that far only in its last frames in the
zone, as it crosses the bottom edge on its way out, too close to the edge for the counter's margin and too late to stay
the minimum dwell. Its travel in `--explain` is the spread of all its footpoints, most of it outside the zone. Track
750, the two boxes on part of a bus and the second boxes on the buses (543, 749) stay.

### The detection-only heuristic against the scorer's straddles

`replay.between` flags bridge boxes from the detections alone, for clips with no annotations, and gives each enter's
track its share of them. Against the straddles:

| `replay.between` rule | flags the straddle at enter (track 750) | flags enters that are not a straddle at enter |
|---|---|---|
| any bridge box in the track | yes | 4 (tracks 404, 131, 656, 726) |
| bridge share 0.5 or more | no | 0 |
| the box at the enter is a bridge box | no | 1 (track 726) |

The same with the zone rule. It flags 4 of track 750's 48 boxes, but not the one it enters with. Per detection it flags
411 boxes in 339 frames (median score 0.14): 247 straddles, 114 boxes that match an annotated vehicle at IoU 0.5, 39
that are unmatched but not straddles, and 11 centred in an ignored region. It finds 247 of the 568 straddles, 233 of
them on one pair of buses (below).

### What the straddles are, looked at after the results

The rule was fixed before any result; these breakdowns were looked at after the results, once the one straddle at
enter turned out to be a box on one car, and change nothing in the rule or in the figures above. Each straddle is
taken with the two annotated boxes the scorer reports for it; the boxes nest when the smaller lies at least half inside
the larger:

| straddles | count | frames | 0.2 or more on each box outside the other | 0.8 or more on one box | boxes nest | under 0.25 | median score | bridge boxes | ByteTrack boxes standing for one |
|---|---|---|---|---|---|---|---|---|---|
| all | 568 | 490 | 218 | 314 | 33 | 517 | 0.142 | 247 | 48 |
| across buses 27 and 28 | 461 | 414 | 182 | 263 | 0 | 451 | 0.139 | 233 | 0 |
| across every other pair | 107 | 101 | 36 | 51 | 33 | 66 | 0.21 | 14 | 48 |

218 of the 568 put at least 0.2 of their area on each annotated box outside the other, as a box across two separate
vehicles would. That does not show they are such boxes: a box on a vehicle the annotators missed in the gap passes
too, and 182 of the 218 are on buses 27 and 28, whose run has not been checked (below). 314 have 0.8 or more of their
area on one of the two boxes, and 33 lie across two annotated boxes that nest, track 750's among them. One pair, buses
27 and 28, carries 461 of the 568 straddles, in 414 of the 490 frames that hold one, and 451 of the 517 that score
under 0.25; the bridge test finds 233 of them and 14 of the other 107. No ByteTrack box stands for any straddle on that
pair, so the per-enter figures do not depend on it; the per-detection ones, the score quartiles and the bridge test's
247, mostly describe it. Whether that run is a vehicle the annotators missed in the gap, part of one bus, or a box
across both has not been checked.

## Fix

The fix moved to the second-box failure behind 8 of the 9 extra enters, one vehicle counted on two tracks; the straddle
candidate (drop a box mostly covered by two higher-scoring boxes, and check that it keeps a car seen in the gap between
two nearer ones) is not pursued on this clip. What the baseline says about that candidate:

- On this clip no enter is counted on a box across two separate vehicles. The rule's one straddle at enter, track 750,
  is a second box on part of car 25, whose annotated box lies mostly inside bus 30's: taking out straddles can remove
  that one, and anything else it changes in the count comes from boxes it should not drop. Besides track 750, 7 of the
  9 extra enters are a second box on one vehicle (the split with the straddle row first), which the candidate does not
  target, and the last is track 388, a car counted again under a new ID.
- It would be measured per detection against the annotated boxes: straddles removed against annotated vehicles left
  with no detection, since a dropped box on a vehicle another detection still covers in that frame loses nothing. The
  bridge test, the closest rule in the harness, flags 247 straddles and 114 boxes that match an annotated vehicle; how
  many of those vehicles it would leave with no detection is not measured here. Per detection, most straddles are one
  pair of buses that no ByteTrack box stands for, and 218 of the 568 have the geometry of a box across two separate
  vehicles, 182 of them on those buses; what they are has not been checked.
- Track 750 is the enter a straddle filter could change: it is born on a box the rule calls a straddle, above the
  birth score, and the bridge test does not flag the box it enters with. It is also one of the second-box enters below,
  a box on part of car 25.

### Declared design: second boxes

A second box is a detection on a vehicle that already has one. When it starts a track of its own and that track enters
the zone, the vehicle is counted twice. This part of the study puts every detection and every enter down to an annotated
vehicle, runs seven steps against the ByteTrack baseline, each one change, and names a winner, or none, by a rule fixed
in advance. MVI_40714 is the clip under study. The counting study's MTID intersection clip (its annotated tracks and
its 14 labelled visits, with the phantom study's baseline tracks, `runs/phantoms/baseline.jsonl`) and the queue fixture
check that a step does no harm. Everything runs in the replay harness, the filters in front of ByteTrack: nothing under
`services/`, `replay/zones.py` or `replay/trackers/` changes. The code is `replay.secondbox`; its tests pin the
per-detection labels, the losses, the enter kinds, the containment filter and the steps. The winner rule is applied by
hand to the tables it prints, and the nms050 dump check is the `cmp` in Reproduce.

**What a second box is, per detection.** A step's detections are filtered first and masked after: on MVI_40714 a
detection centred in an ignored region still reaches the tracker but is not scored (`replay.straddle`'s mask); MTID has
no ignored-region file and is not masked. In each frame, detections and annotated boxes are paired one to one, greedily by IoU, from the pairs
at IoU 0.5 or more (ties: the higher detection score, then the earlier line in the file). A paired detection is its
vehicle's own (`vehicle`). An unpaired detection at IoU 0.5 or more with some annotated box is a `duplicate` of the box it
overlaps most. Otherwise, a detection with 0.8 or more of its own area inside one or more annotated boxes belongs to the
one of those it has the highest IoU with: in a car nested in a bus, the car. When that vehicle has a paired detection in
the frame, every such detection is `part`; when it has none, the one with the highest IoU is its `only` box (ties: score,
then file order) and the rest are part. Everything else is `other`. A detection in a frame outside the clip's annotated
range is `not judged`; one in a frame inside it with no annotated box is other. Second boxes are the duplicates and the
parts: on each vehicle-frame, the detections matching the vehicle less one, whichever of them is called the vehicle's.

A step that changes detections is also scored on what it loses, read on vehicle-frames (a frame and an annotated
vehicle), never by matching detections across runs: lost, paired at the baseline and with no paired detection after the
step; lost nested, the lost vehicle-frames whose annotated box lies 0.8 or more inside another annotated box of its
frame; only lost, an only box at the baseline and neither paired nor only after; and fit lost, paired in both, with the
paired detection's IoU down by 0.1 or more.

**How enters are put down by vehicle.** This puts the split under Where the false enters come from on fixed rules, and
reads no detections. Truth vehicles are the annotated tracks that the debounced counter, at its defaults, gives an
enter of their own, whatever counter setting the tracks are scored with. On MTID they are called annotated visits and are
never merged with its 14 labels, which name no vehicle. A run's enters come from the counter on its tracks, and an
enter's box is its track's box in the frame the enter commits. The enter is put on the annotated box it overlaps most in
that frame if their IoU is 0.5 or more; otherwise on the annotated box holding 0.8 or more of it, as for a part
detection (a part enter); otherwise on none. Enters are judged in order of frame, then highest IoU with their vehicle
first, then track ID, never in the counter's order: the counter emits an enter together with its exit. Each takes the
first kind that applies:

1. not judged: its frame is outside the annotated range;
2. no annotated box: it is put on none;
3. no truth visit: its vehicle has none;
4. second box, part: a part enter, which is never found;
5. found: the vehicle's first matching enter;
6. second box, duplicate: another track's box in the same frame has this vehicle as its best box, at IoU 0.5 or more;
7. again: the same vehicle again, under a new track ID or by the track that found it entering again.

The extra enters are every judged enter that is not found, and the missed are the truth vehicles with no found enter.
Every track on a vehicle is listed, and none is called "the" second box. A step's duplicate, part and again extras are
compared with the baseline's by kind, vehicle and time, never by track ID, since every tracker run restarts its IDs: a
baseline extra is removed when the step has no extra of the same kind on the same vehicle within 2000 ms of it, and a
step extra is new when the baseline has no such extra.

**The baseline under these rules**, on the saved baseline files, with no filter applied and no tracker run (Reproduce).
By vehicle:

| clip, counter | truth vehicles | enters | found | missed (vehicle) | extra | duplicate (tracks) | part (tracks) | again (tracks) | no annotated box | not judged |
|---|---|---|---|---|---|---|---|---|---|---|
| MVI_40714, debounced | 27 | 35 | 26 | 34 | 9 | 5 (9, 453, 543, 592, 749) | 3 (550, 726, 750) | 1 (388) | 0 | 0 |
| MVI_40714, zone rule 30 px | 27 | 30 | 25 | 11, 34 | 5 | 2 (543, 749) | 3 (550, 726, 750) | 0 | 0 | 0 |
| MTID, debounced | 21 | 39 | 20 | 36 | 16 | 0 | 2 (566, 584) | 0 | 14 | 3 |
| MTID, zone rule 30 px | 21 | 21 | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 1 |

No enter is put on a vehicle with no truth visit. On MVI_40714 this agrees with the split above: 26 found, car 34
missed, 9 extra, and 8 of the 9 are second boxes. Five are on a vehicle another track is on (tracks 9 and 543 on bus 27,
453 on car 12, 592 on car 17, 749 on bus 30) and three on part of one (550 on bus 27 and 726 on bus 30, at IoU 0.3 and
0.35 with it, and 750 on car 25, at 0.46); the ninth, 388, is car 17 again under a new ID. With the zone rule: 25 found,
cars 11 and 34 missed, 5 extra, all of them second boxes. On MTID, 20 of the 21 annotated visits are found and vehicle 36
is missed; the 16 judged extra enters are 14 with no annotated box and 2 part enters, tracks 566 and 584, both on
annotated vehicle 23 at IoU 0.0 and 0.01 with it, so boxes far smaller than that vehicle's box, and both static: they
moved 2.2 and 5.8 px (`replay.score --explain`, under 30 px). No MTID enter is a duplicate or an again.

Per detection:

| clip | detections | scored | vehicle | duplicate | part | only | other | not judged | nested annotated vehicle-frames |
|---|---|---|---|---|---|---|---|---|---|
| MVI_40714 | 38485 | 35151 | 27631 | 3171 | 1319 | 152 | 2878 | 0 | 642, on 11 vehicles |
| MTID | 20151 | 20151 | 4480 | 245 | 909 | 495 | 12861 | 1161 | 251, on 12 vehicles |

MVI_40714's annotations and detections both run from frame 0 to 1179. MTID's annotations stop at frame 3098 while its
detections run to 3198, hence its not judged. On MVI_40714, pairing one to one gives 7 vehicle-frames a detection that
putting each detection on its best box (at IoU 0.5 or more) would leave with none, and takes none away; on MTID the two
agree. MTID's part and only are not read as second boxes (Limitations).

**The steps** are alternatives, each one change against the baseline; ByteTrack is replayed on the saved detections in
the edge image, as in the phantom study.

| key | step | what shipping it would take |
|---|---|---|
| baseline | ByteTrack at its defaults, the phantom study's baseline step | nothing: the reference |
| contain080 | drop a detection with 0.8 or more of its own area inside a strictly higher-scoring detection of its frame | a step between detection and tracking, which `model.track()` does not offer |
| contain090 | the same at 0.9 | as contain080 |
| contain080_same | contain080, only when the container has the same detector class | as contain080 |
| contain090_same | contain090, only when the container has the same detector class | as contain080 |
| birth040 | track birth score 0.4 (`new_track_thresh`, default 0.25), the phantom study's step | a tracker yaml passed through the edge's `TRACKER` setting, with no edge code change |
| birth050 | track birth score 0.5, the phantom study's step | as birth040 |
| nms050 | the baseline step on a second detection file per clip, dumped with NMS IoU 0.5 (`dump_detections.py --iou 0.5`; the script's default is 0.7) | an `iou` argument to `model.track()` in `services/edge/src/main.py`, which passes none today |

A winner that needs a service change (a containment step, nms050) is reported as measured in the replay harness only.

The containment filter runs on every detection as dumped, ignored regions included: what ByteTrack sees. It is one pass,
and every other detection in the frame can contain, whether or not it is dropped itself: if A contains B and B contains
C, both B and C go. The container must score strictly higher on the saved scores, so equal scores drop neither, and the
output keeps the input's order. The `_same` steps read the detector's label (car, bus or truck, whichever it scored
highest after class-agnostic NMS), not the vehicle: they keep a box of another class inside a larger one, such as a car
box inside a bus box.

For nms050, each clip is dumped again in one image and session, at NMS IoU 0.7 and at 0.5. Each 0.7 dump must be
byte-identical to the saved detections; if either is not, nms050 is dropped, no nms050 row is computed, and the results
say why. No effect is predicted: NMS compares pairs of boxes, and greedy NMS at a lower IoU can keep a box that 0.7
suppressed, when the box that suppressed it is itself suppressed at 0.5, so the 0.5 set need not be a subset of the 0.7
set.

Not tried: suppressing a track birth inside an active track's box. In Ultralytics 8.4.170, BYTETracker removes
duplicates only between tracked and lost tracks (IoU above 0.85), and starts a track from any unmatched detection at or
above `new_track_thresh` with no check against active tracks (`BYTETracker._init_new_tracks` in
`ultralytics/trackers/byte_tracker.py`; `remove_duplicate_stracks`, called from `merge_track_pools`, in
`ultralytics/trackers/utils/stracks.py`; both printed in Reproduce). Trying it would change tracker code the services
share and override a private method of an unpinned dependency. If it is ever tried, it is labelled looked at after the
results and cannot win.

**How each step is scored.** Every step is reported on both clips; nothing is left out.

- Visits: `replay.phantoms`' table, as in the phantom study (enters, static, moving, matched, false visits, missed
  visits, f1, moving f1, queue matched, queue false), against MVI_40714's 27 truth visits and MTID's 14 labels, on
  enter time, 2 s tolerance.
- By vehicle: the split above on each step's tracks, with the counter at its defaults (this run decides the winner) and
  with the zone rule, 30 px. The zone-rule run is read against the zone rule's own split of the baseline (on MVI_40714:
  25 found, 2 missed, 5 extra); the zone rule alone is a reference row, not a candidate. Columns: found, missed, the
  extras by kind with their track IDs, and on MVI_40714 the extras removed and new against the baseline.
- Per detection: for each step that changes detections (the containment steps and nms050), on both clips, the label
  counts, the change in duplicates and parts against the baseline, and lost, lost nested, only lost and fit lost, each
  also by vehicle: the annotated vehicles behind it, by ID, with their number of frames. The birth steps read
  "unchanged (tracker setting)". Most changed detections are boxes no track uses: the per-detection counts say where
  boxes went, and the enter tables are the result.
- The queue fixture: as in the phantom study, each step's detection filter on the ground-plane tracker, reported but not
  part of the winner rule. Its 634 boxes, all cars of 90 x 60 px, never intersect one another in a frame (183
  same-frame pairs), so no containment step can change it. birth040 and birth050 (ByteTrack settings) and nms050 (the
  fixture has no video to dump again) are n/a on it.
- Scores come from the replay's tracks in memory. The written track files round the boxes, so an enter can commit a
  frame apart from where it does in the saved baseline file; any such enter is noted with the results, and the baseline
  step's written tracks must match the saved baseline file once sorted (Reproduce). The `--dets` run prints the baseline
  step's every enter from its tracks in memory, found ones included, to set beside `--tracks` on the saved file.

**The winner rule.** The candidates are contain080, contain090, contain080_same, contain090_same, birth040, birth050 and
nms050; the baseline never wins. A candidate qualifies when all of these hold, with the counter at its defaults:

1. MVI_40714 by vehicle: all 26 vehicles the baseline found are still found. Finding car 34 does not make up for losing
   another.
2. MVI_40714 by vehicle: fewer than 9 extra enters.
3. MTID by vehicle, against its 21 annotated visits: all 20 vehicles the baseline found are still found, and 16 judged
   extra enters or fewer (the baseline's 14 with no annotated box and 2 part).
4. MTID by `replay.score`, against its 14 labels: all 14 matched.

The winner is the qualifying step with the fewest MVI_40714 extra enters. A tie goes to the fewer lost vehicle-frames
(per detection, MVI_40714 and MTID summed; a step that changes no detection counts 0); a tie after that gives joint
winners, none preferred. If no step qualifies, the result is negative. The grid and the thresholds are fixed: a variation
tried later is labelled looked at after the results and cannot win.

Found sets decide, not the time match, because the time match pairs enters with truth enters by count within 2 s, not by
vehicle (Limitations): a step can lose one vehicle's enter and keep the same matched count whenever another enter lands
within 2 s of that vehicle's truth enter, which is easy where truth enters crowd together, such as the 11 at 160 ms on
MVI_40714. The found set names the vehicles, so a lost one shows.

**Declared with the baseline in view.** This is not a blind design: it was written after the baseline above, and this
study's reading of track 750, had been seen.

- The birth scores 0.4 and 0.5 are the phantom study's steps, whose MTID rows are already published
  [there](case-study-phantoms.md#fixes-one-at-a-time); they are re-reported here, not new.
- The containment shares 0.8 and 0.9 were set after this study had published that track 750's 17 straddles have 0.81
  to 0.98 of their area inside car 25's annotated box, and that track 750 was born on a detection scoring 0.38.
- NMS IoU 0.5 is a round value below the default 0.7.
- The match IoU 0.5, the part share 0.8 and the fit-lost drop 0.1 are fixed here.

**What the result may claim.** The result is a per-enter account on one 47.2 s clip. It says which of the 8 second-box
enters (duplicates 9, 453, 543, 592 and 749; parts 550, 726 and 750) a step removes, which extras it adds and which
vehicle-frames it loses, and makes no rate, percentage or general claim. A step is called a fix for second boxes only
for the second-box enters it removes, less the new ones. MTID has no baseline enter on a vehicle another track is on,
and its 2 part enters are static (tracks 566 and 584 moved 2.2 and 5.8 px). So MTID checks that a step does no harm,
not that the fix carries over.

### Results

The design above was committed before any step was run, and the results change nothing in it. The cells fill in from
the commands marked not run yet at the end of Reproduce; the n/a cells are declared above.

Checks before any row is read:

- Each 0.7 dump byte-identical to the saved detections: TBD.
- The baseline step's tracks against the saved baseline files, sorted, and its rows against the baseline above: TBD.
- Enters whose frame differs from the saved baseline file's: TBD.

Visits, MVI_40714, against its 27 truth visits:

| key | enters | static | moving | matched | false visits | missed visits | f1 | moving f1 | queue matched | queue false |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a | n/a |
| birth050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a | n/a |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a | n/a |

Visits, MTID, against its 14 labels:

| key | enters | static | moving | matched | false visits | missed visits | f1 | moving f1 | queue matched | queue false |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a | n/a |
| birth050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a | n/a |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a | n/a |

By vehicle, MVI_40714, counter at its defaults (this table decides the winner):

| key | found | missed | extra | duplicate | part | again | no truth visit | no annotated box | removed | new |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

By vehicle, MVI_40714, zone rule 30 px, against the zone rule's own split of the baseline (the baseline row is the zone
rule alone, a reference):

| key | found | missed | extra | duplicate | part | again | no truth visit | no annotated box | removed | new |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

By vehicle, MTID, against its 21 annotated visits, counter at its defaults:

| key | found | missed | judged extra | duplicate | part | again | no truth visit | no annotated box | not judged |
|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

By vehicle, MTID, zone rule 30 px (the baseline row is the zone rule alone, a reference):

| key | found | missed | judged extra | duplicate | part | again | no truth visit | no annotated box | not judged |
|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Per detection, MVI_40714, scored after the mask:

| key | vehicle | duplicate | part | only | other | not judged | duplicate change | part change | lost | lost nested | only lost | fit lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | unchanged (tracker setting) | | | | | | | | | | | |
| birth050 | unchanged (tracker setting) | | | | | | | | | | | |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Per detection, MTID:

| key | vehicle | duplicate | part | only | other | not judged | duplicate change | part change | lost | lost nested | only lost | fit lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain080_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| contain090_same | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| birth040 | unchanged (tracker setting) | | | | | | | | | | | |
| birth050 | unchanged (tracker setting) | | | | | | | | | | | |
| nms050 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Winner, by the rule above: TBD.

Per enter: which of the 8 second-box enters each step removes, the extras it adds and the vehicle-frames it loses, by
vehicle: TBD.

## Limitations

- One clip, one detector, one camera, 47.2 s, 27 visits. The headline figure rests on a single enter, and that enter's
  box lies mostly on one car.
- 11 of the 27 truth visits are vehicles already in the zone at the first frame. Under the zone rule they are stamped
  when they move, so the rule cannot be scored by time against this truth.
- The truth carries enters only, no exits; visits are matched on enter time, greedily in time order, so the time match
  pairs enters by count, not by vehicle, across the whole clip: at the start 11 truth enters share one timestamp with
  ByteTrack's own burst of enters, and later a false enter takes another vehicle's truth enter and the pairs after it
  shift onto their neighbours'. Without the zone rule the totals agree with the split by vehicle; the pairs do not. The
  split by vehicle is the check, and it rests on an IoU 0.5 match in one frame per enter.
- The rule overcounts as well as undercounts. A vehicle the annotators missed, seen in the gap between two annotated
  ones, reads as a straddle, and so does a box mostly on one vehicle that puts 0.2 of its area on a neighbour, or a box
  on a car whose annotated box lies mostly inside a bus's: 218 of the 568 straddles put 0.2 of their area on each
  annotated box outside the other, the geometry of a box across two separate vehicles, but a box on a missed vehicle in
  the gap can pass that too, and 182 of the 218 are on buses 27 and 28; the one straddle at enter is not among the 218.
  A box across a vehicle and an ignored region, or across two vehicles one behind the other in the road, one box
  higher in the frame than the other, is unmatched but not a straddle here. These breakdowns were looked at after the
  results; the rule was fixed before them and kept.
- The per-detection straddle figures rest mostly on one pair, buses 27 and 28: 461 of the 568 straddles and 233 of the
  247 the bridge test finds. Whether that run is a vehicle in the gap, part of one bus or a box across both has not
  been checked; no ByteTrack box stands for any of them, so the per-enter figures do not depend on it.
- Queued vehicles sit close to the zone's top edge: three annotated cars spend hundreds of frames with their footpoint
  under 20 px from it, two inside and one outside, so a detector box a little taller or shorter than the annotated one
  can put such a vehicle on the other side.
- A vehicle annotated as class `others` crosses the top edge in the clip's last 8 frames and has no truth visit; no
  ByteTrack enter is on it either. Two annotated cars have their footpoint just inside the top edge for 15 and 17
  frames, also with no truth visit.
- The edge image installs Ultralytics unpinned (8.4.170 here), and the replay `bytetrack` adapter was written against an
  earlier release: a rebuild can change the detections and the tracks.
- Second boxes, per detection: a detection on a vehicle whose annotated box was dropped for an ignored region is labelled
  other. The annotated box is dropped by its own centre and the detection kept by its own, so the two can fall on
  either side of a region's edge.
- On MTID, part and only labels can be static objects the detector scores as vehicles (the phantom study's lane dashes
  and bins) lying under a passing vehicle's annotated box, which the rules cannot tell from a second box; MTID's part
  and only counts are not read as second boxes.
- The second-box steps are tried on one clip, with one detector and one tracker (ByteTrack); MTID and the queue fixture
  check only that a step does no harm.

## Reproduce

```bash
setopt interactive_comments 2>/dev/null || true
# from the repo root, with media/UA-DETRAC/MVI_40714.mp4 and MVI_40714.xml in place (media/SOURCES.md) and the harness
# installed (pip install -e "harness[dev]"); detection and ByteTrack run in the edge image, CPU only (make up-video builds it)
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_read_frames,duration \
  media/UA-DETRAC/MVI_40714.mp4
docker run --rm edge-cv-lab-edge python -c "import ultralytics; print(ultralytics.__version__)"
docker run --rm edge-cv-lab-edge python -c "from ultralytics.utils import YAML; from ultralytics.utils.checks import check_yaml
print(YAML.load(check_yaml('bytetrack.yaml')))"
# the zone, then annotated boxes, ignored regions and visit truth (the debounced counter's defaults on the annotated tracks)
cd harness && mkdir -p runs
echo '{"polygon": [[0, 307], [603, 307], [643, 505], [0, 505]], "frame_size": [960, 540], "video": "UA-DETRAC/MVI_40714.mp4"}' \
  > runs/MVI_40714.zone.json
python scripts/detrac_to_gt.py --xml ../media/UA-DETRAC/MVI_40714.xml --out runs/MVI_40714 --zone runs/MVI_40714.zone.json
python -c "import json; e = json.load(open('runs/MVI_40714.truth.json'))['enters_ms']; print(len(e), 'visits |', e.count(160), 'at 160 ms')"
# detections (yolov8n, class-agnostic NMS, conf 0.1: the script's defaults), then ByteTrack replayed on them
cd .. && docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge python scripts/dump_detections.py \
  --video ../media/UA-DETRAC/MVI_40714.mp4 --fps 25 --model /app/yolov8n.pt --out runs/MVI_40714.dets.jsonl
docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge python -m replay.track \
  --dets runs/MVI_40714.dets.jsonl --tracker bytetrack --out runs/MVI_40714.bytetrack.jsonl
cd harness
python -c "import json; t = [json.loads(l) for l in open('runs/MVI_40714.bytetrack.jsonl')]; print(len({d['track_id'] for d in t}), 'ids', len(t), 'boxes')"
# visits: ByteTrack, then the annotated tracks through the same scorer; without and with the zone rule
for px in 0 30; do
  python -m replay.score --tracks runs/MVI_40714.bytetrack.jsonl --zone runs/MVI_40714.zone.json --truth runs/MVI_40714.truth.json \
    --min-travel-px $px
  python -m replay.score --tracks runs/MVI_40714.bytetrack.jsonl --zone runs/MVI_40714.zone.json --truth runs/MVI_40714.truth.json \
    --min-travel-px $px --explain
  python -m replay.score --tracks runs/MVI_40714.gt.jsonl --zone runs/MVI_40714.zone.json --truth runs/MVI_40714.truth.json \
    --min-travel-px $px
done
# straddles against the annotated boxes, and the enters whose tracks hold one
for px in 0 30; do
  python -m replay.straddle --dets runs/MVI_40714.dets.jsonl --gt runs/MVI_40714.gt.jsonl --ignored runs/MVI_40714.ignored.json \
    --tracks runs/MVI_40714.bytetrack.jsonl --zone runs/MVI_40714.zone.json --min-travel-px $px
done
# each enter put down to an annotated vehicle: its box in the enter frame against the annotated boxes at IoU 0.5; once
# with the straddle row first, and, looked at after the results, once by vehicle alone
python -c "
import json; from collections import defaultdict
from replay import straddle as S; from replay.between import _area, _inter; from replay.detections import read_detections
from replay.schema import read_tracks; from replay.trackers.greedy_iou import iou; from replay.zones import DebouncedZoneCounter, run
poly = [tuple(p) for p in json.load(open('runs/MVI_40714.zone.json'))['polygon']]
dets = S.mask(read_detections('runs/MVI_40714.dets.jsonl'), json.load(open('runs/MVI_40714.ignored.json'))['regions_xyxy'])
gt, tr = list(read_tracks('runs/MVI_40714.gt.jsonl')), list(read_tracks('runs/MVI_40714.bytetrack.jsonl'))
visits = {e.track_id for e in run(DebouncedZoneCounter(poly), gt) if e.kind == 'enter'}
g_at, t_at = defaultdict(list), defaultdict(list)
for b in gt: g_at[b.frame].append(b)
for b in tr: t_at[b.frame].append(b)
best = lambda b: max(g_at[b.frame], key=lambda g: iou(g.bbox, b.bbox))
on = lambda b: best(b).track_id if iou(best(b).bbox, b.bbox) >= 0.5 else None
kinds = ['first enter of a truth visit', 'straddle at enter', 'second box on a vehicle another track is on',
         'vehicle entering again under a new track ID', 'no annotated box at IoU 0.5', 'annotated vehicle with no truth visit']
for px, rule in ((0, True), (0, False), (30, True), (30, False)):
    seen, out = set(), defaultdict(list)
    for r in S.enters_table(dets, tr, gt, poly, min_travel_px=px):
        b = next(t for t in t_at[r['frame']] if t.track_id == r['track_id']); g, x = on(b), best(b)
        s = rule and r['straddle_at_enter']
        k = (kinds[1] if s else kinds[4] if g is None else kinds[5] if g not in visits
             else kinds[0] if g not in seen else kinds[2] if any(on(t) == g for t in t_at[b.frame] if t is not b) else kinds[3])
        if g is not None and not s: seen.add(g)
        out[k].append((r['track_id'], x.track_id, x.cls, round(iou(x.bbox, b.bbox), 2),
                       round(_inter(x.bbox, b.bbox) / _area(b.bbox), 2)))
    print('min_travel', px, '| straddle row first | (track, best annotated id, class, IoU, share of the box inside it)'
          if rule else '| by vehicle alone, no straddle row')
    for k in kinds if rule else (): print('  ', k, len(out[k]), out[k])
    if not rule: print('  ', {k: len(out[k]) for k in kinds}, '|', kinds[4], [o[0] for o in out[kinds[4]]])
    print('   extra enters', sum(len(out[k]) for k in kinds[1:]), '| a second box on one vehicle (another track on it, or no',
          'annotated box at IoU 0.5)', len(out[kinds[2]]) + len(out[kinds[4]]))
    for v in sorted(visits - seen) if rule else ():
        mine = [x for x in gt if x.track_id == v]
        cover = [[t.track_id for t in t_at[x.frame] if iou(t.bbox, x.bbox) >= 0.5] for x in mine]
        print('   truth visit with no enter: annotated', v, mine[0].cls, '| frames', len(mine), '| with a ByteTrack box at IoU 0.5:',
              sum(map(bool, cover)), 'tracks', sorted({t for c in cover for t in c}))"
# the detection-only heuristic against the straddles: per enter, then per detection, with the scores
for px in 0 30; do
  python -m replay.between --dets runs/MVI_40714.dets.jsonl --tracks runs/MVI_40714.bytetrack.jsonl --zone runs/MVI_40714.zone.json \
    --min-travel-px $px
done
python -c "
import json
from replay import between as B, straddle as S; from replay.detections import read_detections; from replay.schema import read_tracks
poly = [tuple(p) for p in json.load(open('runs/MVI_40714.zone.json'))['polygon']]
every = read_detections('runs/MVI_40714.dets.jsonl')
dets = S.mask(every, json.load(open('runs/MVI_40714.ignored.json'))['regions_xyxy'])
gt, tr = list(read_tracks('runs/MVI_40714.gt.jsonl')), list(read_tracks('runs/MVI_40714.bytetrack.jsonl'))
for px in (0, 30):
    st, br = S.enters_table(dets, tr, gt, poly, min_travel_px=px), B.enters_table(every, tr, poly, min_travel_px=px)
    at = [any(f for t, f in zip(tb, fl) if t.frame == e.frame)
          for e, tb, fl in B.enter_flags(every, tr, poly, lambda d, ds: B.bridge(d, ds) is not None, min_travel_px=px)]
    assert [s['track_id'] for s in st] == [b['track_id'] for b in br]
    for name, hit in (('bridge boxes > 0', [b['bridge_boxes'] > 0 for b in br]),
                      ('bridge share >= 0.5', [2 * b['bridge_boxes'] >= b['boxes'] for b in br]),
                      ('box at the enter is a bridge box', at)):
        print('min_travel', px, '|', name, '| straddles at enter flagged:',
              [s['track_id'] for s, h in zip(st, hit) if h and s['straddle_at_enter']],
              'of', [s['track_id'] for s in st if s['straddle_at_enter']], '| flagged, not a straddle at enter:',
              [s['track_id'] for s, h in zip(st, hit) if h and not s['straddle_at_enter']])"
python -c "
import json; from collections import Counter, defaultdict
from replay import straddle as S; from replay.between import bridge; from replay.detections import read_detections
from replay.schema import read_tracks
every = read_detections('runs/MVI_40714.dets.jsonl')
dets = S.mask(every, json.load(open('runs/MVI_40714.ignored.json'))['regions_xyxy'])
gt = list(read_tracks('runs/MVI_40714.gt.jsonl')); g_at, d_at = defaultdict(list), defaultdict(list)
for g in gt: g_at[g.frame].append(g)
for d in every: d_at[d.frame].append(d)
st, kept = {d for v in S.straddles(dets, gt).values() for d, *_ in v}, set(dets)
br = [d for ds in d_at.values() for d in ds if bridge(d, ds)]
print('bridge boxes', len(br), dict(Counter('centred in an ignored region' if d not in kept else 'straddle' if d in st
      else 'unmatched, not a straddle' if S._unmatched(d, g_at[d.frame]) else 'matches an annotated box' for d in br)))
print('straddles that are bridge boxes', len(st & set(br)), 'of', len(st))
for name, xs in (('kept', dets), ('unmatched', [d for d in dets if S._unmatched(d, g_at[d.frame])]), ('straddles', st)):
    s = sorted(d.score for d in xs); n = len(s)
    print(name, n, '| score min, quartiles, max', [round(s[int(p * (n - 1))], 3) for p in (0, 0.25, 0.5, 0.75, 1)],
          '| share under 0.25', round(sum(x < 0.25 for x in s) / n, 3))"
# looked at after the results: the one straddle at enter (track 750, enter frame 1051), what each of its boxes stands
# for, and the detection it was born on (ByteTrack reports a track from the frame after its birth)
python -c "
import json; from collections import Counter, defaultdict
from replay import straddle as S; from replay.detections import read_detections; from replay.schema import read_tracks
from replay.trackers.greedy_iou import iou
every, tr = read_detections('runs/MVI_40714.dets.jsonl'), list(read_tracks('runs/MVI_40714.bytetrack.jsonl'))
dets = S.mask(every, json.load(open('runs/MVI_40714.ignored.json'))['regions_xyxy'])
gt = list(read_tracks('runs/MVI_40714.gt.jsonl')); st = S.straddles(dets, gt); d_at, g_at, e_at = (defaultdict(list) for _ in '123')
for d in dets: d_at[d.frame].append(d)
for g in gt: g_at[g.frame].append(g)
for d in every: e_at[d.frame].append(d)
across = lambda d, f: next(((a.track_id, a.cls, b.track_id, b.cls) for x, a, b in st.get(f, []) if x is d), None)
mine = sorted((t for t in tr if t.track_id == 750), key=lambda t: t.frame); seq = []
for t in mine:
    src = max(d_at[t.frame], key=lambda d: iou(d.bbox, t.bbox), default=None)
    src = src if src is not None and iou(src.bbox, t.bbox) > 0.5 else None
    g = max(g_at[t.frame], key=lambda g: iou(g.bbox, t.bbox))
    seq.append((t.frame, src and round(src.score, 3), across(src, t.frame), g.track_id if iou(g.bbox, t.bbox) >= 0.5 else None))
print('boxes', len(seq), '| first box (frame, score, straddle across, on annotated)', seq[0],
      '| at the enter', [s for s in seq if s[0] == 1051])
print(dict(Counter('straddle across ' + str(p) if p else 'on annotated ' + str(g) if g else 'neither' for f, sc, p, g in seq)))
print('straddle scores', sorted(sc for f, sc, p, g in seq if p))
# born in the frame before its first box, on a detection scoring 0.25 or more that no track box there stands for
f0 = mine[0].frame - 1
held = {id(m) for t in tr if t.frame == f0 for m in [max(e_at[f0], key=lambda d: iou(d.bbox, t.bbox))] if iou(m.bbox, t.bbox) > 0.5}
print('frame before the first box', f0, '| detections there at 0.25 or more that no track box stands for',
      '(score, IoU with the first box, straddle across)',
      [(round(d.score, 3), round(iou(d.bbox, mine[0].bbox), 2), across(d, f0))
       for d in e_at[f0] if d.score >= 0.25 and id(d) not in held])"
# looked at after the results: track 750's straddles against the two annotated boxes they lie across, and the tracks on
# car 25 when it enters
python -c "
import json; from collections import defaultdict
from replay import straddle as S; from replay.between import _area, _inter; from replay.detections import read_detections
from replay.schema import read_tracks; from replay.trackers.greedy_iou import iou; from replay.zones import DebouncedZoneCounter, run
poly = [tuple(p) for p in json.load(open('runs/MVI_40714.zone.json'))['polygon']]
dets = S.mask(read_detections('runs/MVI_40714.dets.jsonl'), json.load(open('runs/MVI_40714.ignored.json'))['regions_xyxy'])
gt = list(read_tracks('runs/MVI_40714.gt.jsonl'))
tr = sorted(read_tracks('runs/MVI_40714.bytetrack.jsonl'), key=lambda t: (t.frame, t.track_id))
st = S.straddles(dets, gt); d_at = defaultdict(list); rows = []
for d in dets: d_at[d.frame].append(d)
share = lambda x, y: round(_inter(x.bbox, y.bbox) / _area(x.bbox), 3)
both = lambda a, b: [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]  # empty when they do not overlap
apart = lambda d, a, b: (min(_inter(d.bbox, a.bbox), _inter(d.bbox, b.bbox)) - _inter(d.bbox, both(a.bbox, b.bbox))
                         >= 0.2 * _area(d.bbox))
for t in (t for t in tr if t.track_id == 750):
    src = max(d_at[t.frame], key=lambda d: iou(d.bbox, t.bbox))
    for d, a, b in (x for x in st.get(t.frame, []) if x[0] is src and iou(src.bbox, t.bbox) > 0.5):
        c, u = sorted((a, b), key=lambda g: _area(g.bbox))
        rows.append((t.frame, c.track_id, u.track_id, share(d, c), round(iou(d.bbox, c.bbox), 3), share(d, u), share(c, d),
                     share(c, u), round(iou(c.bbox, u.bbox), 3), apart(d, a, b)))
col = lambda i: (min(r[i] for r in rows), max(r[i] for r in rows))
print('straddles', len(rows), '| (smaller, larger) annotated box', sorted({r[1:3] for r in rows}), '| least, most: share of the',
      'box inside the smaller', col(3), 'IoU with it', col(4), '| share of the smaller inside the larger', col(7), 'their IoU', col(8),
      '| with 0.2 or more of the box on each outside the other', sum(r[9] for r in rows))
print('at the enter (frame, smaller, larger, box inside smaller, IoU with it, box inside larger, smaller covered by the box,',
      'smaller inside larger, their IoU, 0.2 on each outside the other)', [r for r in rows if r[0] == 1051])
car = next(g for g in gt if g.frame == 1051 and g.track_id == 25)
print('frame 1051: tracks on car 25 at IoU 0.5', [t.track_id for t in tr if t.frame == 1051 and iou(t.bbox, car.bbox) >= 0.5],
      '| enters', [e.track_id for e in run(DebouncedZoneCounter(poly), tr) if e.kind == 'enter' and e.frame == 1051])"
# looked at after the results: every straddle against the two annotated boxes the scorer reports for it; all of them,
# then the most common pair against every other pair
python -c "
import json; from collections import Counter, defaultdict
from replay import straddle as S; from replay.between import _area, _inter, bridge; from replay.detections import read_detections
from replay.schema import read_tracks; from replay.trackers.greedy_iou import iou
every = read_detections('runs/MVI_40714.dets.jsonl')
dets = S.mask(every, json.load(open('runs/MVI_40714.ignored.json'))['regions_xyxy'])
st = [(f, d, a, b) for f, v in S.straddles(dets, list(read_tracks('runs/MVI_40714.gt.jsonl'))).items() for d, a, b in v]
d_at, m_at, held = defaultdict(list), defaultdict(list), Counter()
for d in every: d_at[d.frame].append(d)
for d in dets: m_at[d.frame].append(d)
br = {id(d) for ds in d_at.values() for d in ds if bridge(d, ds)}
for t in read_tracks('runs/MVI_40714.bytetrack.jsonl'):
    src = max(m_at[t.frame], key=lambda d: iou(d.bbox, t.bbox), default=None)
    if src is not None and iou(src.bbox, t.bbox) > 0.5: held[id(src)] += 1
key = lambda a, b: tuple(sorted(((a.track_id, a.cls), (b.track_id, b.cls))))
top = Counter(key(a, b) for f, d, a, b in st).most_common(1)[0][0]
both = lambda a, b: [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]  # empty when they do not overlap
apart = lambda d, a, b: (min(_inter(d.bbox, a.bbox), _inter(d.bbox, b.bbox)) - _inter(d.bbox, both(a.bbox, b.bbox))
                         >= 0.2 * _area(d.bbox))
print('pairs', len({key(a, b) for f, d, a, b in st}), '| most common', top)
for name, xs in (('all', st), ('that pair', [s for s in st if key(*s[2:]) == top]),
                 ('every other pair', [s for s in st if key(*s[2:]) != top])):
    sc = sorted(d.score for f, d, a, b in xs)
    print(name, '| straddles', len(xs), '| frames', len({f for f, *_ in xs}),
          '| 0.2 or more of the box on each outside the other', sum(apart(d, a, b) for f, d, a, b in xs),
          '| 0.8 or more on one', sum(max(_inter(d.bbox, a.bbox), _inter(d.bbox, b.bbox)) >= 0.8 * _area(d.bbox) for f, d, a, b in xs),
          '| the two nest (smaller at least half inside the larger)',
          sum(_inter(a.bbox, b.bbox) >= 0.5 * min(_area(a.bbox), _area(b.bbox)) for f, d, a, b in xs),
          '| under 0.25', sum(x < 0.25 for x in sc), '| median score', sc[int(0.5 * (len(sc) - 1))],
          '| bridge boxes', sum(id(d) in br for f, d, a, b in xs),
          '| ByteTrack boxes standing for one', sum(held[id(d)] for f, d, a, b in xs))"
# true occupancy; annotated vehicles in the zone with no truth visit; annotated vehicles standing by the zone's top edge
python -c "
import json; from collections import Counter, defaultdict
from replay import balance; from replay.geometry import point_in_polygon; from replay.schema import read_tracks
from replay.zones import DebouncedZoneCounter, run
poly = [tuple(p) for p in json.load(open('runs/MVI_40714.zone.json'))['polygon']]
gt = list(read_tracks('runs/MVI_40714.gt.jsonl')); occ = balance.true_occupancy(gt, poly)
print('annotated frames', len(occ), '| annotated vehicles in the zone per frame: mean', round(sum(occ.values()) / len(occ), 3),
      'max', max(occ.values()))
visits = {e.track_id for e in run(DebouncedZoneCounter(poly), gt) if e.kind == 'enter'}; inside = defaultdict(list)
for g in gt:
    if point_in_polygon(g.footpoint, poly): inside[g.track_id].append(g)
print('footpoint inside, no truth visit (id, class, frames inside, first, last, footpoint y inside):',
      [(t, b[0].cls, len(b), b[0].frame, b[-1].frame, [round(min(x.footpoint[1] for x in b)), round(max(x.footpoint[1] for x in b))])
       for t, b in inside.items() if t not in visits])
near = Counter((g.track_id, g.cls, point_in_polygon(g.footpoint, poly)) for g in gt
               if g.footpoint[0] < 603 and abs(g.footpoint[1] - 307) < 20)
print('frames with the footpoint under 20 px from the top edge, 100 or more (id, class, inside):',
      {k: n for k, n in near.items() if n >= 100})"
# second boxes (Fix): the baseline the declared design quotes, on the saved baseline files only, with no filter applied
# and no tracker run. Each enter put down by vehicle on both clips, without and with the zone rule (MTID's annotated
# tracks, zone and labels are the counting study's: case-study-tracking.md; its detections and baseline tracks are the
# phantom study's: case-study-phantoms.md, Reproduce), and MTID's labelled visits
for px in 0 30; do
  python -m replay.secondbox --tracks runs/MVI_40714.bytetrack.jsonl --gt runs/MVI_40714.gt.jsonl --zone runs/MVI_40714.zone.json \
    --min-travel-px $px
  python -m replay.secondbox --tracks runs/phantoms/baseline.jsonl --gt runs/mtid.gt.jsonl --zone zone.json --min-travel-px $px
done
python -c "import json; print(len(json.load(open('truth.json'))['enters_ms']), 'MTID labelled visits')"
# MTID's baseline enters by travel: its 2 part enters, tracks 566 and 584, are static (moved under 30 px)
python -m replay.score --tracks runs/phantoms/baseline.jsonl --zone zone.json --truth truth.json --explain
# each baseline detection labelled against the annotated boxes; one-to-one pairing against putting each detection on its
# best box; annotated boxes nested in another
python -c "
import json; from collections import Counter, defaultdict
from replay import secondbox as SB; from replay.detections import read_detections; from replay.schema import read_tracks
from replay.straddle import mask; from replay.trackers.greedy_iou import iou
for clip, d, g, ign in (('MVI_40714', 'runs/MVI_40714.dets.jsonl', 'runs/MVI_40714.gt.jsonl', 'runs/MVI_40714.ignored.json'),
                        ('MTID', 'runs/dets.agnostic.jsonl', 'runs/mtid.gt.jsonl', None)):
    every, gt, g_at = read_detections(d), list(read_tracks(g)), defaultdict(list)
    for x in gt: g_at[x.frame].append(x)
    dets = mask(every, json.load(open(ign))['regions_xyxy']) if ign else every  # the baseline's: no filter, then the mask
    labels = SB.classify(dets, gt); n = Counter(l for l, *_ in labels); paired = set(SB.cover(dets, labels)[0])
    best = {(x.frame, b.track_id) for x in dets for b in [max(g_at[x.frame], key=lambda b: iou(b.bbox, x.bbox), default=None)]
            if b is not None and iou(b.bbox, x.bbox) >= 0.5}
    nest = SB.nested(gt)
    print(clip, '| annotated frames', min(x.frame for x in gt), max(x.frame for x in gt), '| detection frames',
          min(x.frame for x in every), max(x.frame for x in every), '| detections', len(every), 'scored', len(dets))
    print('   labels', {k: n[k] for k in SB.LABELS})
    print('   vehicle-frames paired', len(paired), '| with each detection on its best box at IoU 0.5 instead', len(best),
          '| paired, not so', len(paired - best), '| so, not paired', len(best - paired))
    print('   nested annotated vehicle-frames', len(nest), 'on', len({v for f, v in nest}), 'vehicles')"
# the queue fixture's boxes: none intersects another in its frame, so no containment step can change the queue
python -c "
from collections import defaultdict; from itertools import combinations
from replay.between import _inter; from replay.detections import read_detections
q, at = read_detections('fixtures/queue.dets.jsonl'), defaultdict(list)
for d in q: at[d.frame].append(d)
pairs = [p for ds in at.values() for p in combinations(ds, 2)]
print('queue boxes', len(q), '| classes', sorted({d.cls for d in q}),
      '| sizes', sorted({(round(d.bbox[2] - d.bbox[0], 1), round(d.bbox[3] - d.bbox[1], 1)) for d in q}),
      '| same-frame pairs', len(pairs), '| pairs that intersect', sum(_inter(a.bbox, b.bbox) > 0 for a, b in pairs))"
cd ..
# Ultralytics' BYTETracker in the edge image (its version is printed above): where a track is born, and where duplicate
# tracks are removed (Not tried, under Fix)
docker run --rm edge-cv-lab-edge python -c "
import inspect; from ultralytics.trackers.byte_tracker import BYTETracker as B; from ultralytics.trackers.utils import stracks as S
for f, keep in ((B.update, ('_init_new_tracks(', 'merge_track_pools(')), (B._init_new_tracks, ('',)),
                (S.merge_track_pools, ('tracker.tracked_stracks, tracker.lost_stracks',)), (S.remove_duplicate_stracks, ('dup_thresh',))):
    lines, start = inspect.getsourcelines(f)
    print(inspect.getsourcefile(f).split('site-packages/')[-1], f.__name__)
    for i, l in enumerate(lines):
        if any(k in l for k in keep): print(' ', start + i, l.strip())"
# NOT RUN YET. The second-box steps, run only after the declared design above was committed; they will produce the
# Results tables under Fix. Edge image, CPU, no build and no download: the versions and the model's hash; both clips
# dumped again at NMS IoU 0.5 and 0.7, each 0.7 dump compared byte for byte with the saved detections (if either
# differs, nms050 is dropped and no nms050 row is computed); every step on both clips, each step's tracks written under
# runs/secondbox/, and the baseline step's every enter printed from its tracks in memory; the baseline step's tracks
# against the saved ones; then the by-vehicle baseline above again, which must print the same, its enters set beside
# the in-memory ones (any frame that differs is noted under Results).
e() { docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge "$@"; }
e python -c "import hashlib, torch, ultralytics
print(ultralytics.__version__, torch.__version__, hashlib.sha256(open('/app/yolov8n.pt', 'rb').read()).hexdigest())"
for iou in 0.5 0.7; do
  e python scripts/dump_detections.py --video ../media/UA-DETRAC/MVI_40714.mp4 --fps 25 --model /app/yolov8n.pt --iou $iou \
    --out runs/MVI_40714.dets.iou$iou.jsonl
  e python scripts/dump_detections.py --video ../media/sample.mp4 --model /app/yolov8n.pt --iou $iou --out runs/dets.agnostic.iou$iou.jsonl
done
cmp harness/runs/MVI_40714.dets.iou0.7.jsonl harness/runs/MVI_40714.dets.jsonl; c1=$?
cmp harness/runs/dets.agnostic.iou0.7.jsonl harness/runs/dets.agnostic.jsonl; c2=$?
m=(); t=()  # nms050 runs only when both 0.7 dumps are byte-identical to the saved detections
if [ "$c1" -eq 0 ] && [ "$c2" -eq 0 ]; then
  m=(--redetected nms050=runs/MVI_40714.dets.iou0.5.jsonl); t=(--redetected nms050=runs/dets.agnostic.iou0.5.jsonl)
else echo "a 0.7 dump differs from the saved detections: nms050 dropped, no nms050 row computed"; fi
e python -m replay.secondbox --dets runs/MVI_40714.dets.jsonl --zone runs/MVI_40714.zone.json --truth runs/MVI_40714.truth.json \
  --gt runs/MVI_40714.gt.jsonl --ignored runs/MVI_40714.ignored.json "${m[@]}" --out-dir runs/secondbox/MVI_40714
e python -m replay.secondbox --dets runs/dets.agnostic.jsonl --zone zone.json --truth truth.json --gt runs/mtid.gt.jsonl \
  "${t[@]}" --out-dir runs/secondbox/mtid
cd harness
diff <(sort runs/secondbox/MVI_40714/baseline.jsonl) <(sort runs/MVI_40714.bytetrack.jsonl)
diff <(sort runs/secondbox/mtid/baseline.jsonl) <(sort runs/phantoms/baseline.jsonl)
for px in 0 30; do
  python -m replay.secondbox --tracks runs/MVI_40714.bytetrack.jsonl --gt runs/MVI_40714.gt.jsonl --zone runs/MVI_40714.zone.json \
    --min-travel-px $px
  python -m replay.secondbox --tracks runs/phantoms/baseline.jsonl --gt runs/mtid.gt.jsonl --zone zone.json --min-travel-px $px
done
cd ..
```

## Citation

Wen, L., Du, D., Cai, Z., et al. (2020). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and
Tracking. Computer Vision and Image Understanding. https://doi.org/10.1016/j.cviu.2020.102907
