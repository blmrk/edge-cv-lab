# Case study: boxes straddling two vehicles side by side

> Status: baseline measured on one annotated public clip. The fix moved to second boxes: its design is declared under
> Fix, its steps have been run, and by the declared rule contain090_same wins, measured in the replay harness only
> (Results). Every measured figure comes from a command in Reproduce.

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
no ignored-region file and is not masked.
In each frame, detections and annotated boxes are paired one to one, greedily by IoU, from the pairs
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

The design above was committed before any step was run, and the results change nothing in it. The cells come from the
last block of Reproduce, run after the design was committed; the n/a cells are declared above. Track IDs are each run's
own: every tracker run restarts its IDs, so one number in two rows need not be one track (nms050's track 453 is a part
enter on bus 27, the baseline's track 453 a duplicate on car 12).

Checks before any row is read:

- Each 0.7 dump byte-identical to the saved detections: yes, on both clips. `cmp` prints nothing for MVI_40714 and for
  MTID, so nms050 is computed. The edge image reports Ultralytics 8.4.170 and torch 2.14.1+cpu, and the model's sha256
  is `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`; no earlier hash is recorded to compare it with.
- The baseline step's tracks against the saved baseline files, sorted, and its rows against the baseline above: both
  sorted diffs print nothing. The baseline rows of the two `--dets` runs equal the by-vehicle baseline above on both
  clips, at 0 and 30 px, and the baseline's per-detection label counts and nested vehicle-frames (642 and 251).
  `--dets` does not print the detections, scored or "on N vehicles" figures; its label counts sum to the scored 35151
  and 20151. The `--tracks` runs on the saved files print the by-vehicle baseline above again.
- Enters whose frame differs from the saved baseline file's: one, as the design allows. On MVI_40714 track 24,
  found on car 19, commits at frame 811 (32440 ms) on the tracks in memory and at frame 810 (32400 ms) on the saved
  file, at 0 and 30 px; its kind and vehicle are the same. Track 388's enter box is at IoU 0.88 with car 17 in memory
  and 0.87 on the saved file, in the same frame. No other enter differs in frame, kind, vehicle or judging order, on
  either clip.

Visits, MVI_40714, against its 27 truth visits:

| key | enters | static | moving | matched | false visits | missed visits | f1 | moving f1 | queue matched | queue false |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 35 | 2 | 33 | 26 | 9 | 1 | 0.839 | 0.867 | 6 | 0 |
| contain080 | 26 | 1 | 25 | 25 | 1 | 2 | 0.943 | 0.962 | 6 | 0 |
| contain090 | 27 | 1 | 26 | 26 | 1 | 1 | 0.963 | 0.981 | 6 | 0 |
| contain080_same | 28 | 1 | 27 | 26 | 2 | 1 | 0.945 | 0.963 | 6 | 0 |
| contain090_same | 28 | 1 | 27 | 26 | 2 | 1 | 0.945 | 0.963 | 6 | 0 |
| birth040 | 32 | 0 | 32 | 26 | 6 | 1 | 0.881 | 0.881 | n/a | n/a |
| birth050 | 31 | 0 | 31 | 25 | 6 | 2 | 0.862 | 0.862 | n/a | n/a |
| nms050 | 28 | 0 | 28 | 26 | 2 | 1 | 0.945 | 0.945 | n/a | n/a |

Visits, MTID, against its 14 labels:

| key | enters | static | moving | matched | false visits | missed visits | f1 | moving f1 | queue matched | queue false |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 39 | 18 | 21 | 14 | 25 | 0 | 0.528 | 0.8 | 6 | 0 |
| contain080 | 39 | 18 | 21 | 14 | 25 | 0 | 0.528 | 0.8 | 6 | 0 |
| contain090 | 39 | 18 | 21 | 14 | 25 | 0 | 0.528 | 0.8 | 6 | 0 |
| contain080_same | 39 | 18 | 21 | 14 | 25 | 0 | 0.528 | 0.8 | 6 | 0 |
| contain090_same | 39 | 18 | 21 | 14 | 25 | 0 | 0.528 | 0.8 | 6 | 0 |
| birth040 | 26 | 5 | 21 | 14 | 12 | 0 | 0.7 | 0.8 | n/a | n/a |
| birth050 | 22 | 1 | 21 | 14 | 8 | 0 | 0.778 | 0.8 | n/a | n/a |
| nms050 | 39 | 18 | 21 | 14 | 25 | 0 | 0.528 | 0.8 | n/a | n/a |

The queue columns read 6 matched and 0 false for the baseline and the four containment steps; no containment step can
change the queue (its boxes never intersect, above), so those columns cannot fail for them. birth040 and birth050 give
the phantom study's MTID rows again. On MVI_40714 the time match and the found sets below disagree for three steps:
contain080 matches 25 and finds 24 vehicles, contain090 matches 26, as the baseline does, and finds 25, and birth050
matches 25 and finds 26. contain090 is the case the found-set rule was declared for: its time match does not show that
it loses car 25. Found sets decide; the commands here do not break the other two differences down.

By vehicle, MVI_40714, counter at its defaults (this table decides the winner):

| key | found | missed | extra | duplicate | part | again | no truth visit | no annotated box | removed | new |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 26 | 34 | 9 | 5: 9 on 27, 453 on 12, 543 on 27, 592 on 17, 749 on 30 | 3: 550 on 27, 750 on 25, 726 on 30 | 1: 388 on 17 | 0 | 0 | 0 | 0 |
| contain080 | 24 | 25, 34, 50 | 2 | 2: 26 on 27, 646 on 30 | 0 | 0 | 0 | 0 | duplicate 453 on 12, 543 on 27, 592 on 17; part 550 on 27, 750 on 25, 726 on 30; again 388 on 17 | 0 |
| contain090 | 25 | 25, 34 | 2 | 2: 26 on 27, 672 on 30 | 0 | 0 | 0 | 0 | duplicate 453 on 12, 543 on 27, 592 on 17; part 550 on 27, 750 on 25, 726 on 30; again 388 on 17 | 0 |
| contain080_same | 26 | 34 | 2 | 2: 26 on 27, 648 on 30 | 0 | 0 | 0 | 0 | duplicate 453 on 12, 543 on 27, 592 on 17; part 550 on 27, 750 on 25, 726 on 30; again 388 on 17 | 0 |
| contain090_same | 26 | 34 | 2 | 2: 26 on 27, 674 on 30 | 0 | 0 | 0 | 0 | duplicate 453 on 12, 543 on 27, 592 on 17; part 550 on 27, 750 on 25, 726 on 30; again 388 on 17 | 0 |
| birth040 | 26 | 34 | 6 | 3: 9 on 27, 69 on 27, 103 on 30 | 3: 72 on 27, 102 on 25, 98 on 30 | 0 | 0 | 0 | duplicate 453 on 12, 592 on 17; again 388 on 17 | 0 |
| birth050 | 26 | 34 | 5 | 3: 9 on 27, 48 on 27, 67 on 30 | 2: 50 on 27, 65 on 30 | 0 | 0 | 0 | duplicate 453 on 12, 592 on 17; part 750 on 25; again 388 on 17 | 0 |
| nms050 | 26 | 34 | 2 | 0 | 1: 453 on 27 | 0 | 1: 687 on 47 | 0 | duplicate 9 on 27, 453 on 12, 543 on 27, 592 on 17, 749 on 30; part 750 on 25, 726 on 30; again 388 on 17 | 0 |

By vehicle, MVI_40714, zone rule 30 px, against the zone rule's own split of the baseline (the baseline row is the zone
rule alone, a reference):

| key | found | missed | extra | duplicate | part | again | no truth visit | no annotated box | removed | new |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 25 | 11, 34 | 5 | 2: 543 on 27, 749 on 30 | 3: 550 on 27, 750 on 25, 726 on 30 | 0 | 0 | 0 | 0 | 0 |
| contain080 | 23 | 11, 25, 34, 50 | 1 | 1: 646 on 30 | 0 | 0 | 0 | 0 | duplicate 543 on 27; part 550 on 27, 750 on 25, 726 on 30 | 0 |
| contain090 | 24 | 11, 25, 34 | 1 | 1: 672 on 30 | 0 | 0 | 0 | 0 | duplicate 543 on 27; part 550 on 27, 750 on 25, 726 on 30 | 0 |
| contain080_same | 25 | 11, 34 | 1 | 1: 648 on 30 | 0 | 0 | 0 | 0 | duplicate 543 on 27; part 550 on 27, 750 on 25, 726 on 30 | 0 |
| contain090_same | 25 | 11, 34 | 1 | 1: 674 on 30 | 0 | 0 | 0 | 0 | duplicate 543 on 27; part 550 on 27, 750 on 25, 726 on 30 | 0 |
| birth040 | 25 | 11, 34 | 5 | 2: 69 on 27, 103 on 30 | 3: 72 on 27, 102 on 25, 98 on 30 | 0 | 0 | 0 | 0 | 0 |
| birth050 | 25 | 11, 34 | 3 | 2: 48 on 27, 67 on 30 | 1: 65 on 30 | 0 | 0 | 0 | part 550 on 27, 750 on 25 | 0 |
| nms050 | 25 | 11, 34 | 2 | 0 | 1: 453 on 27 | 0 | 1: 687 on 47 | 0 | duplicate 543 on 27, 749 on 30; part 750 on 25, 726 on 30 | 0 |

No MVI_40714 enter is not judged, in any step or run. The removed and new columns compare duplicate, part and again
extras only, as declared, so nms050's enter on vehicle 47, an annotated vehicle with no truth visit, is an extra that
the new column does not list. Looked at after the results: vehicle 47 is the annotated vehicle of class `others` that
crosses the zone's top edge in the clip's last 8 frames (Limitations; the true-occupancy command in Reproduce prints
its ID), and track 687 enters on it in frame 1179, the last.

By vehicle, MTID, against its 21 annotated visits, counter at its defaults:

| key | found | missed | judged extra | duplicate | part | again | no truth visit | no annotated box | not judged |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 36 | 16 | 0 | 2: 566 on 23, 584 on 23 | 0 | 0 | 14: 12, 59, 219, 221, 263, 315, 366, 406, 487, 495, 575, 660, 661, 683 | 3: 784, 774, 813 |
| contain080 | 20 | 36 | 16 | 0 | 0 | 0 | 0 | 16: 13, 58, 204, 205, 246, 294, 345, 385, 464, 472, 547, 550, 585, 629, 630, 651 | 3: 749, 740, 777 |
| contain090 | 20 | 36 | 16 | 0 | 0 | 0 | 0 | 16: 13, 57, 203, 204, 245, 294, 345, 385, 464, 472, 548, 552, 587, 631, 632, 653 | 3: 752, 743, 780 |
| contain080_same | 20 | 36 | 16 | 0 | 2: 553 on 23, 571 on 23 | 0 | 0 | 14: 13, 58, 211, 213, 254, 304, 355, 395, 475, 483, 562, 644, 645, 666 | 3: 764, 755, 792 |
| contain090_same | 20 | 36 | 16 | 0 | 2: 552 on 23, 570 on 23 | 0 | 0 | 14: 13, 57, 210, 212, 253, 303, 354, 394, 474, 482, 561, 643, 644, 665 | 3: 764, 755, 792 |
| birth040 | 20 | 36 | 5 | 0 | 0 | 0 | 0 | 5: 6, 23, 59, 61, 112 | 1: 131 |
| birth050 | 20 | 36 | 1 | 0 | 0 | 0 | 0 | 1: 38 | 1: 77 |
| nms050 | 20 | 36 | 16 | 0 | 2: 554 on 23, 571 on 23 | 0 | 0 | 14: 13, 59, 212, 213, 254, 306, 358, 397, 478, 486, 563, 646, 647, 669 | 3: 769, 760, 795 |

By vehicle, MTID, zone rule 30 px (the baseline row is the zone rule alone, a reference):

| key | found | missed | judged extra | duplicate | part | again | no truth visit | no annotated box | not judged |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 774 |
| contain080 | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 740 |
| contain090 | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 743 |
| contain080_same | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 755 |
| contain090_same | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 755 |
| birth040 | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 131 |
| birth050 | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 77 |
| nms050 | 20 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 1: 760 |

Per detection, MVI_40714, scored after the mask:

| key | vehicle | duplicate | part | only | other | not judged | duplicate change | part change | lost | lost nested | only lost | fit lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 27631 | 3171 | 1319 | 152 | 2878 | 0 | +0 | +0 | 0 | 0 | 0 | 0 |
| contain080 | 27263 | 1623 | 442 | 151 | 2593 | 0 | -1548 | -877 | 368 | 159 | 1 | 59 |
| contain090 | 27445 | 1696 | 501 | 151 | 2700 | 0 | -1475 | -818 | 186 | 84 | 1 | 52 |
| contain080_same | 27498 | 1692 | 471 | 152 | 2612 | 0 | -1479 | -848 | 133 | 11 | 0 | 51 |
| contain090_same | 27555 | 1758 | 527 | 152 | 2706 | 0 | -1413 | -792 | 76 | 9 | 0 | 47 |
| birth040 | unchanged (tracker setting) | | | | | | | | | | | |
| birth050 | unchanged (tracker setting) | | | | | | | | | | | |
| nms050 | 27507 | 139 | 1007 | 218 | 2569 | 0 | -3032 | -312 | 125 | 9 | 0 | 1121 |

Per detection, MTID:

| key | vehicle | duplicate | part | only | other | not judged | duplicate change | part change | lost | lost nested | only lost | fit lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 4480 | 245 | 909 | 495 | 12861 | 1161 | +0 | +0 | 0 | 0 | 0 | 0 |
| contain080 | 4433 | 92 | 305 | 466 | 12661 | 1090 | -153 | -604 | 47 | 28 | 30 | 10 |
| contain090 | 4445 | 94 | 344 | 473 | 12730 | 1101 | -151 | -565 | 35 | 24 | 23 | 10 |
| contain080_same | 4467 | 124 | 677 | 494 | 12724 | 1105 | -121 | -232 | 13 | 3 | 3 | 6 |
| contain090_same | 4472 | 126 | 701 | 494 | 12775 | 1116 | -119 | -208 | 8 | 3 | 2 | 6 |
| birth040 | unchanged (tracker setting) | | | | | | | | | | | |
| birth050 | unchanged (tracker setting) | | | | | | | | | | | |
| nms050 | 4471 | 18 | 814 | 500 | 12634 | 1110 | -227 | -95 | 9 | 0 | 3 | 54 |

Vehicle-frames lost against the baseline, by vehicle (vehicle: frames), MVI_40714:

| key | lost | lost nested | only lost | fit lost |
|---|---|---|---|---|
| contain080 | 2: 26, 15: 44, 17: 26, 21: 3, 22: 1, 23: 1, 24: 1, 25: 44, 28: 55, 29: 3, 30: 5, 31: 7, 35: 38, 37: 14, 38: 1, 39: 2, 41: 5, 44: 2, 45: 1, 47: 7, 50: 58, 51: 19, 57: 5 | 2: 3, 15: 42, 25: 38, 35: 37, 37: 14, 47: 2, 50: 23 | 2: 1 | 2: 5, 17: 2, 20: 11, 21: 1, 25: 1, 26: 12, 27: 6, 30: 5, 31: 6, 36: 2, 47: 6, 51: 2 |
| contain090 | 2: 20, 17: 5, 21: 1, 22: 1, 23: 1, 24: 1, 25: 40, 28: 11, 29: 3, 30: 5, 31: 5, 35: 33, 37: 11, 38: 1, 39: 2, 41: 5, 45: 1, 47: 7, 50: 12, 51: 19, 57: 2 | 2: 1, 25: 37, 35: 33, 37: 11, 47: 2 | 2: 1 | 2: 1, 20: 10, 21: 1, 25: 1, 26: 12, 27: 6, 28: 1, 30: 5, 31: 6, 36: 1, 47: 6, 51: 2 |
| contain080_same | 2: 15, 21: 3, 22: 1, 23: 1, 24: 1, 28: 55, 29: 3, 30: 5, 31: 7, 35: 6, 38: 1, 41: 5, 44: 2, 47: 7, 50: 2, 51: 14, 57: 5 | 2: 3, 35: 6, 47: 2 | 0 | 2: 4, 20: 11, 21: 1, 26: 9, 27: 6, 30: 5, 31: 6, 36: 1, 47: 6, 51: 2 |
| contain090_same | 2: 12, 21: 1, 22: 1, 23: 1, 24: 1, 28: 11, 29: 3, 30: 5, 31: 5, 35: 6, 38: 1, 41: 5, 47: 7, 50: 1, 51: 14, 57: 2 | 2: 1, 35: 6, 47: 2 | 0 | 2: 1, 20: 10, 21: 1, 26: 9, 27: 6, 28: 1, 30: 5, 31: 6, 47: 6, 51: 2 |
| nms050 | 2: 23, 17: 1, 21: 4, 24: 2, 25: 5, 28: 4, 29: 1, 30: 4, 31: 2, 34: 4, 36: 14, 37: 1, 41: 5, 43: 6, 44: 5, 46: 9, 47: 16, 51: 19 | 25: 4, 47: 5 | 0 | 2: 51, 7: 233, 11: 2, 17: 151, 20: 11, 22: 1, 25: 4, 26: 16, 27: 458, 28: 2, 30: 75, 31: 13, 36: 44, 39: 1, 41: 1, 42: 1, 44: 2, 46: 21, 47: 31, 50: 1, 51: 2 |

The same, MTID:

| key | lost | lost nested | only lost | fit lost |
|---|---|---|---|---|
| contain080 | 5: 9, 11: 1, 12: 5, 13: 1, 18: 15, 19: 1, 42: 8, 45: 1, 46: 2, 47: 1, 49: 3 | 18: 14, 19: 1, 42: 8, 46: 2, 47: 1, 49: 2 | 11: 2, 19: 10, 25: 1, 41: 3, 43: 2, 46: 1, 47: 8, 49: 2, 64: 1 | 1: 1, 6: 1, 24: 1, 31: 1, 37: 3, 40: 3 |
| contain090 | 5: 6, 11: 1, 12: 3, 18: 13, 19: 1, 42: 8, 46: 1, 49: 2 | 18: 12, 19: 1, 42: 8, 46: 1, 49: 2 | 11: 2, 19: 8, 25: 1, 41: 2, 43: 2, 46: 1, 47: 4, 49: 2, 64: 1 | 1: 1, 6: 1, 24: 1, 31: 1, 37: 3, 40: 3 |
| contain080_same | 5: 4, 12: 5, 13: 1, 42: 3 | 42: 3 | 11: 1, 19: 1, 64: 1 | 1: 1, 6: 1, 31: 1, 40: 3 |
| contain090_same | 5: 2, 12: 3, 42: 3 | 42: 3 | 11: 1, 64: 1 | 1: 1, 6: 1, 31: 1, 40: 3 |
| nms050 | 4: 1, 7: 2, 11: 1, 12: 2, 33: 2, 44: 1 | 0 | 25: 1, 28: 1, 64: 1 | 1: 1, 5: 3, 6: 1, 9: 1, 11: 11, 20: 1, 21: 2, 23: 9, 24: 1, 25: 5, 31: 1, 33: 3, 37: 3, 40: 7, 43: 1, 45: 2, 47: 1, 49: 1 |

The birth steps change no detection and lose no vehicle-frame.

**Winner, by the rule above: contain090_same, measured in the replay harness only.** The four conditions, with the
counter at its defaults, and the tie-break:

| candidate | 1. MVI_40714: the baseline's 26 still found | 2. MVI_40714: under 9 extra | 3. MTID: the baseline's 20 still found, judged extra 16 or fewer | 4. MTID: 14 labels matched | qualifies | lost vehicle-frames, MVI_40714 + MTID |
|---|---|---|---|---|---|---|
| contain080 | no: 24 found, cars 25 and 50 lost | yes: 2 | yes: 20 found, 36 missed; 16 | yes: 14 | no | 368 + 47 = 415 |
| contain090 | no: 25 found, car 25 lost | yes: 2 | yes: 20 found, 36 missed; 16 | yes: 14 | no | 186 + 35 = 221 |
| contain080_same | yes: 26 found, 34 missed | yes: 2 | yes: 20 found, 36 missed; 16 | yes: 14 | yes | 133 + 13 = 146 |
| contain090_same | yes: 26 found, 34 missed | yes: 2 | yes: 20 found, 36 missed; 16 | yes: 14 | yes | 76 + 8 = 84 |
| birth040 | yes: 26 found, 34 missed | yes: 6 | yes: 20 found, 36 missed; 5 | yes: 14 | yes | 0 |
| birth050 | yes: 26 found, 34 missed | yes: 5 | yes: 20 found, 36 missed; 1 | yes: 14 | yes | 0 |
| nms050 | yes: 26 found, 34 missed | yes: 2 | yes: 20 found, 36 missed; 16 | yes: 14 | yes | 125 + 9 = 134 |

A found set is the truth vehicles less the missed, so 26 found with car 34 missed is the baseline's set, and 20 found
with vehicle 36 missed is MTID's. Five steps qualify. The fewest MVI_40714 extra enters among them is 2, shared by
contain080_same, contain090_same and nms050 (birth040 6, birth050 5). The tie goes to the fewer lost vehicle-frames:
contain090_same 84, nms050 134, contain080_same 146. So contain090_same wins alone, no joint winner; it needs a step
between detection and tracking, which `model.track()` does not offer, so it is measured in the replay harness only.

What it may claim, on this one 47.2 s clip: contain090_same removes 6 of the 8 second-box enters (duplicates 453, 543
and 592; parts 550, 726 and 750) and adds none, so it is a fix for those 6 second-box enters and no more. Two of the 8
stay, duplicates 9 and 749: its run has a duplicate on bus 27 at 280 ms and one on bus 30 at 41760 ms (its tracks 26
and 674), within 2000 ms of them. It keeps the 26 vehicles the baseline found and does not find car 34. It loses 76
vehicle-frames on MVI_40714 (9 nested) and 8 on MTID (3 nested); besides those, it loses no only box on MVI_40714 and
2 on MTID, and 47 and 6 vehicle-frames are fit lost. MTID checks that a step does no harm, not that the fix carries
over: there contain090_same finds the baseline's 20 annotated visits, has 16 judged extra enters and matches all 14
labels.

**Per enter, MVI_40714, counter at its defaults.** Of the 8 second-box enters (duplicates 9, 453, 543, 592 and 749;
parts 550, 726 and 750), from the removed and new columns and the extras lists. A baseline second-box enter is kept
when the step has an extra of the same kind on the same vehicle within 2000 ms; the step's own track and time are in
brackets:

| step | second-box enters removed | second-box enters kept | new | other extras |
|---|---|---|---|---|
| contain080 | 6: duplicates 453, 543, 592; parts 550, 726, 750 | duplicate 9 on bus 27 (26, 280 ms); duplicate 749 on bus 30 (646, 41760 ms) | 0 | again 388 removed |
| contain090 | 6: duplicates 453, 543, 592; parts 550, 726, 750 | duplicate 9 (26, 280 ms); duplicate 749 (672, 41760 ms) | 0 | again 388 removed |
| contain080_same | 6: duplicates 453, 543, 592; parts 550, 726, 750 | duplicate 9 (26, 280 ms); duplicate 749 (648, 41760 ms) | 0 | again 388 removed |
| contain090_same | 6: duplicates 453, 543, 592; parts 550, 726, 750 | duplicate 9 (26, 280 ms); duplicate 749 (674, 41760 ms) | 0 | again 388 removed |
| birth040 | 2: duplicates 453, 592 | duplicates 9 (9, 160 ms), 543 (69, 28600 ms), 749 (103, 41760 ms); parts 550 (72, 28920 ms), 750 (102, 42040 ms), 726 (98, 43400 ms) | 0 | again 388 removed |
| birth050 | 3: duplicates 453, 592; part 750 | duplicates 9 (9, 160 ms), 543 (48, 28600 ms), 749 (67, 41760 ms); parts 550 (50, 29000 ms), 726 (65, 43400 ms) | 0 | again 388 removed |
| nms050 | 7: duplicates 9, 453, 543, 592, 749; parts 726, 750 | part 550 on bus 27 (453, 28840 ms) | 0 | again 388 removed; adds 687 on vehicle 47, no truth visit, frame 1179 (47160 ms) |

By the declared wording, each containment step is a fix for 6 of the 8 second-box enters, contain080 and contain090 at
the cost of found vehicles (below), birth040 for 2, birth050 for 3 and nms050 for 7; no step adds a second-box enter.
The again enter, 388, is not a second box: every step removes it, the birth steps included, and it counts toward no
step's fix.

- Found and missed: contain080 finds 24 (cars 25 and 50 lost), contain090 25 (car 25 lost); every other step finds the
  baseline's 26. Every step misses car 34.
- Vehicle-frames lost (tables above). contain080: 368, 159 nested, among them 44 of car 25's (38 nested) and 58 of car
  50's (23 nested). contain090: 186, 84 nested, 40 of them car 25's (37 nested). Car 25's annotated box lies mostly
  inside bus 30's (Baseline); the two `_same` steps, which keep a box inside one of another detector class, lose none
  of car 25's. contain080_same: 133, 11 nested. contain090_same: 76, 9 nested, the most on vehicles 51 (14), 2 (12)
  and 28 (11). nms050: 125, 9 nested, and 1121 fit lost, the most on vehicles 27 (458), 7 (233) and 17 (151). The
  birth steps lose none.
- The zone rule, 30 px, against its own baseline (25 found, cars 11 and 34 missed, 5 extra: duplicates 543 and 749,
  parts 550, 726 and 750). Each containment step removes duplicate 543 and parts 550, 726 and 750, keeps a duplicate on
  bus 30 at 41760 ms and adds none; contain080 finds 23 (cars 25 and 50 lost), contain090 24 (car 25 lost), the `_same`
  steps 25. birth040 removes none of the 5 and adds none; birth050 removes parts 550 and 750; both find 25. nms050
  removes duplicates 543 and 749 and parts 726 and 750, keeps a part on bus 27 (its track 453, 29440 ms), adds its
  enter on vehicle 47 and finds 25. No step finds car 11 or car 34 under the zone rule.
- MTID checks that a step does no harm, not that the fix carries over. Every step finds 20 of the 21 annotated visits
  (vehicle 36 missed, as at the baseline) and matches all 14 labels. contain080 and contain090 take out both part
  enters on vehicle 23 and have 16 enters on no annotated box instead of 14, so judged extra stays 16; the `_same` steps
  and nms050 keep the baseline's 14 on no annotated box and 2 part enters on vehicle 23. birth040 and birth050 leave 5
  and 1 judged extra, all on no annotated box. With the zone rule every step finds 20 and has no judged extra. Per
  detection MTID loses 47, 35, 13, 8 and 9 vehicle-frames under contain080, contain090, contain080_same,
  contain090_same and nms050.

**What did not work.**

- contain080 and contain090 lose vehicles the baseline found, car 25 under both and car 50 under contain080, so they do
  not qualify, whatever second-box enters they remove.
- No step removes all 8 second-box enters. The winner keeps 2 of them, duplicates 9 and 749 (its tracks 26 on bus 27
  at 280 ms and 674 on bus 30 at 41760 ms); only nms050 removes those two, and it keeps part 550 on bus 27.
- The birth steps remove 2 and 3 of the 8.
- nms050 removes the most second-box enters, 7, but adds an enter on vehicle 47, which has no truth visit, so it ties
  the winner at 2 extra enters and loses on the tie-break. Its 1121 fit-lost vehicle-frames on MVI_40714 are a figure
  the rule does not read.
- No step finds car 34, and under the zone rule none finds car 11.
- On MTID neither the containment steps nor nms050 lower the judged extra enters (16 each). That was not asked of them:
  MTID has no baseline enter on a vehicle another track is on, and its 2 part enters are static.

## In the live service (opt-in)

The winner, contain090_same, needs a step between detection and tracking, which `model.track()` does not offer
(Results). The edge service (`services/edge/src/main.py`) now has that step, off by default. This section says what
each setting runs, and declares a live check on the MTID clip, the one `make up-video` streams (`media/sample.mp4`),
before any run of it.

**The flags**, set like the edge's other settings (`CONTAIN_SHARE=0.9 make up-video`):

- `CONTAIN_SHARE`: unset or 0 is off, compose's default. A share above 0 and up to 1 turns the filter on: a detection
  with that share or more of its own area inside a strictly higher-scoring detection of its frame is dropped.
- `CONTAIN_SAME_CLASS`: 1, compose's default, drops a box only inside one of its own detector class, as the `_same`
  steps do; 0 drops it inside any class.
- `CONTAIN_SHARE=0.9` with `CONTAIN_SAME_CLASS=1` is contain090_same. 0.9 is the declared step's share (Declared
  design), a parameter, not a measurement.
- Any other value of either flag stops the edge at start, before the model loads, with a message naming both flags: a
  typo does not run without the filter, or with one nobody asked for.

**Off, the default**, runs the edge as before: `model.track()` on the stream with the `TRACKER` yaml (compose sets
`bytetrack.yaml`), COCO car, bus and truck, class-agnostic NMS and no `conf`; each tracked box goes to the debounced
counter, and the event publish, the occupancy gauge and the heartbeat are unchanged. The one addition is a line in the
log at start, `contain: off`.

**On**, the edge logs at start `contain: on share=0.9 same_class=1 conf=0.1 tracker=bytetrack.yaml` (with the share
and class flag that are set) and creates one ByteTrack tracker, the replay's adapter
(`replay.trackers.create("bytetrack")`) at `bytetrack.yaml`'s defaults, once per process: every `BYTETracker`
restarts the track-ID counter they share. It runs `model.predict()` on the stream with `conf` 0.1, the same classes
and class-agnostic NMS, and no tracker. On each frame it turns the boxes into detections, numbered by the frames read
since start; drops second boxes with `replay.secondbox.contained`, the code the replay ran; and updates the tracker
with what is left, on every frame, empty ones included. The tracker's boxes go to the same counter, publish, gauge and
heartbeat as on the off path. At each heartbeat the edge also logs, at the start of a line:

```
contain ts_ms=<that frame's ts_ms> frames=<frames in the window> dets=<D> dropped=<X> filter_ms=<M>
```

The window is the heartbeat's, and its frame count is the heartbeat's fps numerator. D is the boxes `model.predict()`
gave in the window, after NMS and the class selection and before containment; X is how many of them containment
dropped (D less those kept); M is the milliseconds spent in `contained()`. The counts restart at each heartbeat.

**Why `conf` 0.1.** `model.track()` sets `conf` to 0.1 when none is passed, and `model.predict()` alone would take
0.25 (the pre-run gate in Reproduce prints both lines from the image's Ultralytics). Passing 0.1 keeps the on path at
the off path's threshold, which is also the one the replay's detections were dumped at. Neither path passes an NMS IoU:
both take Ultralytics' default, the 0.7 the replay's detections were dumped at (Declared design).

**`TRACKER` with the filter on.** The on path does not use `TRACKER`: it always runs `bytetrack.yaml`'s defaults through
the replay adapter. So that another yaml, a birth-score one say, is not dropped silently, the edge exits at start when
the filter is on and `TRACKER` is set to anything but `bytetrack.yaml`. With the filter off, `TRACKER` works as before.

**Nothing new leaves the edge.** No MQTT payload or topic, ingest table or Grafana panel is added or changed; the
filter's counts go to the edge's log only.

`harness/tests/test_edge.py` drives the edge's `main()` on the host, with the broker client, the ID generator and the
model stubbed and numpy blocked, as in CI. It pins the off path's `model.track()` call and its events, the on path's
`model.predict()` call and its one tracker, the tracker's input frame by frame against the replay's filter (with the
pure-Python IoU tracker standing in for ByteTrack), the heartbeat line's counts, the MQTT topics, payload keys, QoS and
retain on both paths, and the flags' rules.

### Declared before any run

This part was committed before any live run, and no run result changes it. Its thresholds, windows and run settings
are declared parameters, not measurements.

**Runs.** Three runs on the MTID clip, in the order off1, on, off2, so that drift over the session does not fall on
the on run alone. Each run is `make down`, then `make up-video` with `CONTAIN_SHARE` 0, 0.9 and 0 in turn,
`OCCUPANCY_GAUGE_MS=500` and `GRAFANA_PORT=3001`; everything else stays at compose's defaults: `CONTAIN_SAME_CLASS=1`,
`MIN_TRAVEL_PX=0`, `TRACKER=bytetrack.yaml`. The occupancy gauge is the run clock in all three runs (the window below
starts at its first stored sample) and is not reported; its sampler only reads the counter. `make down` removes the
containers and their logs, so each run's figures and log are saved before the next run starts.

**One image.** The edge image is built once, before off1 (`docker compose --profile video build edge`), and its ID is
recorded before the first run, at each run and after the last; `make up-video` builds every time, so a run is valid
only if its ID equals the one before off1 and the one after the last run (below). Nothing under `services/edge` or
`harness/replay` changes from this declaration's commit until the last run ends (off2, or a repeat after it), and no
other build or container runs alongside the lab. The clip's sha256 and its ffprobe frame count and rate are recorded.

**Gate, before the first run.** The runs do not start if either fails:

- the image reports Ultralytics 8.4.170 and torch 2.14.1+cpu, the versions the replay ran on (Results);
- `Model.track`'s source in the image sets `conf` to 0.1 when none is passed. Otherwise the on path's `conf` 0.1 would
  not equal the off path's.

**Window, the same for every run.**

- One loop of the clip, L, is 3199 frames at 30 fps (the clip's; the ffprobe line in Reproduce prints both), so three
  loops are 319900 ms.
- S0 is the run's first stored gauge sample, the least `zone_occupancy.ts_ms` for edge-01.
- T0 = S0 + 30000 ms, which skips the vehicles already in the zone when the edge joins the loop.
- The counted span is [T0, T0 + 319900 ms): three whole loops, so each run counts every part of the clip three times,
  wherever in the loop it joined.
- A run is read once its stored gauge samples reach S0 + 409900 ms: the 30 s skip, the span and a 60 s margin, the
  margin because an enter is stored only with its exit. The wait polls about every 10 s and gives up after 90 polls
  (TIMEOUT).

**Measures.**

- Enters and exits: `zone_events` rows for edge-01, counter `debounced`, with `ts_ms` in the span.
- fps: ingest keeps only each device's latest heartbeat (`device_status`), so the wait polls it about every 10 s. The
  polls are deduplicated by `updated_at`, the heartbeats with `updated_at` in the span are kept, and fps is their
  median.
- The filter's figures: the on run's `contain ts_ms=` lines with `ts_ms` in the span.
- `make counts` is not used: it counts every stored event, not the span.

**A valid run** meets all of these:

- its edge image ID equals the ones recorded before off1 and after the last run;
- its stored gauge samples reached S0 + 409900 ms (no TIMEOUT);
- the edge container's restart count is 0;
- its saved log holds exactly one `contain: ` startup line, the one its flags should print;
- no log line contains `unresponsive` (Ultralytics' warning for a stalled stream);
- it has at least 2 heartbeats in the span, consecutive ones at most 30 s apart;
- it logs `contain ts_ms=` lines if and only if it is the on run.

Every run is reported, invalid ones included. Validity never depends on enters, fps or drops. An invalid run is
repeated once, right away, as `<run>.2` with the same settings (`RUNS=<run>.2` in Reproduce), and both are reported;
(a) to (c), and the off runs' mean and range, use the valid run of each name. If the repeat is invalid too, the check
is reported as not completed. No other run is repeated, and no setting, threshold or window is changed, because of a
result.

**Pass**, declared now, on the on run:

- (a) Filter active: `dropped` summed over the span is above 0, and no heartbeat line has `dropped` above `dets`. The
  sum is over the span, not per heartbeat window, since a window can pass with no box inside another.
- (b) Throughput: both of these must hold.
  1. `filter_ms` summed over the span is under 0.01 of the span (3199 ms).
  2. Its median fps is at least 0.9 times the lower of the two off runs' medians. The bound is one-sided, so a faster
     on run passes.
- (c) Gross harm: its enters lie within 0.5 to 1.5 times the mean of off1's and off2's enters. This catches a broken
  path, one with no tracks or with IDs restarting so that boxes enter again; it is not a check that nothing changed.

Reported, not scored:

- whether the on run's enters lie within the off runs' range. With two off runs, a run that changes nothing need not
  land inside it;
- exits, and heartbeats in the span;
- the on run's `dropped` / `dets`, set beside the replay's for scale only: on MTID's saved detections at `conf` 0.1,
  contain090_same drops 467 of 20151, 0.0232 (Per detection, MTID, under Results: contain090_same's label counts sum to
  19684, the baseline's to 20151);
- the `filter_ms` share, the image ID and the versions.

A fail in (a), (b) or (c) is published as a fail, and the opt-in code stays merged, off by default. A fix for a code
defect is a new commit with a new declared set of runs; the first set stays reported.

### Live results

TBD until the runs. The cells will come from the live check's commands at the end of Reproduce, whose analysis prints
each run's figures and validity, then (a) to (c).

Before any row is read:

- the clip: sha256 TBD; frame count and rate TBD;
- the gate: Ultralytics and torch in the image TBD; `Model.track`'s `conf` line TBD;
- the edge image ID before off1 and after the last run: TBD.

| run | image ID (short) | valid | enters | exits | heartbeats in the span | median fps | dropped / dets | filter_ms share |
|---|---|---|---|---|---|---|---|---|
| off1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| on | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| off2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

A repeated run gets its own row, `<run>.2`, under the first.

- (a) filter active: TBD.
- (b) throughput: TBD.
- (c) gross harm: TBD.
- The on run's enters within the off runs' range (a fact, not scored): TBD.

### What the live check may claim

- MTID checks that the opt-in path runs and does no gross harm, not that the fix carries over.
- On MTID the replay gives contain090_same the baseline's counts (every frame, counter at its defaults): 39 enters; by
  vehicle 20 found, 16 judged extra, 3 not judged (Results). The live edge has not run on a clip where the filter
  removes an enter, so its effect on enters is measured in the replay harness only (MVI_40714: 6 of the 8 second-box
  enters).
- No live figure is scored against a replay figure:
  - the live edge infers on part of the stream's frames, since Ultralytics' stream reader keeps only the newest frame;
    the share is the run's measured median fps over the clip's 30;
  - it reads a libx264 re-encode of the clip, with unrounded scores;
  - ByteTrack's 30-frame lost buffer and the counter's 5- and 8-frame hysteresis (its defaults) count processed
    frames, so they span longer than in the replay;
  - the looped clip joins its last frame to its first.
- A difference between the on run and the off runs is not read as the filter's effect, and a count inside their range
  is not read as no effect.
- The live figures are read through (a) to (c) only; beyond them the check makes no claim about the filter's effect on
  counts.

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
  baseline ByteTrack enter is on it either. nms050's track 687 enters on it in the last frame (vehicle 47, Results),
  and that enter counts as an extra. Two annotated cars have their footpoint just inside the top edge for 15 and 17
  frames, also with no truth visit.
- The figures were measured on Ultralytics 8.4.170 and torch 2.14.1+cpu (Reproduce). The edge image installed both
  unpinned when they were measured and pins them since; the replay `bytetrack` adapter was written against an earlier
  release. A rebuild on other versions can change the detections and the tracks.
- Second boxes, per detection: a detection on a vehicle whose annotated box was dropped for an ignored region is labelled
  other. The annotated box is dropped by its own centre and the detection kept by its own, so the two can fall on
  either side of a region's edge.
- On MTID, part and only labels can be static objects the detector scores as vehicles (the phantom study's lane dashes
  and bins) lying under a passing vehicle's annotated box, which the rules cannot tell from a second box; MTID's part
  and only counts are not read as second boxes.
- The second-box steps are tried on one clip, with one detector and one tracker (ByteTrack); MTID and the queue fixture
  check only that a step does no harm.
- The found set puts no time limit on a found enter: it is the vehicle's first matching enter whenever it commits,
  where the time match allows 2 s. A step that delays a vehicle's enter keeps that vehicle found.
- The opt-in filter is checked live on MTID only (In the live service (opt-in)): three runs and a repeat of any invalid
  one, one edge image, CPU. On MTID the replay gives contain090_same the baseline's counts (every frame, counter at its
  defaults): 39 enters; by vehicle 20 found, 16 judged extra, 3 not judged (Results). So the check can show that the
  opt-in path runs and does no gross harm, not that the fix carries over.

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
# The second-box steps, run only after the declared design above was committed; they produce the
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
# NOT RUN YET: the live check declared under In the live service (opt-in), whose Live results stay TBD until
# it runs. From the repo root, with nothing else running in Docker.
# the opt-in second-box filter in the live edge on the MTID clip (media/sample.mp4): off, on, off on one image, each
# read 409.9 s after its first stored gauge sample (In the live service (opt-in)); GRAFANA_PORT=3001 when 3000 is taken
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,nb_frames media/sample.mp4
shasum -a 256 media/sample.mp4
mkdir -p harness/runs/live
docker compose --profile video build edge
docker image inspect edge-cv-lab-edge --format '{{.Id}}' > harness/runs/live/image.before
docker run --rm edge-cv-lab-edge python -c "import inspect, torch, ultralytics; from ultralytics.engine.model import Model; print(ultralytics.__version__, torch.__version__); [print(l.strip()) for l in (inspect.getsource(Model.predict) + inspect.getsource(Model.track)).splitlines() if 'conf' in l and ('custom' in l or 'kwargs[' in l)]"
q() { docker compose --profile video exec -T postgres psql -U postgres lab -At -F ' ' -c "$1"; }
# RUNS: the runs, space-separated, off1 on off2 unless set. An invalid run is repeated with RUNS=<run>.2, from this loop
# to the analysis (In the live service (opt-in)); a run's share comes from its name before any '.'. bash splits RUNS
# into words anyway, zsh only with shwordsplit
setopt shwordsplit 2>/dev/null || true
for run in ${RUNS:-off1 on off2}; do
  share=0; [ "${run%%.*}" = on ] && share=0.9
  make down && CONTAIN_SHARE=$share OCCUPANCY_GAUGE_MS=500 GRAFANA_PORT=3001 make up-video
  docker image inspect edge-cv-lab-edge --format '{{.Id}}' > harness/runs/live/$run.image
  i=0; until [ "$(q "select coalesce(max(ts_ms)-min(ts_ms),0) from zone_occupancy where device_id='edge-01'" 2>/dev/null || echo 0)" -ge 409900 ]; do
    i=$((i+1)); [ $i -ge 90 ] && { echo TIMEOUT $run; break; }
    q "select (extract(epoch from updated_at)*1000)::bigint, fps from device_status where device_id='edge-01' and state='online'" >> harness/runs/live/$run.fps 2>/dev/null
    sleep 10
  done
  q "select min(ts_ms), max(ts_ms) from zone_occupancy where device_id='edge-01'" > harness/runs/live/$run.span
  q "select kind, ts_ms, track_id from zone_events where device_id='edge-01' and counter='debounced' order by ts_ms" > harness/runs/live/$run.events
  docker inspect -f '{{.RestartCount}}' $(docker compose --profile video ps -q edge) > harness/runs/live/$run.restarts
  docker compose --profile video logs --no-log-prefix --no-color edge > harness/runs/live/$run.edge.log 2>&1
done
make down
docker image inspect edge-cv-lab-edge --format '{{.Id}}' > harness/runs/live/image.after
# per run, every runs/live/*.span (a repeat <run>.2 included): validity, enters and exits in [T0, T0 + 3 loops),
# median fps, the filter's counts; then (a) to (c) on the valid run of each name
cd harness && python -c "
import glob, os, statistics as st
L3 = 3 * 3199 * 1000 // 30  # 319900 ms: three loops of the 3199-frame, 30 fps clip
B = ('off1', 'on', 'off2')  # the names; a repeat <run>.2 is read under its name, the part before the .
WANT = {'off1': 'contain: off', 'on': 'contain: on share=0.9 same_class=1 conf=0.1 tracker=bytetrack.yaml', 'off2': 'contain: off'}
img = {open(f'runs/live/image.{w}').read().strip() for w in ('before', 'after')}
R = {}
for run in sorted((os.path.basename(f)[:-5] for f in glob.glob('runs/live/*.span')), key=lambda r: (B.index(r.split('.')[0]), r)):
    name, want = run.split('.')[0], WANT[run.split('.')[0]]
    p = f'runs/live/{run}'
    s0, s1 = map(int, open(p + '.span').read().split())
    t0 = s0 + 30000
    inside = lambda t: t0 <= t < t0 + L3
    ev = [l.split() for l in open(p + '.events').read().splitlines() if l.strip()]
    hb = sorted({(int(a), float(b)) for a, b in (l.split() for l in open(p + '.fps').read().splitlines() if l.strip())})
    hb = [(t, f) for t, f in hb if inside(t)]
    log = open(p + '.edge.log').read().splitlines()
    starts = [l for l in log if l.startswith('contain: ')]
    lines = [dict(kv.split('=') for kv in l.split()[1:]) for l in log if l.startswith('contain ts_ms=')]
    span = [d for d in lines if inside(int(d['ts_ms']))]
    r = dict(enters=sum(k == 'enter' and inside(int(t)) for k, t, _ in ev), exits=sum(k == 'exit' and inside(int(t)) for k, t, _ in ev),
             heartbeats=len(hb), fps_median=st.median(f for _, f in hb) if hb else 0.0,
             dets=sum(int(d['dets']) for d in span), dropped=sum(int(d['dropped']) for d in span),
             filter_ms=round(sum(float(d['filter_ms']) for d in span), 1))
    r['over'] = sum(int(d['dropped']) > int(d['dets']) for d in lines)
    r['drop_share'] = round(r['dropped'] / r['dets'], 4) if r['dets'] else 0.0
    r['filter_share'] = round(r['filter_ms'] / L3, 5)
    checks = {'image one ID': img == {open(p + '.image').read().strip()},
              'ran past t0 + 3 loops + 60 s': s1 - s0 >= 30000 + L3 + 60000,
              'no restart': int(open(p + '.restarts').read()) == 0,
              'one startup line, as set': starts == [want],
              'no stream unresponsive': not any('unresponsive' in l for l in log),
              'heartbeats <= 30 s apart': len(hb) > 1 and max(b[0] - a[0] for a, b in zip(hb, hb[1:])) <= 30000,
              'heartbeat lines only when on': bool(lines) == (name == 'on')}
    r['valid'] = all(checks.values())
    print(run, r, '| invalid:', [k for k, v in checks.items() if not v] or 'none')
    R[run] = r
use = {b: next((k for k in R if k.split('.')[0] == b and R[k]['valid']), None) for b in B}
print('scored, the valid run of each name:', use)
if None in use.values():
    print('check not completed: no valid run of', [b for b, k in use.items() if k is None])
else:
    on, offs = R[use['on']], [R[use['off1']], R[use['off2']]]
    mean_off = sum(o['enters'] for o in offs) / 2
    print('(a) filter active:', on['dropped'] > 0 and on['over'] == 0)
    print('(b) throughput: filter', on['filter_share'] < 0.01, '| median fps', on['fps_median'] >= 0.9 * min(o['fps_median'] for o in offs),
          '| pass', on['filter_share'] < 0.01 and on['fps_median'] >= 0.9 * min(o['fps_median'] for o in offs))
    print('(c) enters within 0.5 to 1.5 x the off runs mean', mean_off, ':', 0.5 * mean_off <= on['enters'] <= 1.5 * mean_off)
    print('fact, not scored: on enters inside the off runs range:', min(o['enters'] for o in offs) <= on['enters'] <= max(o['enters'] for o in offs))" && cd ..
# the replay's share, set beside the on run's for scale: contain090_same on MTID's saved detections at conf 0.1
cd harness && python -c "
from replay.detections import read_detections; from replay.secondbox import contained
d = read_detections('runs/dets.agnostic.jsonl'); k = contained(d, 0.9, same_class=True)
print('replay, MTID, contain090_same at conf 0.1: detections', len(d), '| dropped', len(d) - len(k), '| share', round((len(d) - len(k)) / len(d), 4))" && cd ..
```

## Citation

Wen, L., Du, D., Cai, Z., et al. (2020). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and
Tracking. Computer Vision and Image Understanding. https://doi.org/10.1016/j.cviu.2020.102907
