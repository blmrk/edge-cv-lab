# Case study: phantom boxes

> Status: measured on two public clips, one per kind of phantom. Every measured figure comes from a command in
> Reproduce; the visual passes behind the second kind are described where they are used.

## Problem

A zone counter counts tracks, and a track does not have to be a vehicle. The detector scores a lane marking as a car
at low confidence, the tracker keeps the box because it tracks low-score boxes on purpose, and the counter logs a visit
for an object that never moved. The [counting case study](case-study-tracking.md) found 18 such visits among ByteTrack's
39 on the intersection clip. This study measures fixes for them one at a time, and checks that each fix still counts a
real vehicle that stops in the zone. A second kind of phantom moves: in dense traffic a box lands across two vehicles
side by side and travels with them. The last section looks for it and asks whether it gets counted.

## Reproducing it without hardware

- Footage, zone, visit labels and annotated tracks: the counting study's MTID intersection clip (1024x640, 30 fps,
  106.63 s), its zone (`harness/zone.json`) and its 14 labelled visits (`harness/truth.json`); see
  [Reproducing it](case-study-tracking.md#reproducing-it-without-hardware) there. Clip logged in `media/SOURCES.md`.
- Detections: `yolov8n` on CPU in the edge image, class-agnostic NMS at `conf` 0.1, the dump scripts' and the edge's
  default: `runs/dets.agnostic.jsonl`, 20151 boxes.
- Tracker: ByteTrack as Ultralytics runs it (8.4.163 in the edge image). The fixes below put filters in front of it,
  which a video run cannot take, so this study feeds the saved detections to the same Ultralytics `BYTETracker`
  (`--tracker bytetrack`, `harness/replay/trackers/bytetrack.py`). Against the counting study's video run on the same
  detections, the replay gives 115 track IDs and 8850 boxes against 117 and 8858, the same counts (39 enters, 18
  static, 21 moving, moving F1 0.8), and TrackEval HOTA 0.225 against 0.224, IDF1 0.270 against 0.268, IDSW 8 against
  9. Saved detections are rounded to 0.1 px and 3 decimals, so the replay is close to the video run, not identical.
- Stopped vehicles: the queue fixture (`harness/fixtures/queue.*`), six synthetic cars that each drive in and stop at
  the same service window inside the zone. ByteTrack counts 1 of the 6 even at its defaults: it hands one track
  ID from car to car at the window (5 ID transfers, `replay.compare`). The ground-plane tracker counts all 6 with no
  transfer, so the queue check runs each fix's detection filter and zone rule on that tracker instead.

## Baseline

ByteTrack defaults on the class-agnostic detections, debounced counter: 39 enters against 14 labelled visits, 14
matched, 25 false, none missed, F1 0.528. `replay.score --explain` splits the enters in two groups with nothing
between: 18 tracks whose footpoint moved at most 5.8 px, with boxes on average 22.2 to 33.7 px wide and 13.8 to 24.0
px tall at mean scores 0.166 to 0.314, and 21 that moved at least 375.6 px. The 21 moving enters alone match all 14
visits with 7 false (F1 0.8). Drawn on a frame, the static group sits on the white dashes of the bike lane that crosses
the zone.

ByteTrack keeps them by design. Its defaults (`bytetrack.yaml` in the edge image) start a track from a box scoring at
least 0.25 (`new_track_thresh`) and keep it on boxes down to 0.1 (`track_low_thresh`) for up to 30 frames without one
(`track_buffer`); 12806 of the 20151 detections score under 0.3. A dash that scores 0.25 or more can start a track,
and its weaker boxes carry it from then on.

The scene holds other static boxes that the detector scores as vehicles, outside the zone: a parked car cut by the
frame's left edge and a group of bins or bollards at the corner of the crossing at the bottom of the frame. The
dataset's annotations leave both out (no annotated box overlaps either at IoU 0.3). Being outside the zone, neither is
counted, but both matter for one fix below.

## Fixes, one at a time

Each row changes one thing against the baseline (`replay.phantoms`, command in Reproduce). Static and moving split the
enters as `--explain` does (30 px); moving f1 scores the moving enters alone. Queue columns: of the queue fixture's six
stopped cars, how many the fix still counts, and its false visits; n/a for ByteTrack settings, which the ground-plane
tracker does not have (the queue's boxes score 0.6 to 0.95, above every threshold tried here).

| step | enters | static | moving | matched | false visits | missed visits | f1 | moving f1 | queue matched | queue false |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: ByteTrack defaults | 39 | 18 | 21 | 14 | 25 | 0 | 0.528 | 0.8 | 6 | 0 |
| zone rule: moved 30 px before an enter | 21 | 0 | 21 | 14 | 7 | 0 | 0.8 | 0.8 | 6 | 0 |
| detections under score 0.3 dropped | 27 | 6 | 21 | 14 | 13 | 0 | 0.683 | 0.8 | 6 | 0 |
| track birth score 0.4 (default 0.25) | 26 | 5 | 21 | 14 | 12 | 0 | 0.7 | 0.8 | n/a | n/a |
| track birth score 0.5 | 22 | 1 | 21 | 14 | 8 | 0 | 0.778 | 0.8 | n/a | n/a |
| boxes under 32 x 32 px dropped | 21 | 0 | 21 | 14 | 7 | 0 | 0.8 | 0.8 | 6 | 0 |
| static-box mask, boxes in 25% of frames | 30 | 9 | 21 | 14 | 16 | 0 | 0.636 | 0.8 | 3 | 1 |
| static-box mask, boxes in 10% of frames | 22 | 1 | 21 | 14 | 8 | 0 | 0.778 | 0.8 | 1 | 0 |

No fix loses a labelled visit: every row keeps the 21 moving enters and 14 matches, and only the static count falls. The same tracks
scored by TrackEval against the annotated tracks (`scripts/trackeval_run.py`, as in the counting study):

| step | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| baseline | 0.225 | 0.144 | 0.352 | 0.270 | -0.078 | 8 |
| zone rule, 30 px | 0.225 | 0.144 | 0.352 | 0.270 | -0.078 | 8 |
| score floor 0.3 | 0.201 | 0.127 | 0.317 | 0.245 | -0.042 | 13 |
| birth score 0.4 | 0.224 | 0.143 | 0.352 | 0.271 | -0.023 | 7 |
| birth score 0.5 | 0.220 | 0.139 | 0.351 | 0.266 | -0.009 | 3 |
| boxes under 32 x 32 px dropped | 0.232 | 0.153 | 0.353 | 0.287 | 0.009 | 7 |
| static-box mask, 25% | 0.249 | 0.176 | 0.352 | 0.321 | 0.157 | 8 |
| static-box mask, 10% | 0.253 | 0.183 | 0.352 | 0.331 | 0.194 | 8 |

**Zone rule** (`DebouncedZoneCounter(min_travel_px=30)`): an enter commits only once the track's footpoint has moved
30 px from where the track was first seen. 30 px is `--explain`'s static threshold, set before this study. It removes
all 18 phantom enters and changes nothing else: the tracks are the same, so TrackEval is too, and the six queue cars
still count because each drove in. On the counting study's video-run ByteTrack tracks it gives the same 21 enters and
F1 0.8 (`replay.score --min-travel-px 30`); on `greedy_iou:max_age=5` tracks it removes that tracker's 4 static enters,
26 to 22, F1 0.7 to 0.778. The trade-off: a vehicle already standing in the zone when its track starts counts only
once it has moved 30 px, stamped then, and one that never moves is never counted (both are tests in
`harness/tests/test_zones.py`). The rule sits in `harness/replay/zones.py`, which the edge and the sim share; both keep
the default, 0, which leaves the rule off.

**Boxes under 32 x 32 px dropped**, before tracking. 32 x 32 is the small-object bound of the COCO detection
benchmark, not tuned here. It also removes all 18, keeps the queue, and raises every TrackEval score (HOTA 0.225 to
0.232, MOTA -0.078 to 0.009). It depends on the view: the smallest moving
enter here averaged 72.0 x 67.1 px, but a camera further from the road sees real vehicles smaller than 32 x 32, and no
size filter can remove a large static box such as the bins at the crossing.

**Track birth score** 0.4 and 0.5 (`new_track_thresh`, default 0.25) leave 5 and 1 phantom enters. The tracks change
little (HOTA 0.224 and 0.220) and MOTA improves (-0.023 and -0.009 against -0.078): fewer phantom tracks start.

## What did not work

- **The static-box mask.** It learns boxes that recur at IoU 0.5 in at least 25% (or 10%) of the frames, and drops
  detections that overlap one. It is cross-fitted: each half of the clip is masked with boxes learned from the other
  half, so no frame is masked by boxes it helped learn, and no label is used. It scores best on TrackEval, partly for a
  wrong reason, and it drops stopped cars. Both masks learn the parked car at the frame's left edge: a real vehicle, but
  one the annotations leave out, so masking it counts as removing a false positive wherever it is detected. The 10%
  mask also learns the bins at the crossing. On the queue it counts 3 and 1 of the 6 cars: every car stops at the same window, so the window
  box recurs often enough to be learned, and each car is masked while it waits. The 25% mask also leaves 9 of the 18
  phantoms. Occupancy alone cannot tell a dash from a stopping point that every vehicle uses.
- **A score floor of 0.3 before ByteTrack.** It is what the harness trackers do (`min_score`), and it leaves 6
  phantoms. It also takes away the low-score boxes ByteTrack uses to carry a track through a weak stretch: HOTA 0.225 to
  0.201, ID switches 8 to 13.

## Second kind: a box between two vehicles side by side

The intersection clip has too little side-by-side traffic in its zone to show it: `replay.between`, which flags a
detection lying across two higher-scoring detections side by side, finds 15 such boxes in 14 of its 3199 frames, and
one of them in one of the 39 enters' tracks (1 of that track's 59 boxes).

So this part uses a second clip: Vecteezy 6434705, "Busy traffic on the highway" by Bondeto ae (Vecteezy Free License,
attribution required; logged in `media/SOURCES.md`), fetched from a Kaggle mirror. It is stock footage, not a mounted
camera: a fixed telephoto view from overpass height down a congested divided expressway, cars two and three abreast
with motorcycles between them, 1920x1080 at 50 fps, 30.0 s. It is used for metrics only; no frame of it is published.
It has no annotations and no visit labels, so what is measured is enters, not accuracy. Same detector settings (48844
boxes), ByteTrack replayed on them, and a zone fixed before counting: a band across the full width, y 360 to 900.

| | enters | enters on a straddling box (visual passes) | bridge boxes flagged |
|---|---|---|---|
| ByteTrack defaults | 52 | 2 | 1271, in 814 of 1500 frames, median score 0.158 |

The flags are candidates, not phantoms: they also catch a real vehicle seen in the gap between two nearer ones. Tracks
decide what gets counted, and only 6 of the 52 enters' tracks carry any flagged box, none of them for half its boxes.
So each enter was checked visually: `replay.between --sheet` crops every enter's box at the frame it entered, and three
model-assisted visual passes (an AI model reading the sheets, each pass run separately) labelled each crop, kept in the
gitignored `harness/runs/gap.passes.json` as the counting study keeps its labels in `truth.json`. A crop counts as a
straddle when at least two passes say so.

- 2 of the 52 enters sit on a box straddling two vehicles side by side: tracks 1033 (all three passes: the upper half of
  a box truck and the car beside it) and 451 (two passes: a hatchback and half of the SUV next to it, both of which
  are counted on their own tracks too). 49 are a single vehicle in all three passes; one splits three ways.
- The finder flags 28 of track 1033's 90 boxes and none of track 451's 158, and none of 451's even with its score test
  off (1033 would then get 75). Track 451 seldom has two separate boxes around it to bridge: in 130 of its 158
  frames fewer than two other boxes cover a fifth of it without matching it, often because it overlaps the hatchback's
  own box at IoU 0.5 or more (67 frames); in the other 28 the boxes around it are not side by side.

Flagged boxes are common in the detections and rare in the counts: they score low (median 0.158), so ByteTrack does not
start a track on most of them (birth score 0.25), and the debounced counter needs 5 frames inside and 1 s of dwell. Of
the two straddles that do get counted, the zone rule and a birth score of 0.4 stop neither; a birth score of 0.5 stops
one:

| run | enters | track 451's straddle counted | track 1033's straddle counted |
|---|---|---|---|
| ByteTrack defaults | 52 | yes | yes |
| zone rule, 30 px | 47 | yes | yes |
| track birth score 0.4 | 48 | yes | yes |
| track birth score 0.5 | 42 | yes | no |

"Counted" means a counted track follows the straddling box (IoU 0.5) for at least a quarter of its frames and spends at
least half of its own boxes on it, so the track of a vehicle under the straddle does not qualify. The zone
rule cannot see them, since they move with traffic; the size floor and the static mask are built for small or static
boxes, and these are neither. The birth score 0.5 removes the weaker one and 9 other enters, which may be missed
vehicles or double counts: with no visit labels, this clip cannot say. A fix for this kind (for example, dropping a box
mostly covered by two higher-scoring boxes) needs a dense clip with visit labels to be measured; it is left open.

## Limitations

- One clip: one camera, daytime, 106.6 s, 14 labelled visits. The fixes are compared on 18 phantom enters.
- The fixes run on replayed ByteTrack, not the video pipeline: same counts on the baseline, IDs and boxes close but not
  identical (Reproducing it, above). The zone rule, which does not touch the tracks, is also measured on the video run.
- Parameters: 30 px and 32 x 32 px were fixed before measuring, and the birth score 0.4 was planned. The 0.5 value and
  the mask's two shares were picked after seeing this clip's detections (how often boxes recur), never its labels.
- The stopped-car check is synthetic (the queue fixture) and runs on the ground-plane tracker, because ByteTrack fails
  the queue at its defaults.
- Visits are matched on enter time alone (2 s tolerance), as in the counting study.
- The second kind rests on one 30 s stock clip with no labels: enters are counted, not scored, and the band's 52 enters
  are not 52 vehicles (some vehicles change track ID inside it). Which enter is a straddle comes from three AI visual
  passes over one crop per enter, not from annotations.

## Reproduce

```bash
setopt interactive_comments 2>/dev/null || true
# scoring and sheets run locally; detection and ByteTrack run in the edge image, CPU only (make up-video builds it)
pip install -e "harness[dev,trackeval,viz]"
e() { docker run --rm -v "$PWD":/work -w /work/harness edge-cv-lab-edge "$@"; }
e python scripts/dump_detections.py --video ../media/sample.mp4 --model /app/yolov8n.pt --out runs/dets.agnostic.jsonl
e python scripts/dump_tracks.py --video ../media/sample.mp4 --model /app/yolov8n.pt --tracker bytetrack.yaml --out runs/bytetrack.agnostic.jsonl
e python -c "import ultralytics; print(ultralytics.__version__)"
e python -c "from ultralytics.utils import YAML; from ultralytics.utils.checks import check_yaml; print(YAML.load(check_yaml('bytetrack.yaml')))"
# ByteTrack on saved detections (prints its IDs and boxes), to compare with the video run
e python -m replay.track --dets runs/dets.agnostic.jsonl --tracker bytetrack --out runs/bytetrack.replay.jsonl
# the queue: ByteTrack against the ground-plane tracker
e python -m replay.compare --dets fixtures/queue.dets.jsonl --zone fixtures/zone.json --truth fixtures/queue.truth.json \
  --gt fixtures/queue.gt.jsonl --trackers bytetrack groundplane
# the fixes, one row each; each step's tracks go to runs/phantoms/ for TrackEval
e python -m replay.phantoms --dets runs/dets.agnostic.jsonl --zone zone.json --truth truth.json --out-dir runs/phantoms
cd harness
python -c "import json; t = [json.loads(l) for l in open('runs/bytetrack.agnostic.jsonl')]; print(len({d['track_id'] for d in t}), len(t))"
python -m replay.score --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --truth truth.json --explain
python -m replay.score --tracks runs/bytetrack.replay.jsonl --zone zone.json --truth truth.json --explain
python -m replay.score --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --truth truth.json
python -c "from replay.detections import read_detections as r; d = list(r('runs/dets.agnostic.jsonl')); print(len(d), sum(x.score < 0.3 for x in d))"
python -c "from replay.detections import read_detections as r; s = [x.score for x in r('fixtures/queue.dets.jsonl')]; print(min(s), max(s))"
python scripts/trackeval_run.py --gt runs/mtid.gt.jsonl --tracks baseline=runs/phantoms/baseline.jsonl \
  travel30=runs/phantoms/travel30.jsonl floor030=runs/phantoms/floor030.jsonl birth040=runs/phantoms/birth040.jsonl \
  birth050=runs/phantoms/birth050.jsonl area1024=runs/phantoms/area1024.jsonl mask025=runs/phantoms/mask025.jsonl \
  mask010=runs/phantoms/mask010.jsonl video_run=runs/bytetrack.agnostic.jsonl
# the zone rule on the video-run tracks and on greedy_iou:max_age=5
python -m replay.score --tracks runs/bytetrack.agnostic.jsonl --zone zone.json --truth truth.json --min-travel-px 30
python -m replay.track --dets runs/dets.agnostic.jsonl --tracker greedy_iou --param max_age=5 --out runs/greedy_iou_5.agnostic.jsonl
for px in 0 30; do
  python -m replay.score --tracks runs/greedy_iou_5.agnostic.jsonl --zone zone.json --truth truth.json --explain --min-travel-px $px
  python -m replay.score --tracks runs/greedy_iou_5.agnostic.jsonl --zone zone.json --truth truth.json --min-travel-px $px
done
# the boxes each mask learns, per half; annotated boxes on the parked car and on the bins
python -c "
from replay.detections import read_detections as r; from replay.phantoms import static_boxes as s
d = r('runs/dets.agnostic.jsonl'); mid = (min(x.frame for x in d) + max(x.frame for x in d) + 1) // 2
for share in (0.25, 0.10):
    for first in (True, False):
        print(share, 'first' if first else 'second', [tuple(round(v) for v in b) for b in s([x for x in d if (x.frame < mid) == first], share)])"
python -c "from replay.schema import read_tracks; from replay.trackers.greedy_iou import iou
g = list(read_tracks('runs/mtid.gt.jsonl')); print(sum(iou(b.bbox, (0, 117, 42, 162)) >= 0.3 for b in g), sum(iou(b.bbox, (687, 526, 792, 638)) >= 0.3 for b in g))"
# second kind: bridge boxes on the intersection clip, then the expressway clip (media/SOURCES.md)
python -m replay.between --dets runs/dets.agnostic.jsonl --tracks runs/bytetrack.agnostic.jsonl --zone zone.json
cd .. && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration media/vecteezy-6434705.mp4
e python scripts/dump_detections.py --video ../media/vecteezy-6434705.mp4 --model /app/yolov8n.pt --out runs/gap.dets.jsonl
e python -m replay.track --dets runs/gap.dets.jsonl --tracker bytetrack --out runs/gap.bytetrack.jsonl
for t in 0.4 0.5; do
  e python -m replay.track --dets runs/gap.dets.jsonl --tracker bytetrack --param new_track_thresh=$t --out runs/gap.bytetrack.birth$t.jsonl
done
cd harness
echo '{"polygon": [[0, 360], [1920, 360], [1920, 900], [0, 900]], "frame_size": [1920, 1080], "video": "vecteezy-6434705.mp4"}' > runs/gap.zone.json
# every enter, its flagged boxes, and the sheets for the visual passes (runs/gap.sheet-1.jpg, -2.jpg; never published)
python -m replay.between --dets runs/gap.dets.jsonl --tracks runs/gap.bytetrack.jsonl --zone runs/gap.zone.json \
  --sheet runs/gap.sheet.jpg --video ../media/vecteezy-6434705.mp4
python -m replay.between --dets runs/gap.dets.jsonl --tracks runs/gap.bytetrack.jsonl --zone runs/gap.zone.json --min-travel-px 30
for t in 0.4 0.5; do
  python -m replay.between --dets runs/gap.dets.jsonl --tracks runs/gap.bytetrack.birth$t.jsonl --zone runs/gap.zone.json
done
# which runs still count the two straddles (tracks 451 and 1033 of the default run)
python -c "
import json; from collections import Counter
from replay.schema import read_tracks; from replay.trackers.greedy_iou import iou; from replay.zones import DebouncedZoneCounter, run
poly = [tuple(p) for p in json.load(open('runs/gap.zone.json'))['polygon']]
base = list(read_tracks('runs/gap.bytetrack.jsonl'))
for name, px in (('runs/gap.bytetrack.jsonl', 0), ('runs/gap.bytetrack.jsonl', 30), ('runs/gap.bytetrack.birth0.4.jsonl', 0), ('runs/gap.bytetrack.birth0.5.jsonl', 0)):
    tr = sorted(read_tracks(name), key=lambda b: (b.frame, b.track_id))
    entered = {e.track_id for e in run(DebouncedZoneCounter(poly, min_travel_px=px), tr) if e.kind == 'enter'}
    size = Counter(b.track_id for b in tr)
    out = {}
    for ref in (451, 1033):
        box = {b.frame: b.bbox for b in base if b.track_id == ref}
        on = Counter(b.track_id for b in tr if b.frame in box and iou(b.bbox, box[b.frame]) >= 0.5)
        out[ref] = [(t, n, size[t]) for t, n in on.items() if t in entered and n >= 0.25 * len(box) and n >= 0.5 * size[t]]
    print(name, 'min_travel', px, 'enters', len(entered), 'counted tracks on each straddle (id, frames on it, frames):', out)"
# why the finder misses track 451: its boxes with the score test on and off, and what surrounds them
python -c "
from collections import Counter, defaultdict; from dataclasses import replace
from replay.between import bridge, _inter, _area; from replay.detections import read_detections
from replay.schema import read_tracks; from replay.trackers.greedy_iou import iou
by = defaultdict(list)
for d in read_detections('runs/gap.dets.jsonl'): by[d.frame].append(d)
tracks = list(read_tracks('runs/gap.bytetrack.jsonl'))
for tid in (451, 1033):
    c = Counter()
    for t in (t for t in tracks if t.track_id == tid):
        src = max(by[t.frame], key=lambda d: iou(d.bbox, t.bbox))
        near = [d for d in by[t.frame] if d is not src and _inter(src.bbox, d.bbox) >= 0.2 * _area(src.bbox)]
        c['boxes'] += 1
        c['flagged'] += bridge(src, by[t.frame]) is not None
        c['flagged, score test off'] += bridge(replace(src, score=-1.0), by[t.frame]) is not None
        c['on one vehicle (IoU >= 0.5 with another box)'] += any(iou(src.bbox, d.bbox) >= 0.5 for d in near)
        k = sum(iou(src.bbox, d.bbox) < 0.5 for d in near)
        c['fewer than two other boxes cover a fifth of it' if k < 2 else 'two or more do, but no side-by-side pair'] += bridge(replace(src, score=-1.0), by[t.frame]) is None
    print(tid, dict(c))"
# the three visual passes (gitignored labels, one entry per enter of the default run)
python -c "import json; e = json.load(open('runs/gap.passes.json'))['enters']; print(len(e), 'enters | straddle, 2+ of 3:', [(x['track_id'], x['labels'].count('straddle')) for x in e if x['labels'].count('straddle') >= 2], '| single in all 3:', sum(x['labels'] == ['single'] * 3 for x in e), '| other:', [(x['track_id'], x['labels']) for x in e if x['labels'] != ['single'] * 3 and x['labels'].count('straddle') < 2])"
```
