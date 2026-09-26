# Handover

Last updated 2026-09-26. Everything below has been run unless marked otherwise.

## Status

| Piece | State |
|---|---|
| Replay harness, 92 tests (`make test`) | green locally; TrackEval tests skip without the `.cache/TrackEval` clone |
| Fixtures (boundary jitter, shadow, 24-car traffic, 6-car queue) | reproducible, checked in |
| Visuals: compare.gif, trackers.gif, timeline, heatmap, trajectories, spacetime | generated from fixtures via `make visuals` / `make trackers` |
| Real-footage GIF `docs/footage/real-compare.gif` | `make footage`, from the per-class NMS detections |
| Tracker bench: greedy_iou, groundplane, ByteTrack (Ultralytics), MOT bridge, TrackEval runner | tested; **boxmot adapter untested** (needs torch) |
| Compose stack, sim profile, Grafana dashboard | booted and working; `GRAFANA_PORT=3001 make up` if 3000 is taken |
| Delivery drills (`make drill`, `make broker-restart`) and their before runs (`SIM_QOS=0`, `DURABLE_SESSIONS=false`) | run on fresh labs; figures in `docs/case-study-delivery.md` |
| Video profile (MediaMTX + YOLO edge) | booted on the MTID intersection clip with the labelled zone; class-agnostic NMS |
| `tools/label.html` | used to draw the zone on the MTID clip; its visit-labelling flow is **untested on real footage** (visits came from contact-sheet passes and the MTID annotations) |
| `scripts/detrac_to_gt.py` | tested on a synthetic XML only, **not on a real UA-DETRAC file** (task deferred) |

## Case studies

The repo's story is four field failures, each written up as its own case study under `docs/`:

| Case study | Question | State |
|---|---|---|
| Counting accuracy | Why do zone visit counts drift when the detector looks right frame by frame? | done: `docs/case-study-tracking.md` |
| Event delivery over a bad uplink | Does every event arrive exactly once through low bandwidth, latency, outages and broker restarts? | done: `docs/case-study-delivery.md` |
| Phantom boxes | Lane markings scored as vehicles, and boxes in the gap between vehicles side by side: do they get counted? | next (task A) |
| Zone enter/exit balance | Do enters and exits reconcile per zone, and what does a standing imbalance reveal? | not started (task B) |

## Task queue, in order

### A. Phantom boxes case study
- The counting study found the first kind already: 18 of ByteTrack's 39 debounced enters on the MTID clip are tracks
  that never moved, on the bike lane's white dashes inside the zone (`replay.score --explain`). ByteTrack keeps them
  because it tracks boxes down to score 0.1; the harness trackers drop boxes under 0.3 (`min_score`).
- Candidate fixes, each measured as its own step like `replay.ablation`: a track must move before it can enter
  (zone logic, so a test in `harness/tests/` first), a higher score to start a track, a minimum box size, a mask of
  static detections learned from the empty scene. Watch that a real vehicle stopped in the zone still counts.
- Find the second kind (a box between two vehicles side by side) on a public CCTV-angled clip; log it in
  `media/SOURCES.md`.
- Done: `docs/case-study-phantoms.md` with a before/after table and the counting study's phantom paragraph pointing to it.

### B. Zone enter/exit balance case study
- Starting point: the naive counter's net balance (enters minus exits) is 66 on ByteTrack, the debounced counter's 3
  (`replay.cli`). Work out what a standing imbalance means per zone and how to reconcile it.
- Done: `docs/case-study-balance.md` with measured figures and commands.

### Deferred
Not needed for the four case studies; kept in case a benchmark angle is wanted later.
- boxmot adapter: `pip install boxmot`, fix `update()` columns, add `boxmot_bytetrack` and `boxmot_ocsort` rows.
- UA-DETRAC sequence: a second, research-licensed dataset (metrics and citation only, no frames); multi-GB download.
- Published tracker (FastTracker or UCMCTrack) through `replay.motformat` and `scripts/trackeval_run.py`.

### Done
1. Real clip in, video profile up: MTID intersection camera, `media/SOURCES.md`.
2. Labelled the clip: `harness/zone.json`, `harness/truth.json` (gitignored); 14 visits from a two-of-three vote.
3. Detections and baseline, with the per-step ablation (`replay.ablation`).
4. Real-footage GIF (`make footage`) and identity metrics against MTID-derived tracks (`scripts/mtid_to_gt.py`).
7. TrackEval on MTID (`scripts/trackeval_run.py`, TrackEval 12c8791 cloned to `.cache/`), not on DETRAC.
9. Counting case study final pass: no placeholders; the per-class NMS baseline is kept, class-agnostic NMS is a later step.
10. Repo polish: About, topics, social preview, Grafana screenshot.

## Known rough edges
- The counting study's baseline, ablation tables, first tracker table, the README's 78/39/21 and `real-compare.gif`
  use per-class NMS detections (`--per-class-nms`, runs/dets.jsonl). The dump scripts and the edge default to
  class-agnostic NMS.
- The sim and the edge ignore `publish()`'s return code: when the bounded outgoing store (5000) is full, events past
  the limit are dropped with no log line. Not reached in the drills.
- Ingest does not count the duplicates its `ON CONFLICT` turns away, so redeliveries are not measured.
- The harness's `id_switches` and TrackEval's IDSW differ by definition (greedy per-frame matching against CLEAR's
  continuity-first matching); publish TrackEval's.
- `scripts/trackeval_run.py` restores `np.int` / `np.float` / `np.bool` for TrackEval 12c8791; drop it once TrackEval
  is fixed upstream.
- The edge logs `Waiting for stream 0` whenever inference catches up with the RTSP stream. Log noise, not a stall.
- `DebouncedZoneCounter.margin_px=10` was set against the synthetic ±6 px edge jitter in a 1280 px frame. On a real
  clip, check a parked vehicle's footpoint jitter and draw the zone so stops sit more than `margin_px` inside it.
- `sim` service uses `SPEED=2`; Grafana FPS panel shows ~60. Not a bug.
- `groundplane` gate defaults (`gate_along=120`, `gate_across=40`) were tuned on the synthetic queue at
  10 fps and 1280 px wide. Real clips at 25 to 30 fps need smaller gates per frame.
- `DebouncedZoneCounter.lost_ms=3000` closes a visit whose track went silent. On real footage with
  detector flicker inside the zone this may close a visit early and count the vehicle again when its track returns; watch `false_visits`.

## Kick-off prompt for Claude Code
> Read CLAUDE.md and HANDOVER.md. Start at the first open task. Run `make test` first and keep it green. Commit after
> each task with a conventional prefix. Never write an unmeasured number into docs. Stop and ask me
> before downloading anything larger than 500 MB or installing anything that needs a GPU.
