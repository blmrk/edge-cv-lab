# Handover

Last updated 2026-10-02. Everything below has been run unless marked otherwise.

## Status

| Piece | State |
|---|---|
| Replay harness, 233 tests (`make test`) | green locally; TrackEval tests skip without the `.cache/TrackEval` clone, the `bytetrack` adapter tests without numpy (CI installs `[dev]` only) |
| Fixtures (boundary jitter, shadow, 24-car traffic, 6-car queue) | reproducible, checked in |
| Visuals: compare.gif, trackers.gif, timeline, heatmap, trajectories, spacetime | generated from fixtures via `make visuals` / `make trackers` |
| Real-footage GIF `docs/footage/real-compare.gif` | `make footage`, from the per-class NMS detections |
| Tracker bench: greedy_iou, groundplane, ByteTrack (Ultralytics, from video or `bytetrack` on saved detections), MOT bridge, TrackEval runner | tested; **boxmot adapter untested** (needs torch) |
| Compose stack, sim profile, Grafana dashboard | booted and working; `GRAFANA_PORT=3001 make up` if 3000 is taken; opt-in occupancy gauge (`OCCUPANCY_GAUGE_MS=500`) checked on the sim and run live on the video edge (MTID clip, with and without `MIN_TRAVEL_PX=30`) |
| Delivery drills (`make drill`, `make broker-restart`) and their before runs (`SIM_QOS=0`, `DURABLE_SESSIONS=false`) | run on fresh labs; figures in `docs/case-study-delivery.md` |
| Video profile (MediaMTX + YOLO edge) | booted on the MTID intersection clip with the labelled zone; class-agnostic NMS |
| `tools/label.html` | used to draw the zone on the MTID clip; its visit-labelling flow is **untested on real footage** (visits came from contact-sheet passes and the MTID annotations) |
| `scripts/detrac_to_gt.py` | parses a real UA-DETRAC file (test sequence MVI_40714, `docs/datasets.md`); its `--zone` visit truth run on it for task C (27 visits, `docs/case-study-straddles.md`) |
| `replay.straddle` (straddles against annotated boxes) | run on MVI_40714 for the task C baseline (`docs/case-study-straddles.md`) |
| `replay.secondbox` (second boxes: per-detection labels, enters by vehicle, the steps) and `dump_detections.py --iou` | tested; run on the host on the saved baseline files of MVI_40714 and MTID only (`--tracks`, baseline labels), for the declared design in `docs/case-study-straddles.md`; `--dets` and the `--iou` dumps **not yet run** |

## Case studies

The repo's story is four field failures, each written up as its own case study under `docs/`; the phantom boxes' second
kind has a follow-up study of its own (task C):

| Case study | Question | State |
|---|---|---|
| Counting accuracy | Why do zone visit counts drift when the detector looks right frame by frame? | done: `docs/case-study-tracking.md` |
| Event delivery over a bad uplink | Does every event arrive exactly once through low bandwidth, latency, outages and broker restarts? | done: `docs/case-study-delivery.md` |
| Phantom boxes | Lane markings scored as vehicles, and boxes in the gap between vehicles side by side: do they get counted? | done: `docs/case-study-phantoms.md`; the second kind is found and counted, not fixed (task C) |
| Boxes straddling two vehicles (phantom boxes, second kind) | How often is a box across two side-by-side vehicles counted, measured against annotated boxes and visits? | baseline measured; the fix moved to second boxes, its design declared, its steps not run: `docs/case-study-straddles.md` (task C) |
| Zone enter/exit balance | Do enters and exits reconcile per zone, and what does a standing imbalance reveal? | done: `docs/case-study-balance.md` |

## Task queue, in order

### C. A fix for second boxes: one vehicle counted on two tracks (follow-up to boxes straddling two vehicles)
- Found on a dense expressway clip (`media/vecteezy-6434705.mp4`, metrics only): 2 of 52 zone enters sit on a box across
  two vehicles, by three visual passes. The zone rule keeps both; a birth score of 0.5 removes one and 9 other enters
  whose truth is unknown. `replay.between` flags candidates but misses track 451: in 130 of its 158 frames fewer than two
  other boxes cover a fifth of it without matching it, so there is no pair to bridge.
- Clip chosen: UA-DETRAC test sequence MVI_40714 (`media/UA-DETRAC/`, metrics only, logged in `media/SOURCES.md`). Fixed
  elevated view, the near-left carriageway queued several abreast. Its annotated boxes with track IDs give, without
  hand labels, visit truth from the annotated tracks (`detrac_to_gt.py --zone`) and straddles by a fixed rule on the
  annotated boxes (`replay.straddle`): a detection that matches no annotated box but lies across two. The rule
  overcounts as well as undercounts (the study's Limitations). The mp4 is frame-aligned with the XML (checked,
  `docs/datasets.md`).
- Done: the zone on the queued carriageway, clear of the ignored regions, chosen from the frames and the annotations
  only, without reading any detector output (`runs/MVI_40714.zone.json`, written in the study's Reproduce); visit truth
  from `detrac_to_gt.py --zone` (27 visits, 11 of them vehicles standing in the zone at the first frame); the straddle
  scorer `replay.straddle`; the baseline, `docs/case-study-straddles.md`. ByteTrack with the debounced counter: 35
  enters for 27 visits (26 matched, 9 false, 1 missed). Straddle at enter, by the scorer's rule: 1 of 35 (track 750),
  straddle share 0.5 or more: none; the same with the zone rule (1 of 30). That one enter is a second box on part of
  car 25, whose annotated box lies mostly inside bus 30's, so on this clip no enter sits on a box across two separate
  vehicles. 568 of the 35151 kept detections are straddles by the rule, 0.91 of them under ByteTrack's birth score of
  0.25. Looked at after the results, with the rule kept as fixed: 218 of the 568 put 0.2 of their area on each
  annotated box outside the other, the geometry of a box across two separate vehicles, though what they are has not
  been checked; 461 lie across one pair of buses that no ByteTrack box stands for, 182 of the 218 among them.
- What the baseline says: by vehicle alone, 8 of the 9 extra enters are a second box on one vehicle (5 a second track
  on a vehicle already tracked, 3 a box on part of one vehicle, track 750 among them) and none is a box across two
  vehicles. A straddle fix could remove track 750's enter; anything else it changes in the count comes from boxes it
  should not drop, so it would be judged per detection first: straddles it drops against annotated vehicles it leaves
  with no detection. `between.bridge`, the closest existing rule, flags 247 of the 568 straddles (233 on that pair of
  buses) and also 114 boxes that match an annotated vehicle; how many of those vehicles it would leave undetected is
  not measured. Track 750 is born on a box the rule calls a straddle, scoring 0.38, above the birth score, and the
  bridge test does not flag the box it enters with. The zone rule cannot be scored by time on this clip (it restamps
  the vehicles standing at the start); the study scores it by vehicle instead.
- Decision taken: the fix moved to the second-box failure behind 8 of the 9 extra enters (5 a second track on a vehicle
  already tracked, 3 a box on part of one). The straddle candidate is not pursued on this clip; closing C as a negative
  and looking for another clip were not taken. The design is declared in the study's Fix section, before any step runs:
  per-detection labels and enters put down by vehicle (`replay.secondbox`); seven steps, each one change against the
  ByteTrack baseline (a box 0.8 or 0.9 of its own area inside a higher-scoring box dropped, with and without the same
  class; track birth score 0.4 and 0.5; NMS IoU 0.5 instead of 0.7); and a winner rule (all 26 vehicles the baseline
  finds still found, fewer than 9 extra enters, no harm on MTID). The baseline figures it quotes are measured on the
  saved files on the host (the study's Reproduce); the Results tables are TBD.
- Next: commit the declared design, then run the steps in the edge image, the commands marked not run yet at the end
  of the study's Reproduce (no build, no download): dump both clips again at NMS IoU 0.7 and 0.5, compare the 0.7 dumps
  byte for byte with the saved detections (nms050 is dropped if either differs), run `replay.secondbox --dets` on both
  clips and diff the baseline step's tracks against the saved ones. Then fill in the Results tables, the winner or a
  negative result, and the per-enter account, in the wording the Fix section allows.
- The candidate straddle fix stays written down: drop a box mostly covered by two higher-scoring boxes, and check it
  keeps a car seen in the gap between two nearer ones; track 451 shows a straddle with fewer than two boxes around it,
  which that rule would not catch.

### Deferred
Not needed for the four case studies; kept in case a benchmark angle is wanted later.
- boxmot adapter: `pip install boxmot`, fix `update()` columns, add `boxmot_bytetrack` and `boxmot_ocsort` rows.
- Published tracker (FastTracker or UCMCTrack) through `replay.motformat` and `scripts/trackeval_run.py`.

### Done
1. Real clip in, video profile up: MTID intersection camera, `media/SOURCES.md`.
2. Labelled the clip: `harness/zone.json`, `harness/truth.json` (gitignored); 14 visits from a two-of-three vote.
3. Detections and baseline, with the per-step ablation (`replay.ablation`).
4. Real-footage GIF (`make footage`) and identity metrics against MTID-derived tracks (`scripts/mtid_to_gt.py`).
7. TrackEval on MTID (`scripts/trackeval_run.py`, TrackEval 12c8791 cloned to `.cache/`), not on DETRAC.
9. Counting case study final pass: no placeholders; the per-class NMS baseline is kept, class-agnostic NMS is a later step.
10. Repo polish: About, topics, social preview, Grafana screenshot.
11. Phantom boxes case study (task A): `min_travel_px` zone rule (default 0, off in the edge and the sim), `bytetrack`
    replay tracker, `replay.phantoms`, `replay.between`; `docs/case-study-phantoms.md`. CI actions bumped to v7, runner
    pinned to ubuntu-24.04.
12. Zone enter/exit balance case study (task B): `replay.balance` (open visits by reason; occupancy after the fact, live
    and as a gauge, against MTID's annotated tracks); opt-in `DebouncedZoneCounter(enter_after_dwell=True)` and
    `open_visits(now, seen_within_ms)`; `docs/case-study-balance.md`.
13. Occupancy gauge in the services, off by default: `replay.gauge.OccupancyGauge`; `OCCUPANCY_GAUGE_MS` (edge, sim) and
    `MIN_TRAVEL_PX` (edge) in compose; topic `occupancy/<device>` at QoS 0, ingest table `zone_occupancy`, Grafana panel
    "Vehicles in zone (gauge, opt-in)"; the event-sum panel relabelled: not occupancy, not a delivery check.

## Known rough edges
- The edge and the sim publish each debounced enter together with its exit, so the event sum on the dashboard (now
  "Enters minus exits, as received") shows neither occupancy nor delivery: a visit lost whole leaves it flat.
  Occupancy comes only from the opt-in gauge; `make delivery` checks delivery.
- The edge calls `counter.update()` only on frames with track boxes, so `expire()` waits for the next detected vehicle;
  on a quiet camera a lost visit stays open until then.
- `bytetrack` (the replay tracker) is written against `BYTETracker(args)` in ultralytics 8.4.163, the edge image's version
  on 2026-09-28; the image installs `ultralytics>=8.3` unpinned, so a rebuild can bring a release with another signature.
- The edge image now reports Ultralytics 8.4.170 (`docker run --rm edge-cv-lab-edge python -c "import ultralytics;
  print(ultralytics.__version__)"`), the version `docs/case-study-straddles.md` records. The phantom study's figures
  were measured on the version its doc names; the counting and balance studies' docs (`docs/case-study-tracking.md`,
  `docs/case-study-balance.md`) name none. None of them has been re-run.
- ByteTrack fails the queue fixture at its defaults (1 of 6 visits, 5 ID transfers): the phantom study's stopped-car
  check runs on `groundplane` for that reason.
- `DebouncedZoneCounter.min_travel_px` defaults to 0; the edge and the sim run without it. Its trade-off: a vehicle
  standing in the zone when its track starts is counted only once it moves.
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
