# Handover

Last updated 2026-10-05. Everything below has been run unless marked otherwise.

## Status

| Piece | State |
|---|---|
| Replay harness, 268 tests (`make test`) | green locally; TrackEval tests skip without the `.cache/TrackEval` clone, the `bytetrack` adapter tests without numpy (CI installs `[dev]` only) |
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
| `replay.secondbox` (second boxes: per-detection labels, enters by vehicle, the steps) and `dump_detections.py --iou` | tested; run in the edge image on MVI_40714 and MTID, every step (`--dets`, both clips dumped again at NMS IoU 0.7 and 0.5), and on the host on the saved baseline files (`--tracks`); results in `docs/case-study-straddles.md` |
| Opt-in second-box filter in the edge (`CONTAIN_SHARE=0.9` with `CONTAIN_SAME_CLASS=1` is contain090_same, before ByteTrack; off by default) | tested on the host (`harness/tests/test_edge.py`); checked live on MTID only: off, on, off on one image, every run valid, (a) to (c) pass (`docs/case-study-straddles.md`, In the live service (opt-in)); its effect on enters measured in the replay harness only |

## Case studies

The repo's story is four field failures, each written up as its own case study under `docs/`; the phantom boxes' second
kind has a follow-up study of its own (task C, closed):

| Case study | Question | State |
|---|---|---|
| Counting accuracy | Why do zone visit counts drift when the detector looks right frame by frame? | done: `docs/case-study-tracking.md` |
| Event delivery over a bad uplink | Does every event arrive exactly once through low bandwidth, latency, outages and broker restarts? | done: `docs/case-study-delivery.md` |
| Phantom boxes | Lane markings scored as vehicles, and boxes in the gap between vehicles side by side: do they get counted? | done: `docs/case-study-phantoms.md`; the second kind is measured in `docs/case-study-straddles.md` |
| Boxes straddling two vehicles (phantom boxes, second kind) | How often is a box across two side-by-side vehicles counted, measured against annotated boxes and visits? | done: `docs/case-study-straddles.md`; straddles are a measured negative on the clip, the fix moved to second boxes, and by a rule declared in advance contain090_same wins, in the replay harness only (task C, closed); opt-in in the edge since 2026-10-05, checked live on MTID only |
| Zone enter/exit balance | Do enters and exits reconcile per zone, and what does a standing imbalance reveal? | done: `docs/case-study-balance.md` |

## Task queue, in order

Empty. Task C closed on 2026-10-05 on the replay result (Done 14 and 15); what was not taken is in Deferred.

### Deferred
Not needed for the four case studies; kept in case a benchmark angle is wanted later.
- boxmot adapter: `pip install boxmot`, fix `update()` columns, add `boxmot_bytetrack` and `boxmot_ocsort` rows.
- Published tracker (FastTracker or UCMCTrack) through `replay.motformat` and `scripts/trackeval_run.py`.
- The second-box steps on another annotated clip: the result is one clip, one detector, one tracker.
- The candidate straddle fix (drop a box mostly covered by two higher-scoring boxes, keeping a car seen in
  the gap between two nearer ones): no enter on MVI_40714 sits on a box across two separate vehicles, so
  it has nothing to score there; track 451 on the expressway clip shows a straddle with fewer than two
  boxes around it, which that rule would not catch.

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
14. Straddles (task C, first half): UA-DETRAC test sequence MVI_40714 (metrics only), a zone from the frames and the
    annotations, visit truth from `detrac_to_gt.py --zone`, the straddle scorer `replay.straddle` (rule fixed before
    any result). By the rule 1 of 35 enters sits on a straddle, and that box is on part of one car whose annotated box
    lies inside a bus's: no enter on the clip sits on a box across two separate vehicles. 8 of the 9 extra enters are a
    second box on one vehicle. `docs/case-study-straddles.md`, Baseline.
15. Second boxes (task C, second half): `replay.secondbox` (detections and enters put down to annotated vehicles; the
    containment filter; seven steps), `dump_detections.py --iou`, `phantoms.table(steps=)`. Design declared and committed
    before any step ran; an independent re-derivation agreed on every table. By the declared rule contain090_same wins,
    measured in the replay harness only: extra enters 9 to 2, 6 of the 8 second-box enters removed, none added, every
    vehicle the baseline found still found. `docs/case-study-straddles.md`, Fix and Results.
16. The second-box filter in the edge, opt-in and off by default: `CONTAIN_SHARE` and `CONTAIN_SAME_CLASS` in compose
    (`CONTAIN_SHARE=0.9` is contain090_same). On, the edge runs `model.predict()` at `conf` 0.1, drops second boxes
    with `replay.secondbox.contained`, then updates one replay `bytetrack` tracker on every frame; off, it runs
    `model.track()` as before. `harness/tests/test_edge.py` drives `main()` on the host. A live check, declared and
    committed before any run, ran on 2026-10-05 on MTID only: off, on, off on one image, CPU; every run valid, and (a)
    filter active, (b) throughput and (c) gross harm pass. It checks that the opt-in path runs and does no gross harm,
    not that the fix carries over: on MTID the replay gives contain090_same the baseline's counts, the live edge has not
    run on a clip where the filter removes an enter, and the filter's effect on enters is measured in the replay harness
    only. `docs/case-study-straddles.md`, In the live service (opt-in).

## Known rough edges
- The edge and the sim publish each debounced enter together with its exit, so the event sum on the dashboard (now
  "Enters minus exits, as received") shows neither occupancy nor delivery: a visit lost whole leaves it flat.
  Occupancy comes only from the opt-in gauge; `make delivery` checks delivery.
- The edge calls `counter.update()` only on frames with track boxes, so `expire()` waits for the next detected vehicle;
  on a quiet camera a lost visit stays open until then.
- With `CONTAIN_SHARE` on, `TRACKER` is not used: the on path always runs `bytetrack.yaml`'s defaults through the
  replay adapter, and the edge exits at start when `TRACKER` is set to anything but `bytetrack.yaml`.
- The filter's live check ran on MTID only, where the replay gives contain090_same the baseline's counts: it shows the
  opt-in path runs and does no gross harm, not that the fix carries over (`docs/case-study-straddles.md`).
- `bytetrack` (the replay tracker) is written against `BYTETracker(args)` in ultralytics 8.4.163, the edge image's version
  on 2026-09-28, and runs on 8.4.170, the version the image pins since 2026-10-05 with torch 2.14.1. The pinned image
  was rebuilt on 2026-10-05 (`docker compose --profile video build edge`): it reports 8.4.170 and 2.14.1+cpu, and a
  detector dump of MVI_40714 and a ByteTrack replay on the saved detections in it are byte-identical to the saved
  `runs/MVI_40714.dets.jsonl` and `runs/MVI_40714.bytetrack.jsonl` (`cmp`; `diff` of the sorted files), the model
  sha256 unchanged.
- The edge image reports Ultralytics 8.4.170 (`docker run --rm edge-cv-lab-edge python -c "import ultralytics;
  print(ultralytics.__version__)"`), the version `docs/case-study-straddles.md` records and `services/edge` now pins
  (`requirements.txt`, `Dockerfile`). The phantom study's figures
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
