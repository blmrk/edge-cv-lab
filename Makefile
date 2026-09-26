.PHONY: test fixtures visuals trackers footage up up-video down logs counts delivery drill broker-restart
ALL = --profile sim --profile video
Z = fixtures/zone.json
T = fixtures/traffic.jsonl
test:      ; cd harness && python -m pytest -q
fixtures:  ; cd harness && python scripts/make_fixtures.py
visuals:   ## regenerate the fixture images in docs/img (make trackers makes trackers.gif and spacetime.png)
	cd harness && python -m replay.viz compare --tracks $(T) --zone $(Z) --out ../docs/img/compare.gif --seconds 18 \
	 && python -m replay.viz heatmap --tracks $(T) --zone $(Z) --out ../docs/img/heatmap.png \
	 && python -m replay.viz trajectories --tracks $(T) --zone $(Z) --out ../docs/img/trajectories.png \
	 && python -m replay.viz timeline --tracks $(T) --zone $(Z) --out ../docs/img/timeline.png
trackers:  ## tracker comparison table + space-time diagram on the queue fixture
	cd harness && mkdir -p runs \
	 && python -m replay.compare --dets fixtures/queue.dets.jsonl --zone $(Z) --truth fixtures/queue.truth.json \
	      --gt fixtures/queue.gt.jsonl --trackers greedy_iou:max_age=5 greedy_iou greedy_iou:max_age=100 groundplane \
	 && python -m replay.track --dets fixtures/queue.dets.jsonl --tracker greedy_iou --out runs/iou.jsonl \
	 && python -m replay.track --dets fixtures/queue.dets.jsonl --tracker groundplane --out runs/ground.jsonl \
	 && python -m replay.spacetime --tracks "greedy_iou (3 s buffer)=runs/iou.jsonl" groundplane=runs/ground.jsonl \
	      --band 440 560 --out ../docs/img/spacetime.png \
	 && python -m replay.trackviz --dets fixtures/queue.dets.jsonl --gt fixtures/queue.gt.jsonl --zone $(Z) \
	      --trackers "greedy_iou:max_age=5" greedy_iou groundplane \
	      --labels "IoU tracker, 0.5 s buffer" "IoU tracker, 3 s buffer" "ground-plane tracker" \
	      --occluder 440 560 --every 3 --fps 10 --width 800 --out ../docs/img/trackers.gif
# footage needs media/sample.mp4, harness/zone.json (drawn in tools/label.html) and harness/runs/dets.jsonl, all
# gitignored: docs/case-study-tracking.md, Reproduce, makes the detections (the --per-class-nms run). 20-33 s holds 5 of the 14 labelled visits in harness/truth.json.
footage:   ## tracker comparison over the real clip, not the fixture: docs/footage/real-compare.gif
	cd harness && mkdir -p ../docs/footage \
	 && python -m replay.trackviz --video ../media/sample.mp4 --dets runs/dets.jsonl --zone zone.json \
	      --trackers greedy_iou "greedy_iou:max_age=5" \
	      --labels "greedy_iou, 1 s buffer" "greedy_iou:max_age=5, 0.17 s buffer" \
	      --start 20 --seconds 13 --crop 0 600 --every 3 --fps 10 --width 640 --out ../docs/footage/real-compare.gif
up:        ; docker compose --profile sim up -d --build     # synthetic traffic, no video needed
up-video:  ; docker compose --profile video up -d --build   # needs media/sample.mp4
down:      ; docker compose $(ALL) down -v
logs:      ; docker compose $(ALL) logs -f sim edge ingest
counts:    ; docker compose exec postgres psql -U postgres lab -c "select counter, kind, count(*) from zone_events group by 1,2 order by 1,2"
delivery:  ; cd harness && python scripts/check_delivery.py   # every finished sim scene stored exactly once?
drill:     ; chaos/drill.sh                                  # ~9 min: 64k, latency, outage, then delivery + lag
broker-restart: ; chaos/broker-restart.sh                    # ~5 min: ingest away across a broker restart, then delivery
