"""Deterministic synthetic tracks that reproduce known failure modes. Run from harness/."""
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay.synth import PILLAR, generate_queue, generate_traffic  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "fixtures"
FPS, W, H = 30, 80, 60  # box size
ZONE = [[400, 200], [800, 200], [800, 600], [400, 600]]


def rows(track_id, centres, start=0):
    for i, (cx, cy) in enumerate(centres):
        f = start + i
        yield {"frame": f, "ts_ms": int(f * 1000 / FPS), "track_id": track_id,
               "bbox": [cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], "score": 0.9, "cls": "car"}


def write(name, records):
    with open(OUT / name, "w") as fh:
        for r in sorted(records, key=lambda r: (r["frame"], r.get("track_id", 0))):
            fh.write(json.dumps(r) + "\n")


def main():
    rnd = random.Random(7)
    OUT.mkdir(exist_ok=True)
    (OUT / "zone.json").write_text(json.dumps({"polygon": ZONE}))

    # 1. Boundary jitter: one car parks on the zone's left edge (x=400), box jitters +-6px for 10 s,
    #    then drives through and leaves. Ground truth: 1 visit.
    c = [(300 + i * 4, 400) for i in range(25)]                       # approach
    c += [(400 + rnd.uniform(-6, 6), 400) for _ in range(300)]        # jitter on boundary
    c += [(400 + i * 5, 400) for i in range(1, 100)]                  # cross and exit right
    write("boundary_jitter.jsonl", rows(1, c))

    # 2. Shadow expansion: car drives in the lane BELOW the zone (footpoint y=640 > 600),
    #    but a shadow stretches the box upward so the centroid drifts into the zone. Truth: 0 visits.
    recs = []
    for i in range(200):
        cx, foot = 300 + i * 3, 640
        top = foot - H - (120 if 60 < i < 150 else 0)               # shadow makes box taller
        recs.append({"frame": i, "ts_ms": int(i * 1000 / FPS), "track_id": 2,
                     "bbox": [cx - W / 2, top, cx + W / 2, foot], "score": 0.8, "cls": "car"})
    write("shadow_expansion.jsonl", recs)

    # 3. Mixed traffic: two lanes, stops, boundary idlers, shadows. Truth is written alongside.
    boxes, truth = generate_traffic(seed=11, vehicles=24)
    write("traffic.jsonl", [{**asdict(b), "bbox": [round(v, 1) for v in b.bbox]} for b in boxes])
    (OUT / "traffic.truth.json").write_text(json.dumps({"expected_visits": truth}))

    # 4. Queue with a pillar: detections (no ids), ground-truth tracks, and visit truth.
    dets, gt, _ = generate_queue(seed=5, cars=6)
    write("queue.dets.jsonl", [{**asdict(d), "bbox": [round(v, 1) for v in d.bbox]} for d in dets])
    write_gt = [{**asdict(b), "bbox": [round(v, 1) for v in b.bbox]} for b in gt]
    write("queue.gt.jsonl", write_gt)
    # a visit "enters" when the car reappears from behind the pillar, inside the zone
    enters = {}
    for b in gt:
        cx = (b.bbox[0] + b.bbox[2]) / 2
        if cx > PILLAR[1] and b.track_id not in enters:
            enters[b.track_id] = b.ts_ms
    (OUT / "queue.truth.json").write_text(json.dumps({"expected_visits": len(enters), "enters_ms": sorted(enters.values())}))


if __name__ == "__main__":
    main()
