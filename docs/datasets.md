# Steady-camera datasets

Stock sites are full of drones and timelapses. For genuinely fixed CCTV footage with tracking ground
truth, use the research benchmarks below. They are research-licensed: publish metrics and a citation,
never frames. Anything shown in this repo comes from the Pexels clip or self-shot footage
(see `media/SOURCES.md`).

| Dataset | Scene | Ground truth | Best for | Notes |
|---|---|---|---|---|
| UA-DETRAC | Overpass cameras, 24 sites, 960x540, 25 fps | Per-frame boxes + IDs, occlusion, weather | Queueing, occlusion, ID handover | Mirrored on IEEE DataPort / Kaggle; original site unreliable |
| Urban Tracker (Polytechnique Montreal) | Static intersection cameras | Boxes + IDs, vehicles and pedestrians | Zone dwell, turning traffic | Small, free for research |
| GRAM Road-Traffic Monitoring | Fixed road cameras | Boxes + IDs | Flow counting | Free for research |
| AAU RainSnow | Lamp-post intersection cameras | Boxes | Bad-weather robustness | Confirm licence on Kaggle |
| CDnet 2014 (`highway`, `traffic`) | Static clips | Change masks only | Detector sanity checks | No track IDs |

## UA-DETRAC in the harness

Keep the download under `media/UA-DETRAC/`, which is gitignored, never under `harness/`.

```bash
cd harness
# 1. ground truth from the XML annotations (drops boxes inside the sequence's ignored regions)
python scripts/detrac_to_gt.py --xml ../media/UA-DETRAC/DETRAC-Train-Annotations-XML/MVI_20011.xml --out runs/MVI_20011

# 2. detections from the frame directory (25 fps; frames carry no rate)
python scripts/dump_detections.py --video ../media/UA-DETRAC/Insight-MVT_Annotation_Train/MVI_20011 --fps 25 --out runs/MVI_20011.dets.jsonl

# 3. draw a zone over the queueing area, away from runs/MVI_20011.ignored.json, then derive visit truth
#    from the annotated tracks (perfect IDs) so trackers can be scored on visits as well as identity
python scripts/detrac_to_gt.py --xml ../media/UA-DETRAC/DETRAC-Train-Annotations-XML/MVI_20011.xml --out runs/MVI_20011 --zone runs/zone.json

# 4. compare
python -m replay.compare --dets runs/MVI_20011.dets.jsonl --zone runs/zone.json \
    --truth runs/MVI_20011.truth.json --gt runs/MVI_20011.gt.jsonl --trackers greedy_iou groundplane
```

To make an mp4 for `tools/label.html` or the lab's RTSP camera:
`ffmpeg -framerate 25 -i ../media/UA-DETRAC/Insight-MVT_Annotation_Train/MVI_20011/img%05d.jpg -c:v libx264 -pix_fmt yuv420p ../media/MVI_20011.mp4`
(`media/` is gitignored; it is research data).

The converter was written against the documented annotation format and tested on a synthetic file,
not on a downloaded sequence. If a real file fails to parse, the element names are the first thing to check.

Citation: Wen et al., "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking",
Computer Vision and Image Understanding, 2020.
