# Steady-camera datasets

Stock sites are full of drones and timelapses. For genuinely fixed CCTV footage with tracking ground
truth, use the research benchmarks below. They are research-licensed: publish metrics and a citation,
never frames. Anything shown in this repo is synthetic or from republishable footage
logged in `media/SOURCES.md`.

| Dataset | Scene | Ground truth | Best for | Notes |
|---|---|---|---|---|
| UA-DETRAC | Overpass cameras, 24 sites, 960x540, 25 fps | Per-frame boxes + IDs, occlusion, weather | Queueing, occlusion, ID handover | Mirrored on IEEE DataPort / Kaggle; original site unreliable |
| Urban Tracker (Polytechnique Montreal) | Static intersection cameras | Boxes + IDs, vehicles and pedestrians | Zone dwell, turning traffic | Small, free for research |
| GRAM Road-Traffic Monitoring | Fixed road cameras | Boxes + IDs | Flow counting | Free for research |
| AAU RainSnow | Lamp-post intersection cameras | Boxes | Bad-weather robustness | Confirm licence on Kaggle |
| CDnet 2014 (`highway`) | Static clip, baseline category | Change masks only | Detector sanity checks | No track IDs. Not `traffic`: it is in the camera-jitter category |

## UA-DETRAC in the harness

Keep the download under `media/UA-DETRAC/`, which is gitignored, never under `harness/`.

```bash
setopt interactive_comments 2>/dev/null || true
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

One test sequence can be fetched on its own instead of the multi-GB set: an mp4 of its frames plus its XML, from the
Kaggle mirrors logged in `media/SOURCES.md` (MVI_40714, the task C clip: 46317496 and 11609640 bytes). The mp4 is an
uploader's encode, so check it lines up with the annotations before using it: `ffprobe -count_frames` must give the
XML's frame count at 25 fps, and a few original JPEGs must match decoded frames n-1. For MVI_40714 both hold: 1180
frames, and img00001, img00590 and img01180 match decoded frames 0, 589 and 1179 best. Then run the detector on the mp4
with `--fps 25`. The converter parses the real XML: 33749 boxes in 62 tracks over all 1180 frames, 186 boxes dropped
in three ignored regions.

```bash
cd media/UA-DETRAC
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_read_frames MVI_40714.mp4
python3 -c "
import cv2, numpy as np
cap = cv2.VideoCapture('MVI_40714.mp4'); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
for n in (1, 590, 1180):
    j = cv2.imread(f'check/img{n:05d}.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    d = {k: round(float(np.abs(frames[k] - j).mean()), 2) for k in range(n - 3, n + 2) if 0 <= k < len(frames)}
    print(f'img{n:05d}.jpg best matches decoded frame', min(d, key=d.get), d)"
cd ../../harness && python scripts/detrac_to_gt.py --xml ../media/UA-DETRAC/MVI_40714.xml --out runs/MVI_40714
```

A second test sequence, MVI_40855 (task C's second clip, logged in `media/SOURCES.md`), was fetched from the same mirrors
and checked the same way: 1090 frames at 25 fps, and img00001, img00545 and img01090 (kept in `check-40855/`) match
decoded frames 0, 544 and 1089 best. The converter prints
`runs/MVI_40855.gt.jsonl: 29802 boxes, 55 tracks, 291 boxes dropped in ignored regions`. The check is the one above with
the file names swapped:

```bash
cd media/UA-DETRAC
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_read_frames MVI_40855.mp4
python3 -c "
import cv2, numpy as np
cap = cv2.VideoCapture('MVI_40855.mp4'); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
for n in (1, 545, 1090):
    j = cv2.imread(f'check-40855/img{n:05d}.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    d = {k: round(float(np.abs(frames[k] - j).mean()), 2) for k in range(n - 3, n + 2) if 0 <= k < len(frames)}
    print(f'img{n:05d}.jpg best matches decoded frame', min(d, key=d.get), d)"
cd ../../harness && python scripts/detrac_to_gt.py --xml ../media/UA-DETRAC/MVI_40855.xml --out runs/MVI_40855
```

Citation: Wen et al., "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking",
Computer Vision and Image Understanding, 2020.
