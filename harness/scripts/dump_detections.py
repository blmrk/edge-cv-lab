"""Video -> detections JSONL (no IDs). Run the detector once, then compare any number of trackers.
Needs: pip install -e '.[video]'

python scripts/dump_detections.py --video ../media/clip.mp4 --out runs/dets.jsonl
conf defaults to 0.1 on purpose: ByteTrack-style trackers use low-score boxes to ride through occlusion.
"""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--classes", type=int, nargs="*", default=[2, 5, 7])  # COCO car, bus, truck
    a = ap.parse_args()

    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30
    model = YOLO(a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for frame, r in enumerate(model.predict(a.video, conf=a.conf, classes=a.classes, stream=True, verbose=False)):
            for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                fh.write(json.dumps({"frame": frame, "ts_ms": int(frame * 1000 / fps),
                                     "bbox": [round(v, 1) for v in xyxy], "score": round(conf, 3),
                                     "cls": model.names[int(cls)]}) + "\n")


if __name__ == "__main__":
    main()
