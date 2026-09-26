"""Video -> tracks JSONL using Ultralytics' built-in trackers. Needs: pip install -e '.[video]'

python scripts/dump_tracks.py --video ../media/sample.mp4 --tracker bytetrack.yaml --out runs/bytetrack.jsonl
python scripts/dump_tracks.py --video ../media/sample.mp4 --tracker botsort.yaml   --out runs/botsort.jsonl
NMS is class-agnostic, as in dump_detections.py.
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
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--classes", type=int, nargs="*", default=[2, 5, 7])  # COCO car, bus, truck
    ap.add_argument("--per-class-nms", action="store_true",
                    help="NMS within each class only, which keeps a car box and a truck box on one vehicle "
                         "(Ultralytics' default; the case study's baseline figures used it)")
    a = ap.parse_args()

    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30
    model = YOLO(a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        results = model.track(a.video, tracker=a.tracker, classes=a.classes,
                              agnostic_nms=not a.per_class_nms, stream=True, verbose=False)
        for frame, r in enumerate(results):
            if r.boxes.id is None:
                continue
            for xyxy, tid, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(),
                                            r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                fh.write(json.dumps({"frame": frame, "ts_ms": int(frame * 1000 / fps), "track_id": int(tid),
                                     "bbox": [round(v, 1) for v in xyxy], "score": round(conf, 3),
                                     "cls": model.names[int(cls)]}) + "\n")


if __name__ == "__main__":
    main()
