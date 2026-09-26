"""Video -> detections JSONL (no IDs). Run the detector once, then compare any number of trackers.
Needs: pip install -e '.[video]'

python scripts/dump_detections.py --video ../media/clip.mp4 --out runs/dets.jsonl
python scripts/dump_detections.py --video ../media/UA-DETRAC/Insight-MVT_Annotation_Train/MVI_20011 --fps 25 --out runs/MVI_20011.dets.jsonl
A directory of frames (UA-DETRAC style img00001.jpg ...), a glob (dir/*.jpg) or a .txt list of frames
works too; --fps is required for those, since images carry none. Videos in any format, stream URLs and
webcam indexes are numbered in read order at their own fps.
Image frames are numbered from the file name (img00001.jpg -> frame 0), not by read order, so a frame
that fails to load leaves a gap instead of shifting every later frame against the ground truth.
conf defaults to 0.1 on purpose: ByteTrack-style trackers use low-score boxes to ride through occlusion.
NMS is class-agnostic: one vehicle scored as both car and truck is one box, not two for a tracker to follow.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay.detections import frame_from_filename, is_frame_source, is_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file, stream URL, or frames: a directory, glob or .txt list")
    ap.add_argument("--fps", type=float, help="required for frames (a directory, glob, .txt list or image)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--classes", type=int, nargs="*", default=[2, 5, 7])  # COCO car, bus, truck
    ap.add_argument("--per-class-nms", action="store_true",
                    help="NMS within each class only, which keeps a car box and a truck box on one vehicle "
                         "(Ultralytics' default; the case study's baseline figures used it)")
    a = ap.parse_args()

    if is_frame_source(a.video):
        if not a.fps:
            raise SystemExit(f"--fps is required: {a.video} is read as image frames, which carry no fps")
        fps = a.fps
    else:
        fps = a.fps or cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30
    model = YOLO(a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prev = None
    with open(a.out, "w") as fh:
        results = model.predict(a.video, conf=a.conf, classes=a.classes, agnostic_nms=not a.per_class_nms,
                                stream=True, verbose=False)
        for i, r in enumerate(results):
            frame = frame_from_filename(r.path) if is_image(r.path) else i  # videos and streams: read order
            if frame is None:
                raise SystemExit(f"{r.path}: file name does not end in a frame number (e.g. img00001.jpg)")
            if frame < 0:
                raise SystemExit(f"{r.path}: frame numbers in file names start at 1 (img00001.jpg)")
            if prev is not None and frame <= prev:
                raise SystemExit(f"{r.path}: frame {frame} after frame {prev}; frame numbers must increase in "
                                 f"file-name order (zero-pad them: img00001.jpg)")
            prev = frame
            for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                fh.write(json.dumps({"frame": frame, "ts_ms": int(frame * 1000 / fps),
                                     "bbox": [round(v, 1) for v in xyxy], "score": round(conf, 3),
                                     "cls": model.names[int(cls)]}) + "\n")


if __name__ == "__main__":
    main()
