FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
COPY services/edge/requirements.txt .
# torch pinned with ultralytics (requirements.txt): the detections the case studies measure depend on both
RUN pip install --no-cache-dir torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
# fetch the default weights at build time, not on every container start, into /app, where YOLO('yolov8n.pt') looks
# first; the sha256 is that of the file in the image the case studies were measured on, checked before any unpickling
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt', 'yolov8n.pt')" \
 && echo "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36  yolov8n.pt" | sha256sum -c -
# offline: no usage events to the vendor's analytics; no autoinstall: no pip installs at runtime;
# safe load: a checkpoint is unpickled with weights_only, against ultralytics' allow-list of model classes
ENV YOLO_OFFLINE=1 YOLO_AUTOINSTALL=false ULTRALYTICS_SAFE_LOAD=1
# load check: the verified weights load under safe load
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
COPY harness/replay ./replay
COPY services/edge/src ./
CMD ["python", "-u", "main.py"]
