FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
COPY services/edge/requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
# fetch the default weights at build time, not on every container start
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
COPY harness/replay ./replay
COPY services/edge/src ./
CMD ["python", "-u", "main.py"]
