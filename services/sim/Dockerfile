FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir "paho-mqtt>=2.0" "python-ulid>=2.7"
COPY harness/replay ./replay
COPY services/sim/src ./
CMD ["python", "-u", "main.py"]
