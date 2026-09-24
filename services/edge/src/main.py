"""Simulated edge node: RTSP -> detect+track -> zone state machine -> MQTT (QoS 1, persistent session).

The zone logic is imported from the replay harness, so what is tested offline is exactly what runs here.
"""
import json
import os
import time

import paho.mqtt.client as mqtt
from ulid import ULID
from ultralytics import YOLO

from replay.schema import TrackBox
from replay.zones import DebouncedZoneCounter

RTSP = os.environ["RTSP_URL"]
DEVICE = os.environ.get("DEVICE_ID", "edge-01")
ZONE = [tuple(p) for p in json.loads(os.environ["ZONE_POLYGON"])]
HOST, PORT = os.environ.get("MQTT_HOST", "toxiproxy"), int(os.environ.get("MQTT_PORT", "1883"))


def mqtt_client() -> mqtt.Client:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=DEVICE, clean_session=False)
    c.reconnect_delay_set(min_delay=1, max_delay=60)  # backoff; never hammer a metered uplink
    c.max_queued_messages_set(5000)                   # bounded store-and-forward
    c.will_set(f"devices/{DEVICE}/status", json.dumps({"state": "offline"}), qos=1, retain=True)
    c.connect_async(HOST, PORT, keepalive=30)
    c.loop_start()
    return c


def main():
    client, counter, model = mqtt_client(), DebouncedZoneCounter(ZONE), YOLO(os.environ.get("MODEL", "yolov8n.pt"))
    frames, last_hb = 0, time.monotonic()
    for r in model.track(RTSP, tracker=os.environ.get("TRACKER", "bytetrack.yaml"),
                         classes=[2, 5, 7], stream=True, verbose=False):
        frames += 1
        ts_ms = time.time_ns() // 1_000_000  # integer ms, UTC
        if r.boxes.id is not None:
            for xyxy, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist()):
                for ev in counter.update(TrackBox(frames, ts_ms, int(tid), tuple(xyxy))):
                    payload = {"event_id": str(ULID()),  # one ID, generated once, used everywhere downstream
                               "device_id": DEVICE, "counter": "debounced", "kind": ev.kind, "track_id": ev.track_id, "ts_ms": ev.ts_ms}
                    client.publish(f"events/{DEVICE}/zone", json.dumps(payload), qos=1)
        if time.monotonic() - last_hb > 10:
            # App-level health, distinct from network liveness: a crash-looping node stops sending this.
            fps = frames / (time.monotonic() - last_hb)
            client.publish(f"devices/{DEVICE}/status", json.dumps({"state": "online", "fps": round(fps, 1)}),
                           qos=1, retain=True)
            frames, last_hb = 0, time.monotonic()


if __name__ == "__main__":
    main()
