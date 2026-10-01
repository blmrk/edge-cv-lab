"""Simulated edge node: RTSP -> detect+track -> zone state machine -> MQTT (QoS 1, persistent session).

The zone logic is imported from the replay harness, so what is tested offline is exactly what runs here.
"""
import json
import os
import time

import paho.mqtt.client as mqtt
from ulid import ULID
from ultralytics import YOLO

from replay.gauge import OccupancyGauge
from replay.schema import TrackBox
from replay.zones import DebouncedZoneCounter

RTSP = os.environ["RTSP_URL"]
DEVICE = os.environ.get("DEVICE_ID", "edge-01")
ZONE = [tuple(p) for p in json.loads(os.environ["ZONE_POLYGON"])]
HOST, PORT = os.environ.get("MQTT_HOST", "toxiproxy"), int(os.environ.get("MQTT_PORT", "1883"))
# Both off (0) by default. Measured pair: MIN_TRAVEL_PX=30 (docs/case-study-phantoms.md) keeps static phantoms from
# entering, and OCCUPANCY_GAUGE_MS=500 (docs/case-study-balance.md) publishes vehicles in the zone once a second.
MIN_TRAVEL_PX = float(os.environ.get("MIN_TRAVEL_PX", "0"))
GAUGE_MS = int(os.environ.get("OCCUPANCY_GAUGE_MS", "0"))


def mqtt_client() -> mqtt.Client:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=DEVICE, clean_session=False)
    c.reconnect_delay_set(min_delay=1, max_delay=60)  # backoff; never hammer a metered uplink
    c.max_queued_messages_set(5000)                   # bounded store-and-forward
    c.will_set(f"devices/{DEVICE}/status", json.dumps({"state": "offline"}), qos=1, retain=True)
    c.connect_async(HOST, PORT, keepalive=30)
    c.loop_start()
    return c


def main():
    client, model = mqtt_client(), YOLO(os.environ.get("MODEL", "yolov8n.pt"))
    counter = DebouncedZoneCounter(ZONE, min_travel_px=MIN_TRAVEL_PX)
    gauge = OccupancyGauge(counter, GAUGE_MS)
    frames, last_hb = 0, time.monotonic()
    for r in model.track(RTSP, tracker=os.environ.get("TRACKER", "bytetrack.yaml"),
                         classes=[2, 5, 7], agnostic_nms=True,  # no car + truck box pair on one vehicle
                         stream=True, verbose=False):
        frames += 1
        ts_ms = time.time_ns() // 1_000_000  # integer ms, UTC
        if r.boxes.id is not None:
            for xyxy, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist()):
                for ev in counter.update(TrackBox(frames, ts_ms, int(tid), tuple(xyxy))):
                    payload = {"event_id": str(ULID()),  # one ID, generated once, used everywhere downstream
                               "device_id": DEVICE, "counter": "debounced", "kind": ev.kind, "track_id": ev.track_id, "ts_ms": ev.ts_ms}
                    client.publish(f"events/{DEVICE}/zone", json.dumps(payload), qos=1)
        # every frame, detections or not: the gauge reads the clock, so a track gone quiet drops out on time
        if (in_zone := gauge.sample(ts_ms)) is not None:  # QoS 0: a stale sample is not worth queueing through an outage
            client.publish(f"occupancy/{DEVICE}", json.dumps({"device_id": DEVICE, "counter": "debounced",
                                                             "in_zone": in_zone, "ts_ms": ts_ms}), qos=0)
        if time.monotonic() - last_hb > 10:
            # App-level health, distinct from network liveness: a crash-looping node stops sending this.
            fps = frames / (time.monotonic() - last_hb)
            client.publish(f"devices/{DEVICE}/status", json.dumps({"state": "online", "fps": round(fps, 1)}),
                           qos=1, retain=True)
            frames, last_hb = 0, time.monotonic()


if __name__ == "__main__":
    main()
