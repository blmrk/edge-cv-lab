"""No-video edge node. Replays seeded synthetic traffic in real time, runs BOTH counters on it,
and publishes like a real device. Lets the whole lab (broker, chaos, ingest, Grafana) run with
no footage, no model and no GPU. Each loop uses a new seed, so traffic never repeats exactly.
"""
import json
import os
import time

import paho.mqtt.client as mqtt
from ulid import ULID

from replay.synth import FPS, ZONE, by_frame, generate_traffic
from replay.zones import DebouncedZoneCounter, NaiveZoneCounter

DEVICE = os.environ.get("DEVICE_ID", "sim-01")
SPEED = float(os.environ.get("SPEED", "2"))  # 2 = twice real time
HOST, PORT = os.environ.get("MQTT_HOST", "toxiproxy"), int(os.environ.get("MQTT_PORT", "1883"))


def main():
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=DEVICE, clean_session=False)
    c.reconnect_delay_set(1, 60)
    c.max_queued_messages_set(5000)
    c.will_set(f"devices/{DEVICE}/status", json.dumps({"state": "offline"}), qos=1, retain=True)
    c.connect_async(HOST, PORT, keepalive=30)
    c.loop_start()

    seed = int(os.environ.get("SEED", "11"))
    while True:
        boxes, truth = generate_traffic(seed=seed)
        counters = {"naive": NaiveZoneCounter(ZONE, "centroid"), "debounced": DebouncedZoneCounter(ZONE)}
        t0, last_hb, n = time.monotonic(), 0.0, 0
        for frame, bucket in by_frame(boxes):
            delay = t0 + frame / FPS / SPEED - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            n += 1
            for name, counter in counters.items():
                for b in bucket:
                    for ev in counter.update(b):
                        c.publish(f"events/{DEVICE}/zone", json.dumps({
                            "event_id": str(ULID()), "device_id": DEVICE, "counter": name, "kind": ev.kind,
                            "track_id": seed * 1000 + ev.track_id, "ts_ms": time.time_ns() // 1_000_000}), qos=1)
            if time.monotonic() - last_hb > 10:
                fps = n / (time.monotonic() - last_hb) if last_hb else FPS * SPEED
                c.publish(f"devices/{DEVICE}/status", json.dumps({"state": "online", "fps": round(fps, 1)}),
                          qos=1, retain=True)
                last_hb, n = time.monotonic(), 0
        print(f"seed {seed} done, ground truth visits: {truth}", flush=True)
        c.publish(f"truth/{DEVICE}", json.dumps({"seed": seed, "expected_visits": truth,
                                                  "ts_ms": time.time_ns() // 1_000_000}), qos=1)
        seed += 1


if __name__ == "__main__":
    main()
