# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""No-video edge node. Replays seeded synthetic traffic in real time, runs BOTH counters on it,
and publishes like a real device. Lets the whole lab (broker, chaos, ingest, Grafana) run with
no footage, no model and no GPU. Each loop uses a new seed, so traffic never repeats exactly.
"""
import json
import os
import time

import paho.mqtt.client as mqtt
from ulid import ULID

from replay.gauge import OccupancyGauge
from replay.synth import FPS, ZONE, by_frame, generate_traffic
from replay.zones import DebouncedZoneCounter, NaiveZoneCounter

DEVICE = os.environ.get("DEVICE_ID", "sim-01")
SPEED = float(os.environ.get("SPEED", "2"))  # 2 = twice real time
HOST, PORT = os.environ.get("MQTT_HOST", "toxiproxy"), int(os.environ.get("MQTT_PORT", "1883"))
# Zone events only. 0 = fire-and-forget, the failure the uplink drill measures; truth and heartbeat stay at 1,
# since the delivery check needs each scene's truth to know the scene finished.
QOS = int(os.environ.get("QOS", "1"))
# 0 = off (default). Set, e.g. 500, to publish the debounced counter's vehicles in the zone once a second (QoS 0).
GAUGE_MS = int(os.environ.get("OCCUPANCY_GAUGE_MS", "0"))


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
        gauge = OccupancyGauge(counters["debounced"], GAUGE_MS, every_ms=int(1000 * SPEED))  # scene time: 1 s of wall
        t0, wall0, last_hb, n = time.monotonic(), time.time_ns() // 1_000_000, 0.0, 0
        for frame, bucket in by_frame(boxes):
            delay = t0 + frame / FPS / SPEED - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            n += 1
            for name, counter in counters.items():
                for b in bucket:
                    for ev in counter.update(b):
                        # event time, not publish time: debounced enters are held until the visit closes
                        c.publish(f"events/{DEVICE}/zone", json.dumps({
                            "event_id": str(ULID()), "device_id": DEVICE, "counter": name, "kind": ev.kind,
                            "track_id": seed * 1000 + ev.track_id, "ts_ms": wall0 + int(ev.ts_ms / SPEED)}), qos=QOS)
            if (in_zone := gauge.sample(bucket[0].ts_ms)) is not None:
                c.publish(f"occupancy/{DEVICE}", json.dumps({"device_id": DEVICE, "counter": "debounced", "in_zone": in_zone,
                                                             "ts_ms": wall0 + int(bucket[0].ts_ms / SPEED)}), qos=0)
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
