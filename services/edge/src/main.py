"""Simulated edge node: RTSP -> detect+track -> zone state machine -> MQTT (QoS 1, persistent session).

The zone logic is imported from the replay harness, so what is tested offline is exactly what runs here. So are the
opt-in second-box filter and the ByteTrack adapter it feeds (CONTAIN_SHARE); off, the default, model.track() runs
as before.
"""
import json
import os
import time

import paho.mqtt.client as mqtt
from ulid import ULID
from ultralytics import YOLO

from replay.detections import Detection
from replay.gauge import OccupancyGauge
from replay.schema import TrackBox
from replay.secondbox import contained
from replay.trackers import create
from replay.zones import DebouncedZoneCounter

RTSP = os.environ["RTSP_URL"]
DEVICE = os.environ.get("DEVICE_ID", "edge-01")
ZONE = [tuple(p) for p in json.loads(os.environ["ZONE_POLYGON"])]
HOST, PORT = os.environ.get("MQTT_HOST", "toxiproxy"), int(os.environ.get("MQTT_PORT", "1883"))
# Both off (0) by default. Measured pair: MIN_TRAVEL_PX=30 (docs/case-study-phantoms.md) keeps static phantoms from
# entering, and OCCUPANCY_GAUGE_MS=500 (docs/case-study-balance.md) publishes vehicles in the zone once a second.
MIN_TRAVEL_PX = float(os.environ.get("MIN_TRAVEL_PX", "0"))
GAUGE_MS = int(os.environ.get("OCCUPANCY_GAUGE_MS", "0"))


def contain_config(env) -> tuple[float, bool]:
    """(share, same_class) from CONTAIN_SHARE and CONTAIN_SAME_CLASS, share 0 when off. Any other value stops the edge
    here, before the model loads: a typo must not run without the filter, or with one nobody asked for."""
    raw, same = env.get("CONTAIN_SHARE", "0"), env.get("CONTAIN_SAME_CLASS", "1")
    try:
        share = float(raw)
    except ValueError:
        share = float("nan")
    if not 0 <= share <= 1 or same not in ("0", "1"):  # nan fails every comparison, inf the range
        raise SystemExit(f"CONTAIN_SHARE={raw!r} CONTAIN_SAME_CLASS={same!r}: CONTAIN_SHARE is 0 (off) or a share in "
                         "(0, 1], CONTAIN_SAME_CLASS 1 (inside a box of its own class only) or 0 (any class)")
    if share and env.get("TRACKER", "bytetrack.yaml") != "bytetrack.yaml":
        # the on path runs bytetrack.yaml's defaults through the replay adapter: another yaml would be dropped silently
        raise SystemExit(f"TRACKER={env['TRACKER']!r} with CONTAIN_SHARE={raw!r}: the filter runs bytetrack.yaml only")
    return share, same == "1"


# Off (0) by default. CONTAIN_SHARE=0.9 drops a box lying 0.9 of its own area inside a higher-scoring one of its class
# (any class with CONTAIN_SAME_CLASS=0) before tracking: contain090_same, measured in the replay harness
# (docs/case-study-straddles.md).
CONTAIN_SHARE, CONTAIN_SAME_CLASS = contain_config(os.environ)


def mqtt_client() -> mqtt.Client:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=DEVICE, clean_session=False)
    c.reconnect_delay_set(min_delay=1, max_delay=60)  # backoff; never hammer a metered uplink
    c.max_queued_messages_set(5000)                   # bounded store-and-forward
    c.will_set(f"devices/{DEVICE}/status", json.dumps({"state": "offline"}), qos=1, retain=True)
    c.connect_async(HOST, PORT, keepalive=30)
    c.loop_start()
    return c


def main():
    print(f"contain: on share={CONTAIN_SHARE:g} same_class={int(CONTAIN_SAME_CLASS)} conf=0.1 tracker=bytetrack.yaml"
          if CONTAIN_SHARE else "contain: off")
    client, model = mqtt_client(), YOLO(os.environ.get("MODEL", "yolov8n.pt"))
    counter = DebouncedZoneCounter(ZONE, min_travel_px=MIN_TRAVEL_PX)
    gauge = OccupancyGauge(counter, GAUGE_MS)
    frames, last_hb = 0, time.monotonic()  # before track() or predict(), as today: their setup is in the first window
    if CONTAIN_SHARE:
        tracker = create("bytetrack")  # once per process: every BYTETracker restarts the track-ID counter they share
        # conf 0.1 is what model.track() forces for itself (predict() alone takes 0.25); no iou: the default on both
        source = model.predict(RTSP, conf=0.1, classes=[2, 5, 7], agnostic_nms=True, stream=True, verbose=False)
    else:
        source = model.track(RTSP, tracker=os.environ.get("TRACKER", "bytetrack.yaml"),
                             classes=[2, 5, 7], agnostic_nms=True,  # no car + truck box pair on one vehicle
                             stream=True, verbose=False)
    dets_n, dropped_n, filter_s = 0, 0, 0.0  # the filter's counts since the last heartbeat
    for n, r in enumerate(source):  # n: frames read since start, never reset (frames resets at each heartbeat)
        frames += 1
        ts_ms = time.time_ns() // 1_000_000  # integer ms, UTC
        if CONTAIN_SHARE:
            dets = [Detection(n, ts_ms, tuple(xyxy), conf, model.names[int(c)])
                    for xyxy, conf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            t = time.perf_counter()
            kept = contained(dets, CONTAIN_SHARE, CONTAIN_SAME_CLASS)
            filter_s += time.perf_counter() - t
            dets_n, dropped_n = dets_n + len(dets), dropped_n + len(dets) - len(kept)
            boxes = tracker.update(n, ts_ms, kept)  # every frame, empty ones too: tracks age per update
        elif r.boxes.id is not None:
            boxes = [TrackBox(frames, ts_ms, int(tid), tuple(xyxy))
                     for xyxy, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist())]
        else:
            boxes = []
        for box in boxes:
            for ev in counter.update(box):
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
            if CONTAIN_SHARE:  # the filter's window, to the log only: nothing new goes over MQTT
                print(f"contain ts_ms={ts_ms} frames={frames} dets={dets_n} dropped={dropped_n} "
                      f"filter_ms={filter_s * 1000:.1f}")
                dets_n, dropped_n, filter_s = 0, 0, 0.0
            frames, last_hb = 0, time.monotonic()


if __name__ == "__main__":
    main()
