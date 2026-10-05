# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""MQTT -> Postgres. Idempotent on event_id, so QoS 1 redelivery never double counts."""
import json
import os
import time

import paho.mqtt.client as mqtt
import psycopg

DDL = """
CREATE TABLE IF NOT EXISTS zone_events (
  event_id   text PRIMARY KEY,
  device_id  text NOT NULL,
  kind       text NOT NULL CHECK (kind IN ('enter','exit')),
  track_id   int  NOT NULL,
  ts_ms      bigint NOT NULL,
  counter    text NOT NULL DEFAULT 'debounced',
  received_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS device_status (
  device_id text PRIMARY KEY, state text NOT NULL, fps real, updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ground_truth (
  device_id text NOT NULL, seed int NOT NULL, expected_visits int NOT NULL, ts_ms bigint NOT NULL,
  PRIMARY KEY (device_id, seed)
);
CREATE INDEX IF NOT EXISTS zone_events_received_at ON zone_events (received_at);
CREATE TABLE IF NOT EXISTS zone_occupancy (
  device_id text NOT NULL, counter text NOT NULL, in_zone int NOT NULL, ts_ms bigint NOT NULL,
  received_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS zone_occupancy_received_at ON zone_occupancy (received_at);"""


def connect():
    while True:
        try:
            return psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
        except psycopg.OperationalError:
            time.sleep(2)


def main():
    db = connect()
    db.execute(DDL)

    def on_connect(c, *_):
        c.subscribe([("events/+/zone", 1), ("devices/+/status", 1), ("truth/+", 1), ("occupancy/+", 0)])

    def on_message(_c, _u, msg):
        if (d := valid(msg)) is None: return  # not JSON, a key missing, a value its column cannot take: see valid()
        if msg.topic.startswith("events/"):
            db.execute("INSERT INTO zone_events (event_id, device_id, kind, track_id, ts_ms, counter) "
                       "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                       (d["event_id"], d["device_id"], d["kind"], d["track_id"], d["ts_ms"],
                        d.get("counter", "debounced")))
        elif msg.topic.startswith("occupancy/"):  # a gauge sample, sent only by devices with OCCUPANCY_GAUGE_MS set
            db.execute("INSERT INTO zone_occupancy (device_id, counter, in_zone, ts_ms) VALUES (%s,%s,%s,%s)",
                       (d["device_id"], d.get("counter", "debounced"), d["in_zone"], d["ts_ms"]))
        elif msg.topic.startswith("truth/"):
            db.execute("INSERT INTO ground_truth VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                       (msg.topic.split("/")[1], d["seed"], d["expected_visits"], d["ts_ms"]))
        else:
            dev = msg.topic.split("/")[1]
            db.execute("INSERT INTO device_status (device_id, state, fps) VALUES (%s,%s,%s) "
                       "ON CONFLICT (device_id) DO UPDATE SET state=EXCLUDED.state, fps=EXCLUDED.fps, updated_at=now()",
                       (dev, d["state"], d.get("fps")))

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ingest", clean_session=False)
    c.on_connect, c.on_message = on_connect, on_message
    c.connect(os.environ.get("MQTT_HOST", "emqx"), 1883)
    c.loop_forever()


# on_message's guard. paho runs on_message before the QoS 1 PUBACK and re-raises what it raises, so a message it raised
# on was never acked: restart plus the persistent session (clean_session=False), or the retained status, brought it
# back, and one bad message crash-looped ingest. valid() logs and drops a bad one instead, before any SQL runs.
dropped = 0  # bad messages since start
PREFIX = 64  # of the topic and the payload, all a log line shows: a hostile message cannot flood the log


def _text(v):  # text: a str, no NUL, no lone surrogate (JSON can escape one; UTF-8, so Postgres, cannot hold it)
    return isinstance(v, str) and "\0" not in v and v.encode(errors="replace").decode() == v


KEY = 256  # chars in a primary key's text: at most 1 KB of UTF-8; Postgres raises on a btree index row past 2704 bytes


def _key(v):  # text in a primary key: event_id, and the device id a truth or status topic carries
    return _text(v) and len(v) <= KEY


def _int(bits):  # int (32) or bigint (64). Not a bool: an int to Python, a type of its own to Postgres
    return lambda v: type(v) is int and -2 ** (bits - 1) <= v < 2 ** (bits - 1)


def _fps(v):  # real, nullable. Postgres errors on a finite value past real's range, about 1e-37 to 1e37 (its docs)
    return v is None or type(v) in (int, float) and (v == 0 or v != v or abs(v) == float("inf")
                                                       or 1e-37 <= abs(v) <= 1e37)


# per topic, told apart as on_message does ("" is its else: a device status), each key its INSERT reads and what that
# key's column takes. on_message reads counter and fps with d.get(), so either may be absent: DEFAULT is what it reads
FIELDS = {"events/": {"event_id": _key, "device_id": _text, "kind": lambda v: v in ("enter", "exit"),
                      "track_id": _int(32), "ts_ms": _int(64), "counter": _text},
          "occupancy/": {"device_id": _text, "counter": _text, "in_zone": _int(32), "ts_ms": _int(64)},
          "truth/": {"seed": _int(32), "expected_visits": _int(32), "ts_ms": _int(64)},
          "": {"state": _text, "fps": _fps}}
DEFAULT = {"counter": "debounced", "fps": None}  # any other key absent reads as None, which its check refuses


def valid(msg):
    """msg's JSON object if every value on_message's INSERT reads from it fits its column. Else None, after one line
    to stdout: the topic, why, the payload's size and its first PREFIX bytes, and the count so far."""
    global dropped
    topic = "?"
    try:
        topic = msg.topic  # paho decodes the topic name on each read, and raises if it is not UTF-8
        d = json.loads(msg.payload)  # ValueError if not UTF-8 JSON, RecursionError if nested too deep
    except (ValueError, RecursionError) as e:
        why = f"{type(e).__name__}: {e}"
    else:
        fields = next(f for prefix, f in FIELDS.items() if topic.startswith(prefix))
        why = "not a JSON object" if not isinstance(d, dict) else next(
            (f"{'bad' if k in d else 'missing'} {k}" for k, ok in fields.items() if not ok(d.get(k, DEFAULT.get(k)))),
            None)
        if why is None and not topic.startswith(("events/", "occupancy/")) and not _key(topic.split("/")[1]):
            why = "bad topic"  # its device id, which on_message stores as ground_truth's or device_status's key
        if why is None:
            return d
    dropped += 1
    print(f"ingest: dropped a message on {topic[:PREFIX]!r}: {why}; payload {len(msg.payload)} bytes, "
          f"{msg.payload[:PREFIX]!r}; {dropped} dropped since start")
    return None


if __name__ == "__main__":
    main()
