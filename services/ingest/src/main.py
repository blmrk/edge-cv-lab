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
CREATE INDEX IF NOT EXISTS zone_events_received_at ON zone_events (received_at);"""


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
        c.subscribe([("events/+/zone", 1), ("devices/+/status", 1), ("truth/+", 1)])

    def on_message(_c, _u, msg):
        d = json.loads(msg.payload)
        if msg.topic.startswith("events/"):
            db.execute("INSERT INTO zone_events (event_id, device_id, kind, track_id, ts_ms, counter) "
                       "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                       (d["event_id"], d["device_id"], d["kind"], d["track_id"], d["ts_ms"],
                        d.get("counter", "debounced")))
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


if __name__ == "__main__":
    main()
