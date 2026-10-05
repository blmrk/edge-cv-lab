# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""services/ingest/src/main.py driven on the host: paho and psycopg stubbed at the import boundary, main() run to
loop_forever() (which returns at once), then on_message fed messages by hand. A bad message (not JSON, not UTF-8, a key
missing, a value its column cannot take) must not raise out of on_message: paho would send no PUBACK, the persistent
session (clean_session=False) and a retained status would bring the message back, and ingest would crash-loop on it.
It must not reach the SQL either, where Postgres would raise on it. Pins what a good message stores, and the one
bounded log line a bad one gets."""
import importlib.util
import itertools
import json
import sys
import types
from pathlib import Path

import pytest

INGEST = Path(__file__).resolve().parents[2] / "services" / "ingest" / "src" / "main.py"
DB_URL = "postgresql://postgres:lab@postgres:5432/lab"  # docker-compose.yml's
T0 = 1_700_000_000_000  # ms: past int32, as every real ts_ms is
_names = itertools.count()

# main.py's SQL before the fix, copied: a good message must still run exactly this, with exactly these parameters
EVENTS_SQL = ("INSERT INTO zone_events (event_id, device_id, kind, track_id, ts_ms, counter) "
              "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING")
OCCUPANCY_SQL = "INSERT INTO zone_occupancy (device_id, counter, in_zone, ts_ms) VALUES (%s,%s,%s,%s)"
TRUTH_SQL = "INSERT INTO ground_truth VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING"
STATUS_SQL = ("INSERT INTO device_status (device_id, state, fps) VALUES (%s,%s,%s) "
              "ON CONFLICT (device_id) DO UPDATE SET state=EXCLUDED.state, fps=EXCLUDED.fps, updated_at=now()")

EV, OCC, TRUTH, STATUS = "events/edge-01/zone", "occupancy/edge-01", "truth/sim-01", "devices/edge-01/status"
ULID = "01J9ZQ3X8K7V5N2M4P6R8T0W1Y"
EVENT = {"event_id": ULID, "device_id": "edge-01", "counter": "debounced", "kind": "enter", "track_id": 7, "ts_ms": T0}
# per topic: a good message as the edge or sim sends it, and the one execute() it must make
GOOD = {EV: (EVENT, (EVENTS_SQL, (ULID, "edge-01", "enter", 7, T0, "debounced"))),
        OCC: ({"device_id": "edge-01", "counter": "debounced", "in_zone": 3, "ts_ms": T0},
              (OCCUPANCY_SQL, ("edge-01", "debounced", 3, T0))),
        TRUTH: ({"seed": 11, "expected_visits": 42, "ts_ms": T0}, (TRUTH_SQL, ("sim-01", 11, 42, T0))),
        STATUS: ({"state": "online", "fps": 9.8}, (STATUS_SQL, ("edge-01", "online", 9.8)))}


class Msg:
    """paho's MQTTMessage as on_message reads it: the topic name kept as bytes and decoded, strictly, on each read of
    .topic (paho 2.x still calls on_message when that decode fails); the payload as bytes, a dict given is JSON; the
    QoS ingest subscribed at, unless given: 0 on occupancy/, 1 on the rest."""

    def __init__(self, topic, payload, qos=None, retain=False):
        self._topic = topic if isinstance(topic, bytes) else topic.encode()
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.qos = (0 if self._topic.startswith(b"occupancy/") else 1) if qos is None else qos
        self.retain = retain

    @property
    def topic(self):
        return self._topic.decode("utf-8")


def load_ingest(monkeypatch):
    """main.py imported fresh under a name of its own, paho and psycopg stubbed in sys.modules, the env compose sets,
    and main() run through. Returns (module, record): rec.sql every db.execute(sql, params), rec.client the one MQTT
    client, with the on_connect and on_message main() gave it; set rec.fail to an exception for execute() to raise."""
    rec = types.SimpleNamespace(connects=[], sql=[], clients=[], fail=None)

    class Connection:  # psycopg's, autocommit: ingest runs every statement on it directly
        def execute(self, sql, params=None):
            rec.sql.append((sql, params))
            if rec.fail:  # set by a test: Postgres down
                raise rec.fail

    def connect(url, **kwargs):
        rec.connects.append((url, kwargs))
        return Connection()

    class Client:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs, self.calls = args, kwargs, []
            rec.clients.append(self)

        def subscribe(self, topics): self.calls.append(("subscribe", topics))
        def connect(self, *args): self.calls.append(("connect", args))
        def loop_forever(self): self.calls.append(("loop_forever",))

    mqtt = types.SimpleNamespace(Client=Client, CallbackAPIVersion=types.SimpleNamespace(VERSION2=2))
    psycopg = types.SimpleNamespace(connect=connect, OperationalError=type("OperationalError", (Exception,), {}))
    for name, stub in {"paho": types.SimpleNamespace(mqtt=types.SimpleNamespace(client=mqtt)),
                       "paho.mqtt": types.SimpleNamespace(client=mqtt), "paho.mqtt.client": mqtt,
                       "psycopg": psycopg}.items():
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("MQTT_HOST", "emqx")
    spec = importlib.util.spec_from_file_location(f"ingest_main_{next(_names)}", INGEST)
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)
    ingest.main()
    (rec.client,) = rec.clients
    return ingest, rec


def deliver(rec, topic, payload, **kw):
    """One message into on_message, as paho's network thread calls it. Returns the SQL it ran."""
    before = len(rec.sql)
    rec.client.on_message(rec.client, None, Msg(topic, payload, **kw))
    return rec.sql[before:]


def drop_then_store(monkeypatch, capsys, topic, payload, reason, good_topic, **kw):
    """The bad message: on_message returns, runs no SQL and prints one line naming `reason`. Then a good message on
    good_topic: stored as today, and nothing printed, so ingest is still up and storing."""
    ingest, rec = load_ingest(monkeypatch)
    assert deliver(rec, topic, payload, **kw) == []
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and reason in out[0], out
    good, stored = GOOD[good_topic]
    assert deliver(rec, good_topic, good) == [stored]
    assert capsys.readouterr().out == ""


def test_startup_as_today(monkeypatch):
    ingest, rec = load_ingest(monkeypatch)
    assert rec.connects == [(DB_URL, {"autocommit": True})]
    assert rec.sql == [(ingest.DDL, None)]
    c = rec.client
    assert (c.args, c.kwargs) == ((2,), {"client_id": "ingest", "clean_session": False})
    assert c.calls == [("connect", ("emqx", 1883)), ("loop_forever",)]
    c.on_connect(c, None, None, None, None)
    assert c.calls[-1] == ("subscribe", [("events/+/zone", 1), ("devices/+/status", 1), ("truth/+", 1),
                                         ("occupancy/+", 0)])


# (e) what the edge and sim send, and the edges a too strict check would cut: each stored with today's SQL and params
K256 = "東" * 256  # 768 bytes of UTF-8
STORED = {
    "event": (EV, EVENT, GOOD[EV][1]),
    "event-sim-naive-exit": ("events/sim-01/zone", dict(EVENT, device_id="sim-01", counter="naive", kind="exit",
                                                        track_id=11007),
                             (EVENTS_SQL, (ULID, "sim-01", "exit", 11007, T0, "naive"))),
    "event-no-counter": (EV, {k: v for k, v in EVENT.items() if k != "counter"}, GOOD[EV][1]),
    "event-extra-key": (EV, dict(EVENT, extra=[1, {"a": None}]), GOOD[EV][1]),
    "event-int-top": (EV, dict(EVENT, track_id=2**31 - 1, ts_ms=2**63 - 1),
                      (EVENTS_SQL, (ULID, "edge-01", "enter", 2**31 - 1, 2**63 - 1, "debounced"))),
    "event-int-bottom": (EV, dict(EVENT, track_id=-2**31, ts_ms=-2**63),
                         (EVENTS_SQL, (ULID, "edge-01", "enter", -2**31, -2**63, "debounced"))),
    "event-non-ascii": (EV, dict(EVENT, device_id="caméra-01 東"),
                        (EVENTS_SQL, (ULID, "caméra-01 東", "enter", 7, T0, "debounced"))),
    "occupancy": (OCC, *GOOD[OCC]),
    "occupancy-empty-no-counter": (OCC, {"device_id": "edge-01", "in_zone": 0, "ts_ms": T0},
                                   (OCCUPANCY_SQL, ("edge-01", "debounced", 0, T0))),
    "truth": (TRUTH, *GOOD[TRUTH]),
    "status": (STATUS, *GOOD[STATUS]),
    "status-will-no-fps": (STATUS, {"state": "offline"}, (STATUS_SQL, ("edge-01", "offline", None))),
    "status-fps-null": (STATUS, {"state": "online", "fps": None}, (STATUS_SQL, ("edge-01", "online", None))),
    "status-fps-0": (STATUS, {"state": "online", "fps": 0.0}, (STATUS_SQL, ("edge-01", "online", 0.0))),
    "status-fps-int": (STATUS, {"state": "online", "fps": 25}, (STATUS_SQL, ("edge-01", "online", 25))),
    # json.loads gives its one NaN object, json.decoder.NaN: == on a tuple holds only for that same object
    "status-fps-nan": (STATUS, b'{"state": "online", "fps": NaN}',
                       (STATUS_SQL, ("edge-01", "online", json.decoder.NaN))),
    "status-fps-inf": (STATUS, b'{"state": "online", "fps": Infinity}',
                       (STATUS_SQL, ("edge-01", "online", float("inf")))),
    "status-fps-neg-inf": (STATUS, b'{"state": "online", "fps": -Infinity}',
                           (STATUS_SQL, ("edge-01", "online", float("-inf")))),
    "occupancy-extra-key": (OCC, dict(GOOD[OCC][0], extra=1), GOOD[OCC][1]),
    "truth-extra-key": (TRUTH, dict(GOOD[TRUTH][0], device_id="sim-01"), GOOD[TRUTH][1]),
    "status-extra-key": (STATUS, dict(GOOD[STATUS][0], uptime_s=60), GOOD[STATUS][1]),
    # text in a primary key at 256 characters, the bound: event_id, and the device id a truth or status topic carries
    "event-id-256": (EV, dict(EVENT, event_id=K256), (EVENTS_SQL, (K256, "edge-01", "enter", 7, T0, "debounced"))),
    "truth-device-256": (f"truth/{K256}", GOOD[TRUTH][0], (TRUTH_SQL, (K256, 11, 42, T0))),
    "status-device-256": (f"devices/{K256}/status", GOOD[STATUS][0], (STATUS_SQL, (K256, "online", 9.8))),
}


@pytest.mark.parametrize("topic,payload,stored", STORED.values(), ids=STORED.keys())
def test_good_message_stored_as_today(monkeypatch, capsys, topic, payload, stored):
    ingest, rec = load_ingest(monkeypatch)
    sql = deliver(rec, topic, payload, retain=topic.startswith("devices/"))
    assert sql == [stored]
    assert all(type(a) is type(b) for a, b in zip(sql[0][1], stored[1])), sql  # 0.0 stays a float, 25 an int
    assert capsys.readouterr().out == ""


# (a) not JSON at all, on the events topic
NOT_JSON = {"text": (b"not json", "JSONDecodeError"), "empty": (b"", "JSONDecodeError"),
            "truncated": (b'{"event_id": "01', "JSONDecodeError"),
            "trailing-brace": (json.dumps(EVENT).encode() + b"}", "JSONDecodeError"),
            "nested-100k": (b"[" * 100_000, "RecursionError"),
            "int-5000-digits": (b"1" * 5000, "ValueError")}


@pytest.mark.parametrize("payload,reason", NOT_JSON.values(), ids=NOT_JSON.keys())
def test_not_json_dropped(monkeypatch, capsys, payload, reason):
    drop_then_store(monkeypatch, capsys, EV, payload, reason, EV)


# (b) JSON, but a key missing, a value of the wrong type or past its column's range, or not an object at all
def _without(key):
    return {k: v for k, v in EVENT.items() if k != key}


BAD_EVENT = {**{f"missing-{k}": (_without(k), f"missing {k}") for k in ("event_id", "device_id", "kind", "track_id",
                                                                        "ts_ms")},
             **{f"{k}={v!r}"[:40]: (dict(EVENT, **{k: v}), f"bad {k}") for k, v in [
                 ("ts_ms", str(T0)), ("ts_ms", float(T0)), ("ts_ms", True), ("ts_ms", None), ("ts_ms", [T0]),
                 ("ts_ms", 2**63), ("ts_ms", -2**63 - 1),
                 ("track_id", "7"), ("track_id", 7.0), ("track_id", 2**31), ("track_id", -2**31 - 1),
                 ("event_id", 123), ("event_id", "01J\x00"), ("device_id", None), ("device_id", {"id": 1}),
                 ("kind", "enters"), ("kind", "ENTER"), ("kind", None), ("counter", None), ("counter", 5),
                 ("counter", ["debounced"])]},
             **{f"not-object-{n}": (p, "not a JSON object") for n, p in [("list", b"[]"), ("null", b"null"),
                                                                         ("string", b'"enter"'), ("number", b"42")]}}


@pytest.mark.parametrize("payload,reason", BAD_EVENT.values(), ids=BAD_EVENT.keys())
def test_missing_key_or_wrong_type_dropped(monkeypatch, capsys, payload, reason):
    drop_then_store(monkeypatch, capsys, EV, payload, reason, EV)


# (c) not UTF-8: in the payload, a lone surrogate (as UTF-8 bytes or a JSON escape) that UTF-8 cannot carry, the topic
_EVENT_BYTES = json.dumps(EVENT).encode()
NOT_UTF8 = {"payload-byte-ff": (EV, _EVENT_BYTES.replace(b"edge-01", b"edge-\xff1"), "UnicodeDecodeError"),
            "payload-cut-sequence": (EV, b'{"event_id": "\xc3"}', "UnicodeDecodeError"),
            "payload-all-ff": (EV, b"\xff" * 64, "UnicodeDecodeError"),
            "surrogate-as-bytes": (EV, _EVENT_BYTES.replace(b"edge-01", b"edge-\xed\xa0\x80"), "bad device_id"),
            "surrogate-escaped": (EV, dict(EVENT, device_id="\ud800"), "bad device_id"),
            "topic-byte-ff": (b"events/edge-\xff/zone", EVENT, "UnicodeDecodeError")}


@pytest.mark.parametrize("topic,payload,reason", NOT_UTF8.values(), ids=NOT_UTF8.keys())
def test_not_utf8_dropped(monkeypatch, capsys, topic, payload, reason):
    drop_then_store(monkeypatch, capsys, topic, payload, reason, EV)


# (d) the same on the other three topics: occupancy, device status (retained, so redelivered on every reconnect) and
# ground truth
def _bad(topic, **change):
    return dict(GOOD[topic][0], **change)


BAD_OTHER = {
    "occupancy-not-json": (OCC, b"three", "JSONDecodeError"),
    "occupancy-not-object": (OCC, b"3", "not a JSON object"),
    "occupancy-missing-in_zone": (OCC, {"device_id": "edge-01", "ts_ms": T0}, "missing in_zone"),
    "occupancy-missing-device_id": (OCC, {"in_zone": 3, "ts_ms": T0}, "missing device_id"),
    "occupancy-missing-ts_ms": (OCC, {"device_id": "edge-01", "in_zone": 3}, "missing ts_ms"),
    "occupancy-in_zone-str": (OCC, _bad(OCC, in_zone="3"), "bad in_zone"),
    "occupancy-in_zone-float": (OCC, _bad(OCC, in_zone=1.5), "bad in_zone"),
    "occupancy-in_zone-bool": (OCC, _bad(OCC, in_zone=True), "bad in_zone"),
    "occupancy-in_zone-2^31": (OCC, _bad(OCC, in_zone=2**31), "bad in_zone"),
    "occupancy-counter-null": (OCC, _bad(OCC, counter=None), "bad counter"),
    "occupancy-ts_ms-str": (OCC, _bad(OCC, ts_ms="now"), "bad ts_ms"),
    "status-not-json": (STATUS, b"online", "JSONDecodeError"),
    "status-not-object": (STATUS, b'["online"]', "not a JSON object"),
    "status-empty-object": (STATUS, {}, "missing state"),
    "status-fps-only": (STATUS, {"fps": 9.8}, "missing state"),
    "status-state-null": (STATUS, _bad(STATUS, state=None), "bad state"),
    "status-state-int": (STATUS, _bad(STATUS, state=1), "bad state"),
    "status-state-nul": (STATUS, _bad(STATUS, state="on\x00line"), "bad state"),
    "status-fps-str": (STATUS, _bad(STATUS, fps="fast"), "bad fps"),
    "status-fps-bool": (STATUS, _bad(STATUS, fps=True), "bad fps"),
    "status-fps-list": (STATUS, _bad(STATUS, fps=[9.8]), "bad fps"),
    "status-fps-past-real": (STATUS, _bad(STATUS, fps=1e39), "bad fps"),
    "status-fps-past-real-negative": (STATUS, _bad(STATUS, fps=-1e39), "bad fps"),
    "status-fps-past-real-int": (STATUS, _bad(STATUS, fps=10**39), "bad fps"),
    "status-fps-under-real": (STATUS, _bad(STATUS, fps=1e-50), "bad fps"),
    "truth-not-json": (TRUTH, b"{seed: 11}", "JSONDecodeError"),
    "truth-missing-seed": (TRUTH, {"expected_visits": 42, "ts_ms": T0}, "missing seed"),
    "truth-missing-expected_visits": (TRUTH, {"seed": 11, "ts_ms": T0}, "missing expected_visits"),
    "truth-missing-ts_ms": (TRUTH, {"seed": 11, "expected_visits": 42}, "missing ts_ms"),
    "truth-seed-str": (TRUTH, _bad(TRUTH, seed="11"), "bad seed"),
    "truth-seed-2^31": (TRUTH, _bad(TRUTH, seed=2**31), "bad seed"),
    "truth-expected_visits-float": (TRUTH, _bad(TRUTH, expected_visits=1.5), "bad expected_visits"),
    "truth-ts_ms-null": (TRUTH, _bad(TRUTH, ts_ms=None), "bad ts_ms"),
}


@pytest.mark.parametrize("topic,payload,reason", BAD_OTHER.values(), ids=BAD_OTHER.keys())
def test_bad_occupancy_status_truth_dropped(monkeypatch, capsys, topic, payload, reason):
    drop_then_store(monkeypatch, capsys, topic, payload, reason, topic, retain=topic == STATUS)


# text that goes into a primary key, too long for it: Postgres refuses a btree index row over 2704 bytes, so the INSERT
# would raise. event_id from the payload; on truth and status, the device id on_message takes from the topic
LONG = "k" * 257  # one past KEY: pins the bound from above, with the 256-char rows pinning it from below
LONG_KEY = {"event_id": (EV, dict(EVENT, event_id=LONG), "bad event_id", EV),
            "truth-topic-device": (f"truth/{LONG}", GOOD[TRUTH][0], "bad topic", TRUTH),
            "status-topic-device": (f"devices/{LONG}/status", GOOD[STATUS][0], "bad topic", STATUS)}


@pytest.mark.parametrize("topic,payload,reason,good_topic", LONG_KEY.values(), ids=LONG_KEY.keys())
def test_primary_key_too_long_dropped(monkeypatch, capsys, topic, payload, reason, good_topic):
    drop_then_store(monkeypatch, capsys, topic, payload, reason, good_topic, retain=good_topic == STATUS)


# (f) one line per bad message: the topic, the reason, the payload's size and first bytes, never more of it
HUGE = "A" * 1_000_000
HOSTILE = [(EV, b'{"event_id": "' + HUGE.encode(), "JSONDecodeError"),  # 1 MB, not JSON
           (EV, dict(EVENT, ts_ms=HUGE), "bad ts_ms"),  # 1 MB of JSON, and all of it the value of the wrong type
           (STATUS, {"fps": 1, "pad": HUGE}, "missing state"),
           (TRUTH, b"\xff" * 1_000_000, "UnicodeDecodeError"),
           ("events/" + "A" * 65_000 + "/zone", b"{", "JSONDecodeError")]  # a topic near MQTT's 65535-byte limit


def test_bad_message_logged_once_bounded_and_counted(monkeypatch, capsys):
    ingest, rec = load_ingest(monkeypatch)
    for n, (topic, payload, reason) in enumerate(HOSTILE, 1):
        assert deliver(rec, topic, payload) == []
        (line,) = capsys.readouterr().out.splitlines()
        assert topic[:40] in line and reason in line, line[:600]
        # 1 MB in, a line of a few hundred characters (64 bytes of \xff repr'd are 259): a prefix, never all of it
        assert len(line) < 600 and "A" * 100 not in line, line[:600]
        assert ingest.dropped == n
    # the first one again, as a redelivery would bring it: logged and counted again, still not raised
    assert deliver(rec, *HOSTILE[0][:2]) == []
    (line,) = capsys.readouterr().out.splitlines()
    assert '{"event_id": "AAAA' in line and "JSONDecodeError" in line, line[:400]
    assert ingest.dropped == len(HOSTILE) + 1
    # a good message neither logs nor counts
    for topic, (good, stored) in GOOD.items():
        assert deliver(rec, topic, good) == [stored]
    assert capsys.readouterr().out == "" and ingest.dropped == len(HOSTILE) + 1


def test_database_error_still_raises(monkeypatch, capsys):
    """Postgres down: execute() raises on a good message. on_message lets it out, so paho sends no PUBACK and
    the QoS 1 message comes back after the restart; caught, it would be acked and lost. Not a bad message:
    nothing logged, nothing counted."""
    ingest, rec = load_ingest(monkeypatch)
    rec.fail = ingest.psycopg.OperationalError("the connection is lost")
    for topic, (good, stored) in GOOD.items():
        with pytest.raises(ingest.psycopg.OperationalError):
            deliver(rec, topic, good)
        assert rec.sql[-1] == stored
    assert capsys.readouterr().out == "" and ingest.dropped == 0
