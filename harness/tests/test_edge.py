"""services/edge/src/main.py driven on the host: paho, ulid and ultralytics stubbed at the import boundary, numpy
blocked as in CI ([dev] only), a fake clock, no MQTT or RTSP. Pins what the edge runs and publishes per path."""
import importlib.util
import itertools
import json
import os
import sys
import types
from pathlib import Path

import pytest

from replay import trackers
from replay.schema import TrackBox
from replay.zones import DebouncedZoneCounter

EDGE = Path(__file__).resolve().parents[2] / "services" / "edge" / "src" / "main.py"
RTSP = "rtsp://mediamtx:8554/cam1"
POLYGON = "[[552,112],[334,235],[882,581],[1020,277]]"  # docker-compose.yml's ZONE_POLYGON
ZONE = [tuple(p) for p in json.loads(POLYGON)]
T0 = 1_700_000_000_000  # ms
_names = itertools.count()


class Clock:
    """time_ns: T0 + 40 ms a call; the edge reads it once a frame, so 0-based frame n reads ts(n). monotonic: 11 s
    more for each frame in `beats` that has read the clock, so the 10 s heartbeat fires on exactly those frames.
    perf_counter: a clock of its own that stands still; load_edge moves it 0.5 ms in each contained() call, 50 ms in
    each Detection built and 100 ms in each tracker update, so filter_ms holds contained()'s time and nothing else."""

    def __init__(self):
        self.reads, self.beats, self.perf = 0, (), 0.0

    @staticmethod
    def ts(n):
        return T0 + 40 * (n + 1)

    def time_ns(self):
        self.reads += 1
        return self.ts(self.reads - 1) * 1_000_000

    def monotonic(self):
        return 11.0 * sum(k < self.reads for k in self.beats)

    def perf_counter(self):
        return self.perf


class _List(list):
    """A tensor's stand-in: .tolist(), and bool() raises past one element, as torch's does, so only `is not None`
    tells a frame with track IDs from one without."""

    def tolist(self):
        return list(self)

    def __bool__(self):
        if len(self) > 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return len(self) == 1 and bool(self[0])


def box(x, y, score=0.9, cls=2, w=80, h=60):
    """A detector box with its footpoint at (x, y): (xyxy, score, class index)."""
    return (x - w // 2, y - h, x + w // 2, y), score, cls


class Model:
    """YOLO stand-in with COCO's vehicle class names. `script` holds each frame's boxes. track() yields them with
    ids (1, 2, ... in frame order), predict() without, as Ultralytics' predict results carry none; both record their
    call."""
    names = {2: "car", 5: "bus", 7: "truck"}

    def __init__(self, script):
        self.script, self.calls = script, []

    def _results(self, ids):
        for frame in self.script:
            yield types.SimpleNamespace(boxes=types.SimpleNamespace(
                xyxy=_List(list(b) for b, _, _ in frame), conf=_List(s for _, s, _ in frame),
                cls=_List(float(c) for _, _, c in frame),
                id=_List(float(i) for i in range(1, len(frame) + 1)) if ids and frame else None))

    def track(self, *args, **kwargs):
        self.calls.append(("track", args, kwargs))
        return self._results(ids=True)

    def predict(self, *args, **kwargs):
        self.calls.append(("predict", args, kwargs))
        return self._results(ids=False)


def load_edge(monkeypatch, env, model):
    """main.py imported fresh under a name of its own with `env` set (it reads env at import) and every other CONTAIN_*
    and TRACKER cleared, its clock replaced by a Clock. Returns (module, record of what it published and created).
    A tracker it creates must be create("bytetrack") and is greedy_iou behind a recorder of every update: pure Python
    and stateful, so the on path runs without numpy."""
    rec = types.SimpleNamespace(published=[], created=[], updates=[], clock=Clock())

    class Client:
        def __init__(self, *a, **kw): pass
        def reconnect_delay_set(self, **kw): pass
        def max_queued_messages_set(self, n): pass
        def will_set(self, *a, **kw): pass
        def connect_async(self, *a, **kw): pass
        def loop_start(self): pass

        def publish(self, topic, payload, qos, retain=False):
            rec.published.append((topic, json.loads(payload), qos, retain))

    mqtt = types.SimpleNamespace(Client=Client, CallbackAPIVersion=types.SimpleNamespace(VERSION2=2))
    for name, stub in {"numpy": None,  # import numpy -> ImportError, as in CI
                       "paho": types.SimpleNamespace(mqtt=types.SimpleNamespace(client=mqtt)),
                       "paho.mqtt": types.SimpleNamespace(client=mqtt), "paho.mqtt.client": mqtt,
                       "ulid": types.SimpleNamespace(ULID=lambda: "01TEST"),
                       "ultralytics": types.SimpleNamespace(YOLO=lambda *_: model)}.items():
        monkeypatch.setitem(sys.modules, name, stub)
    for k in [k for k in os.environ if k.startswith("CONTAIN_")] + ["TRACKER", "MIN_TRAVEL_PX", "OCCUPANCY_GAUGE_MS",
                                                                   "DEVICE_ID", "MODEL"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RTSP_URL", RTSP)
    monkeypatch.setenv("ZONE_POLYGON", POLYGON)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(f"edge_main_{next(_names)}", EDGE)
    edge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edge)
    c = rec.clock
    monkeypatch.setattr(edge, "time", types.SimpleNamespace(time_ns=c.time_ns, monotonic=c.monotonic,
                                                            perf_counter=c.perf_counter))

    def create(*args, **params):
        assert (args, params) == (("bytetrack",), {})
        rec.created.append(args)
        inner = trackers.create("greedy_iou")

        def update(frame, ts_ms, dets):
            rec.updates.append((frame, ts_ms, list(dets)))
            c.perf += 0.1  # the tracker's time: outside filter_ms
            return inner.update(frame, ts_ms, dets)
        return types.SimpleNamespace(update=update)
    monkeypatch.setattr(edge, "create", create, raising=False)
    for name, s in (("contained", 0.0005), ("Detection", 0.05)):  # neither is in main.py before the filter
        if (real := getattr(edge, name, None)) is not None:
            def timed(*a, _real=real, _s=s, **kw):
                c.perf += _s
                return _real(*a, **kw)
            monkeypatch.setattr(edge, name, timed)
    return edge, rec


def events(rec):
    return [(p["kind"], p["track_id"], p["ts_ms"]) for topic, p, *_ in rec.published if topic == "events/edge-01/zone"]


# what today's edge sends, per topic: payload keys, qos, retain. A2: the same publish; A3: nothing added over MQTT.
MQTT = {"events/edge-01/zone": ({"event_id", "device_id", "counter", "kind", "track_id", "ts_ms"}, 1, False),
        "occupancy/edge-01": ({"device_id", "counter", "in_zone", "ts_ms"}, 0, False),
        "devices/edge-01/status": ({"state", "fps"}, 1, True)}


def assert_mqtt_as_today(rec):
    assert rec.published
    for topic, p, qos, retain in rec.published:
        assert topic in MQTT and (set(p), qos, retain) == MQTT[topic], (topic, p, qos, retain)
        assert p.get("counter", "debounced") == "debounced", (topic, p)


DRIVE = [[box(680, y)] for y in range(120, 600, 8)]  # one car down through the zone and out, 8 px a frame


@pytest.mark.parametrize("env", [{}, {"CONTAIN_SHARE": "0"}, {"CONTAIN_SHARE": "0.0"}, {"CONTAIN_SHARE": "0e0"},
                                 {"CONTAIN_SHARE": "-0"}, {"TRACKER": "botsort.yaml"}],
                         ids=["unset", "0", "0.0", "0e0", "-0", "TRACKER=botsort.yaml"])
def test_off_path_is_todays_track_call(monkeypatch, env):
    model = Model(DRIVE)
    edge, rec = load_edge(monkeypatch, env, model)
    edge.main()
    # once, and nothing else: no predict(); share 0 in any spelling is off; TRACKER goes to model.track() as today
    assert model.calls == [("track", (RTSP,), {"tracker": env.get("TRACKER", "bytetrack.yaml"), "classes": [2, 5, 7],
                                               "agnostic_nms": True, "stream": True, "verbose": False})]
    assert rec.created == []
    counter = DebouncedZoneCounter(ZONE)
    want = [(e.kind, e.track_id, e.ts_ms) for n, frame in enumerate(DRIVE)
            for e in counter.update(TrackBox(n, Clock.ts(n), 1, frame[0][0]))]
    assert [k for k, _, _ in want] == ["enter", "exit"]
    assert events(rec) == want
    assert_mqtt_as_today(rec)
