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
from replay.detections import Detection, frames
from replay.phantoms import _track
from replay.schema import TrackBox
from replay.secondbox import contained
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


def test_on_path_calls_predict_and_one_bytetrack(monkeypatch):
    model = Model(DRIVE)
    edge, rec = load_edge(monkeypatch, {"CONTAIN_SHARE": "0.9"}, model)
    rec.clock.beats = (20, 40)  # heartbeats crossed: the one tracker outlives them
    edge.main()
    # once, and no track(): conf 0.1 is what track() forces for itself, predict() alone would take 0.25
    assert model.calls == [("predict", (RTSP,), {"conf": 0.1, "classes": [2, 5, 7], "agnostic_nms": True,
                                                 "stream": True, "verbose": False})]
    assert rec.created == [("bytetrack",)]  # once for the process, not per frame or per heartbeat


@pytest.mark.parametrize("env", [{}, {"CONTAIN_SHARE": "0.9"}], ids=["off", "on"])
def test_heartbeat_clock_starts_before_the_source_call(monkeypatch, env):
    # Ultralytics' predict(), which track() runs through, sets its predictor up before it returns the stream. As today,
    # that time falls in the first heartbeat window: the clock is read before track() or predict() is called.
    order = []

    class Recorded(Model):
        def _results(self, ids):
            order.append("source")
            return super()._results(ids)

    edge, rec = load_edge(monkeypatch, env, Recorded(DRIVE))
    mono = edge.time.monotonic
    monkeypatch.setattr(edge.time, "monotonic", lambda: order.append("clock") or mono())
    edge.main()
    assert order[:2] == ["clock", "source"]


def pair(x, y, inside, outer_cls=2, inner_cls=2):
    """A 100 x 60 box scored 0.9 and a 40 x 40 one scored 0.5 with `inside` of its own area in the first."""
    stick = round(40 * (1 - inside))  # px of the small box past the big one's right edge
    return [((x, y, x + 100, y + 60), 0.9, outer_cls),
            ((x + 60 + stick, y + 10, x + 100 + stick, y + 50), 0.5, inner_cls)]


# a lone car parked above the zone, scored 0.15: contained() keeps it, and it is under bytetrack.yaml's 0.25 and
# greedy_iou's 0.3, so the tracker gets it and greedy_iou drops it (as ByteTrack drops many of conf 0.1's boxes)
LOW = ((450, 0, 490, 40), 0.15, 2)


def detections(script, ts, only=None):
    """The fake's boxes as Detections, frame n at ts(n)."""
    return [Detection(n, ts(n), tuple(b), s, Model.names[c]) for n, frame in enumerate(script)
            if only is None or n in only for b, s, c in frame]


def scene():
    """56 frames: a car down through the zone with a second box wholly inside its own, three pairs parked above the
    zone (a car in a car 0.95 inside, a car in a truck 0.95 inside, a car in a car 0.85 inside) and LOW beside them,
    so (i) sees a score cut before the tracker. Frame 30 is empty."""
    return [[] if n == 30 else [box(680, y), ((644, y - 50, 716, y - 2), 0.5, 2)]
            + pair(0, 0, 0.95) + pair(150, 0, 0.95, outer_cls=7) + pair(300, 0, 0.85) + [LOW]
            for n, y in enumerate(range(120, 568, 8))]


@pytest.mark.parametrize("share", [0.8, 0.9])
@pytest.mark.parametrize("same_env, same", [({}, True), ({"CONTAIN_SAME_CLASS": "0"}, False)],
                         ids=["same_class_unset", "same_class_0"])
def test_tracker_sees_the_replays_input(monkeypatch, share, same_env, same):
    model = Model(scene())
    edge, rec = load_edge(monkeypatch, {"CONTAIN_SHARE": str(share), **same_env}, model)
    rec.clock.beats = (20,)
    edge.main()
    # every frame, empty ones too, numbered from 0 and never reset
    assert [f for f, _, _ in rec.updates] == list(range(len(model.script)))
    D = detections(model.script, {f: ts for f, ts, _ in rec.updates}.get)
    keep = lambda ds: contained(ds, share, same)  # noqa: E731
    assert rec.updates == list(frames(keep(D)))  # holding what the replay's filter keeps, low scores too
    counter = DebouncedZoneCounter(ZONE)  # update only, as the edge: never zones.run, which flushes at the end
    want = sorted((e.kind, e.track_id, e.ts_ms) for b in _track(D, {}, keep, tracker="greedy_iou")
                  for e in counter.update(b))
    assert [k for k, _, _ in want] == ["enter", "exit"]  # the car once: its second box never enters
    assert sorted(events(rec)) == want
    assert_mqtt_as_today(rec)


def test_heartbeat_log_counts_each_window(monkeypatch, capsys):
    # LOW is in every frame with boxes: kept by contained() and dropped by the tracker, so `dropped` counts the
    # filter's drops, not the boxes left untracked. Frames 0-9: a lone car and a pair, 1 of 4 dropped; frames 10-24:
    # the pair and a pair 0.85 inside (kept at 0.9), 1 of 5 dropped, frame 17 empty; frames 25-39: no box at all, a
    # line and a heartbeat all the same, on an empty frame; frames 40-54: the 0.85 pair, nothing to drop; frames 55-59
    # end before a fifth heartbeat
    def frame(n):
        if n == 17 or 25 <= n < 40:
            return []
        if n < 10:
            return [box(680, 600), LOW] + pair(0, 0, 0.95)
        return [LOW] + (pair(0, 0, 0.95) if n < 25 else []) + pair(300, 0, 0.85)
    script = [frame(n) for n in range(60)]
    windows = (range(0, 10), range(10, 25), range(25, 40), range(40, 55))
    beats, want = tuple(w[-1] for w in windows), []
    for w in windows:
        D = detections(script, Clock.ts, only=w)
        want.append(f"contain ts_ms={Clock.ts(w[-1])} frames={len(w)} dets={len(D)} "
                    f"dropped={len(D) - len(contained(D, 0.9, True))} "
                    f"filter_ms={0.5 * len(w):.1f}")  # 0.5 ms in each contained(), none from the tracker or Detection
    # the {state, fps} heartbeat, the same on both paths: the window's frames over the fake clock's 11 s
    status = [{"state": "online", "fps": round(len(w) / 11, 1)} for w in windows]
    edge, rec = load_edge(monkeypatch, {"CONTAIN_SHARE": "0.9"}, Model(script))
    rec.clock.beats = beats
    edge.main()
    assert [line for line in capsys.readouterr().out.splitlines() if "contain ts_ms=" in line] == want
    assert [p for topic, p, *_ in rec.published if topic == "devices/edge-01/status"] == status
    assert_mqtt_as_today(rec)
    edge, rec = load_edge(monkeypatch, {}, Model(script))
    rec.clock.beats = beats
    edge.main()
    assert [p for topic, p, *_ in rec.published if topic == "devices/edge-01/status"] == status
    assert_mqtt_as_today(rec)
    assert "contain ts_ms=" not in capsys.readouterr().out


def test_gauge_samples_every_frame_on_the_on_path(monkeypatch):
    # a car parked in the zone, and no boxes on the frames a sample falls due on (each 1000 ms: frames 0, 25 and 50)
    model = Model([[] if n % 25 == 0 else [box(680, 300)] for n in range(60)])
    edge, rec = load_edge(monkeypatch, {"CONTAIN_SHARE": "0.9", "OCCUPANCY_GAUGE_MS": "500"}, model)
    edge.main()
    assert [call for call, _, _ in model.calls] == ["predict"]
    sampled = [p["ts_ms"] for topic, p, *_ in rec.published if topic == "occupancy/edge-01"]
    assert sampled == [Clock.ts(n) for n in (0, 25, 50)]
    assert_mqtt_as_today(rec)


class StreamEnded(Exception):
    pass


class Endless(Model):
    """A stream that does not end, as RTSP's: past its script it fails, so a line main() prints after its frame loop
    never shows."""

    def _results(self, ids):
        yield from super()._results(ids)
        raise StreamEnded


ON = "contain: on share={} same_class={} conf=0.1 tracker=bytetrack.yaml"


STARTUP = [
    ({}, "contain: off"),
    ({"CONTAIN_SHARE": "0"}, "contain: off"),
    ({"TRACKER": "botsort.yaml"}, "contain: off"),  # off: TRACKER is not refused (test 1: it goes to model.track())
    ({"CONTAIN_SHARE": "1"}, ON.format(1, 1)),
    ({"CONTAIN_SHARE": "0.9", "CONTAIN_SAME_CLASS": "0", "TRACKER": "bytetrack.yaml"}, ON.format(0.9, 0)),
    ({"CONTAIN_SHARE": "0.90"}, ON.format(0.9, 1)),  # the parsed share, printed with :g, not the string as set
    # SystemExit at import, before the model loads, naming these variables; set but empty is not unset
    *[({"CONTAIN_SHARE": v}, ("CONTAIN_SHARE", "CONTAIN_SAME_CLASS")) for v in ("-0.1", "1.5", "abc", "nan", "")],
    *[({"CONTAIN_SAME_CLASS": v}, ("CONTAIN_SAME_CLASS",)) for v in ("2", "true", "", " 1")],
    # with the filter on, TRACKER is bytetrack.yaml exactly: not padded, not a part of it, not another yaml named for it
    *[({"CONTAIN_SHARE": "0.9", "TRACKER": v}, ("TRACKER",))
      for v in ("botsort.yaml", "bytetrack.yaml ", "bytetrack", "bytetrack_birth.yaml", "")],
]


@pytest.mark.parametrize("env, want", STARTUP, ids=[" ".join(f"{k}={v!r}" for k, v in e.items()) or "unset"
                                                     for e, _ in STARTUP])
def test_startup_line_and_env_rules(monkeypatch, capsys, env, want):
    if isinstance(want, tuple):
        with pytest.raises(SystemExit) as exc:
            load_edge(monkeypatch, env, Model([]))
        assert all(v in str(exc.value) for v in want)
        return
    edge, _ = load_edge(monkeypatch, env, Endless([]))
    with pytest.raises(StreamEnded):  # live, the loop never ends: the line is out before it starts
        edge.main()
    assert capsys.readouterr().out == want + "\n"
