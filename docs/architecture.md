# Architecture

```mermaid
flowchart LR
  V[media/sample.mp4] --> F[ffmpeg loop] --> M[MediaMTX RTSP]
  M --> E[edge: YOLO + tracker + zone state machine]
  E -- MQTT QoS1 --> T[Toxiproxy: bandwidth / latency / outage]
  T --> Q[EMQX]
  S[sim: synthetic traffic, naive + debounced] -- MQTT QoS1 --> T
  Q --> I[ingest] --> P[(Postgres)] --> G[Grafana]
  H[harness: replay + pytest] -. same replay.zones module .-> E
```

## Decisions

| Decision | Why | Trade-off |
|---|---|---|
| Zone logic lives in `harness/replay`, imported by the edge service | One implementation, tested offline, deployed unchanged | Edge image build context is the repo root |
| Footpoint anchor by default | Ground contact is stable under shadows and tall vehicles | Wrong for overhead cameras, where centroid is better |
| Enter event held until the visit closes | Short visits emit nothing, so downstream never has to retract | Enter events arrive together with the exit, up to a whole visit late |
| Edge reaches the broker only through Toxiproxy | Every network scenario is a one-line, scriptable change | TCP-level only |
| ULID event IDs + idempotent insert | At-least-once delivery becomes effectively exactly-once | Requires a unique index on the hot table |
| Retained status + last-will | Portal can tell "unreachable" from "reachable but not processing frames" | Slightly more broker state |

## Extending

- **New failure mode:** add a generator in `harness/scripts/make_fixtures.py`, then a failing test for the naive counter and a passing one for the fix.
- **Real footage:** `harness/scripts/dump_tracks.py` writes the same JSONL schema from any video.
- **More cameras:** duplicate the `camera` and `edge` services with a new path and `DEVICE_ID`.
