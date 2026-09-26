# Case study: every event exactly once over a bad uplink

> Status: measured in the lab on synthetic traffic from one device. Every figure comes from a command in Reproduce.

## Problem

Edge nodes on cellular or metered links publish zone events that end up in counts people act on. When the link drops,
events sent fire-and-forget are gone. When the broker restarts while a consumer is away, whatever the broker held for
that consumer can go with it. After a reconnect, at-least-once delivery can hand the same event over twice. None of
this shows up in a camera test: the counter is right on the device and wrong in the database.

## Reproducing it without hardware

- Device: the `sim` service replays seeded synthetic traffic (`replay.synth`) at twice real time, runs the naive and
  debounced counters on it, and publishes every event over MQTT with a ULID `event_id`. At the end of each scene it
  publishes that scene's ground truth, which tells the check the scene has finished.
- Uplink: Toxiproxy sits between the device and the broker (`chaos/scenarios.sh`): a 64 kbps cap, 400 ms latency with
  150 ms jitter in both directions, and a full outage.
- Broker and consumer: EMQX 5.8, and an `ingest` service that writes events to Postgres with
  `ON CONFLICT (event_id) DO NOTHING`.
- Check: `harness/scripts/check_delivery.py` replays every finished scene offline, with the same seed and the same
  counters, and compares the events it should have produced with the rows stored, per counter and kind: lost, extra and
  duplicated. It also reports delivery lag for naive events, which are published the moment they happen (debounced
  enters are held until the visit ends, so their lag is not delivery lag).
- Uplink drill (`make drill`, `chaos/drill.sh`, about 9 minutes on a fresh lab): 60 s clean, 90 s at 64 kbps, 90 s of
  latency, a 120 s outage, then 150 s to drain.
- Broker restart drill (`make broker-restart`, about 5 minutes on a fresh lab): once the first scene has finished,
  ingest stops, the broker restarts, and ingest stays away for 60 s while the device keeps publishing.

## Before and after

Each fix can be switched off for a run: `SIM_QOS=0` makes the device publish zone events at QoS 0, and
`DURABLE_SESSIONS=false` turns off EMQX's durable sessions. Ground truth and heartbeats stay at QoS 1 in every run,
since the check needs each scene's truth to know it finished.

| Run | Fix switched off | Scenes | Expected | Stored | Lost | Extra | Duplicates |
|---|---|---|---|---|---|---|---|
| Uplink drill | QoS 1 at the device (`SIM_QOS=0`) | 5 | 1388 | 926 | 462 | 0 | 0 |
| Uplink drill | none | 5 | 1388 | 1388 | 0 | 0 | 0 |
| Broker restart | durable sessions (`DURABLE_SESSIONS=false`) | 3 | 846 | 689 | 157 | 0 | 0 |
| Broker restart | none | 3 | 846 | 846 | 0 | 0 | 0 |

At QoS 0 a third of the drill's events never reach the database, and nothing on the device notices. The Paho client
keeps a QoS 1 message in a bounded outgoing store (5000 messages here) until the broker acknowledges it, and resends it
after a reconnect; a QoS 0 message is sent once and not kept (paho-mqtt 2.1.0, `Client.publish`).

The broker restart loses events even though the device publishes at QoS 1 throughout. The only switch between the two
runs is durable sessions, so the loss is between the broker and the consumer: without them, ingest's session and the
messages queued for it live in broker memory, and the restart drops them. With durable sessions EMQX keeps both on disk.

Delivery lag of naive events in the fixed uplink drill:

| window | events | median lag | max lag |
|---|---|---|---|
| clean | 195 | 0.09 s | 1.07 s |
| 64 kbps | 229 | 0.13 s | 6.01 s |
| latency | 143 | 0.86 s | 3.57 s |
| after the outage | 700 | 26.14 s | 120.05 s |

The outage turns into lag instead of loss: its events arrive once the link is back, the oldest 120.05 s late, the length
of the outage. In the QoS 0 run the window after the outage receives 272 naive events against 700, with a median lag of
0.11 s: there is no backlog because nothing was kept. Anything that buckets events by arrival time puts the backlog in
the wrong minute; dashboards and counts should use the event's own timestamp. The Grafana screenshot in the README
shows the flatline and the catch-up during an outage.

Stored duplicates were 0 in every run. Ingest does not count the inserts its `ON CONFLICT` clause turns away, so these
runs show that no duplicate got through, not how many redeliveries happened.

## Fixes in place

| Failure | Fix | Where |
|---|---|---|
| Events published while the link is down are lost | QoS 1, persistent session (`clean_session=False`), bounded outgoing store, reconnect backoff 1 to 60 s | `services/sim`, `services/edge` |
| A broker restart drops the offline consumer's queue | EMQX durable sessions on disk (single-node discovery, which the local storage backend needs) | `docker-compose.yml` |
| A redelivered event is stored twice | one ULID `event_id` generated once per event; `ON CONFLICT (event_id) DO NOTHING` | `services/sim`, `services/edge`, `services/ingest` |
| A device that has died still looks online | heartbeat with FPS every 10 s, MQTT last will marks it offline | `services/sim`, `services/edge` |

## What did not work

- The first latency scenario delayed nothing that mattered: Toxiproxy's latency toxic applies to the downstream
  direction by default, so events went up undelayed. `chaos/scenarios.sh` now adds it to both directions.
- QoS 1 at the device is not enough on its own: the broker restart run above loses 157 events with the device at QoS 1
  the whole time. Delivery guarantees have to hold on both hops, device to broker and broker to consumer.

## Limitations

- One simulated device, synthetic traffic at twice real time, one broker node, and every container on one host.
- The outgoing store was not filled in these runs. When it is full, `publish` returns `MQTT_ERR_QUEUE_SIZE`, and neither
  the sim nor the edge checks the return code, so events past the limit would be dropped without a log line.
- Duplicate protection is shown only by its result (no duplicate stored), not by a count of redeliveries.

## Reproduce

Each run starts from a fresh lab. Add `GRAFANA_PORT=3001` to `make up` if port 3000 is taken. The runs in the table used
commits 4acd1c7 (both uplink drills and the broker restart without durable sessions) and a19ddb4 (the broker restart
with them); the services and compose file are the same at both.

```bash
make down && SIM_QOS=0 make up && make drill                        # before: zone events at QoS 0
make down && make up && make drill                                  # after
make down && DURABLE_SESSIONS=false make up && make broker-restart  # before: no durable sessions
make down && make up && make broker-restart                         # after
make down
```

A run with a fix switched off ends with a non-zero exit, since the check fails on any lost event; the table above is
printed before it exits.
