#!/usr/bin/env bash
# Broker restart drill on a freshly started lab (`make up`, sim profile), about 5 minutes. Once the first
# scene has finished, ingest goes away, the broker restarts, and ingest stays away for 60 s while the sim
# keeps publishing. EMQX durable sessions keep ingest's session and those events on disk, so the delivery
# check (harness/scripts/check_delivery.py) expects every event exactly once.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'docker compose start ingest >/dev/null 2>&1 || true' EXIT  # never leave ingest stopped
say() { echo "$(date -u +%H:%M:%S) $*"; }
fail() { echo "$*" >&2; exit 1; }
stored() { docker compose exec -T postgres psql -U postgres lab -tAc "SELECT coalesce(max(seed), 0) FROM ground_truth" 2>/dev/null || echo 0; }
sim_done() { docker compose logs --no-log-prefix sim 2>/dev/null | sed -n 's/^seed \([0-9]*\) done.*/\1/p' | tail -1; }
wait_until() {  # wait_until SECONDS WHAT CHECK: run CHECK every 3 s until it passes, fail after SECONDS
  local deadline=$((SECONDS + $1))
  until "$3"; do [ "$SECONDS" -lt "$deadline" ] || fail "timed out waiting for $2"; sleep 3; done
}
first_scene_stored() { [ "$(stored)" -ge 1 ]; }
broker_healthy() { [ "$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q emqx)" 2>/dev/null)" = healthy ]; }
target_stored() { [ "$(stored)" -ge "$target" ]; }

[ "$(stored)" -le "$(sim_done || true)" ] 2>/dev/null || [ "$(stored)" = 0 ] ||
  fail "Postgres holds scenes the running sim has not played (an earlier run?): make down && make up first"
wait_until 900 "the first scene" first_scene_stored
say "first scene finished; stopping ingest and restarting the broker"
docker compose stop ingest >/dev/null
docker compose restart emqx >/dev/null
wait_until 180 "the broker to report healthy" broker_healthy
say "broker healthy; ingest stays away for 60 s while the sim keeps publishing"
sleep 60
cur=$(sim_done)
[[ $cur =~ ^[0-9]+$ ]] || fail "could not read the sim's progress from its logs"
target=$((cur + 2))  # seed cur+1 spans the gap; cur+2 is one more scene of margin
docker compose start ingest >/dev/null
say "ingest back after seed $cur; waiting until seed $target is stored"
wait_until 900 "seed $target" target_stored
cd harness && python scripts/check_delivery.py
