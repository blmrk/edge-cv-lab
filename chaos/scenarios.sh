#!/usr/bin/env bash
# Usage: chaos/scenarios.sh {64k|128k|256k|512k|latency|outage <sec>|reset}
set -euo pipefail
API=${TOXIPROXY:-http://localhost:8474}
toxic() { curl -fsS -X POST "$API/proxies/mqtt/toxics" -H 'Content-Type: application/json' -d "$1" >/dev/null; }
reset() { for t in $(curl -fsS "$API/proxies/mqtt/toxics" | grep -o '"name":"[^"]*"' | cut -d'"' -f4); do
            curl -fsS -X DELETE "$API/proxies/mqtt/toxics/$t" >/dev/null; done; }
case "${1:-}" in
  64k|128k|256k|512k) reset; kbps=${1%k}  # toxiproxy bandwidth rate is KB/s
    toxic "{\"type\":\"bandwidth\",\"stream\":\"upstream\",\"attributes\":{\"rate\":$((kbps/8))}}" ;;
  latency) reset; for s in upstream downstream; do  # toxiproxy defaults to downstream only: events would pass undelayed
             toxic "{\"type\":\"latency\",\"stream\":\"$s\",\"attributes\":{\"latency\":400,\"jitter\":150}}"; done ;;
  outage)  curl -fsS -X POST "$API/proxies/mqtt" -d '{"enabled":false}' >/dev/null; sleep "${2:-60}"
           curl -fsS -X POST "$API/proxies/mqtt" -d '{"enabled":true}'  >/dev/null ;;
  reset)   reset ;;
  *) echo "usage: $0 {64k|128k|256k|512k|latency|outage <sec>|reset}"; exit 1 ;;
esac
echo "applied: $*"
