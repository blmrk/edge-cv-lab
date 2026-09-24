#!/usr/bin/env bash
# Uplink drill on a freshly started lab (`make up`, sim profile), about 9 minutes:
# baseline, 64 kbps, latency, 2 minute outage, drain. Then checks that every event arrived exactly once
# and reports delivery lag per phase (harness/scripts/check_delivery.py). Events cannot arrive during the
# outage, so its backlog shows up in the post-outage window.
set -euo pipefail
cd "$(dirname "$0")/.."
s=chaos/scenarios.sh
trap '$s reset >/dev/null 2>&1 || true' EXIT  # an interrupted drill must not leave the uplink impaired
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
$s reset
t0=$(now); sleep 60
t1=$(now); $s 64k;     sleep 90; $s reset
t2=$(now); $s latency; sleep 90; $s reset
t3=$(now); $s outage 120
t4=$(now); sleep 150  # drain the backlog and let the scenes that spanned the outage finish
t5=$(now)
cd harness && python scripts/check_delivery.py --window baseline "$t0" "$t1" --window 64k "$t1" "$t2" \
  --window latency "$t2" "$t3" --window post-outage "$t4" "$t5"
