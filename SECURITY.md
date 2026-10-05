# Security policy

## Scope

edge-cv-lab is a lab for reproducing video-analytics failures on one machine. It is not a product, and nothing in it
is hardened for the internet: fixed passwords, no TLS, containers running as root, ports open on every interface. Run
it on a machine and network you trust, and treat the defaults below as things to replace, not to deploy.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability"). If that is
not available, contact the author through the repository owner's GitHub profile, https://github.com/blmrk. Please do
not open a public issue for a vulnerability.

## Supported versions

Only `main`. There are no maintained releases.

## Lab-only defaults

Change each of these before any real deployment.

| Default | Where | Change to |
|---|---|---|
| Postgres superuser `postgres`, password `lab` | `docker-compose.yml:45` | a generated secret, passed in from outside the repo |
| ingest connects as that superuser | `docker-compose.yml:53` | its own role, limited to the tables it writes |
| Grafana reads Postgres as the superuser, password `lab`, `sslmode: disable`. Anonymous access gives every visitor the Viewer role, and per Grafana's docs a Viewer "can issue any possible query to a data source" (https://grafana.com/docs/grafana/latest/administration/data-source-management/). So anyone who can reach port 3000, without logging in, can send any SQL to this datasource, and it runs as superuser | `grafana/provisioning/datasources/postgres.yml:7`, `:9`, `:10`; `docker-compose.yml:86`, `:88-89` | a SELECT-only role, with TLS; anonymous access off |
| `psql -U postgres` through `docker exec` | `Makefile:36`, `chaos/broker-restart.sh:11`, `harness/scripts/check_delivery.py:28` | the read-only role |
| Grafana admin password `lab`; anonymous access on, role Viewer | `docker-compose.yml:90`; `:88-89` | a generated admin secret; anonymous access off |
| Grafana `secret_key` left at the default published in Grafana's own `conf/defaults.ini` (https://github.com/grafana/grafana/blob/v11.2.0/conf/defaults.ini, not read from the image; no `GF_SECURITY_SECRET_KEY` set); Grafana encrypts the secrets it stores, such as the datasource password, under it | `docker-compose.yml:87-91` | a generated secret, passed in from outside the repo |
| dashboards editable in the UI | `grafana/provisioning/dashboards/dashboards.yml:5`, `grafana/dashboards/edge-cv-lab.json:16` | `false` |
| EMQX dashboard login `admin` / `public` (no `EMQX_DASHBOARD__DEFAULT_PASSWORD` set). The same listener on 18083 serves EMQX's whole management REST API: `POST /api/v5/login` returns a token for that login, and `POST /api/v5/publish` publishes to any topic, so forged zone events, ground truth and device status reach Postgres through ingest; the API also manages clients and broker configuration | `docker-compose.yml:25`, `README.md:88`; EMQX v5.8.6 source (the local image's version label), https://github.com/emqx/emqx/tree/v5.8.6, not read from the image: `apps/emqx_dashboard/src/emqx_dashboard.erl:114`, `emqx_dashboard_api.erl:221-238`, `apps/emqx_management/src/emqx_mgmt_api_publish.erl:49-50` | a generated password, and the dashboard not published |
| MQTT with no authentication and no TLS (no `EMQX_AUTHENTICATION` set; per EMQX docs, anonymous clients are then allowed) | `docker-compose.yml:23-33`; clients at `services/edge/src/main.py:27`, `services/sim/src/main.py:20`, `services/ingest/src/main.py:72` | broker authentication, a credential per device, MQTT over TLS |
| ingest trusts every MQTT payload: `on_message` has no validation or error handling, and paho-mqtt re-raises callback errors before the QoS 1 PUBACK. One malformed QoS 1 or retained message (not JSON, a missing key, `kind` other than `enter`/`exit`) stops ingest, and restart plus the persistent session (`clean_session=False`) bring the same message back, so ingest crash-loops and stores no events. Any client that reaches the broker, or the EMQX publish API with the default login, can send one | `services/ingest/src/main.py:51-68`, `:70`; `docker-compose.yml:55` | validate each payload and log and drop a bad one inside `on_message`; broker authentication with a publish ACL per device |
| Toxiproxy API with no authentication, listening on all interfaces; anyone reaching 8474 can cut or degrade the uplink, or change the proxy's upstream and send the edge and sim MQTT traffic to any host the container can reach | `docker-compose.yml:38`, `:40`; `chaos/toxiproxy.json:1` | no Toxiproxy outside a lab |
| mediamtx on its stock config: per its docs, any client can read or publish `cam1` | `docker-compose.yml:6-10` | read and publish credentials |
| Published ports on every host interface: RTSP 8554, EMQX dashboard 18083, Toxiproxy 8474, Postgres 5432, Grafana 3000 | `docker-compose.yml:9`, `:25`, `:38`, `:46`, `:86` | bind to `127.0.0.1` or drop the mapping. RTSP already takes `RTSP_BIND=127.0.0.1`. MQTT 1883 is not published, but anyone reaching 18083 can publish through the EMQX REST API |
| No TLS anywhere: MQTT, RTSP, Grafana over HTTP, Postgres `sslmode: disable` | as above | TLS on every hop that leaves the host |
| Containers as root: edge, ingest and sim (no `USER`; uid 0 inside all three images); mediamtx, camera and toxiproxy (empty `Config.User`) | `services/*/Dockerfile`; `docker-compose.yml:8`, `:13`, `:36` | a non-root `USER`. emqx runs as `emqx` and grafana as 472; postgres drops to its own user per its image docs |
| Only edge has a memory limit (2 GB); no CPU or pids limits, `read_only`, `cap_drop` or `no-new-privileges` anywhere | `docker-compose.yml:71` | limits and hardening on every service |
| `ultralytics` settings in the edge image have `sync: true`, so the edge sends usage events to the vendor's analytics when online, and `YOLO_AUTOINSTALL` defaults to true, so missing requirements are pip-installed at runtime | `services/edge/Dockerfile:4`; inside the image, `ultralytics/utils/events.py:69`, `:87-93` and `ultralytics/utils/__init__.py:71` | `YOLO_OFFLINE=1` (or `sync=False`) and `YOLO_AUTOINSTALL=false` |
| Model weights downloaded at build time with no checksum. `.pt` checkpoints hold a pickle, and `ultralytics` loads them with `weights_only=False` by default (`ULTRALYTICS_SAFE_LOAD` unset), so a tampered file, or any file named by `MODEL`, runs code as root when it loads, at build and at every edge start; a `MODEL` that is a URL, or a vendor asset name not found locally, is downloaded first with no checksum | `services/edge/Dockerfile:10`; `services/edge/src/main.py:70`; inside the image, `ultralytics/utils/patches.py:203-204`, `ultralytics/utils/__init__.py:73`, `ultralytics/nn/tasks.py:1869`, `:1873` | a pinned sha256; `ULTRALYTICS_SAFE_LOAD=1` (`yolov8n.pt` loads and predicts with it); `MODEL` only ever a trusted local file |

An image scanner will flag `ultralytics/utils/events.py:69` in the edge image: it embeds the vendor's own analytics
measurement ID and API secret. It is not a secret of this repository.

## Secrets

A scan on 2026-10-05 of every tracked file and of every line added or removed in every commit up to `8c36699`
(all refs and the reflog; 39 regular expressions; no dedicated scanner was installed) found no real secret: the only
hits are the lab defaults above and false positives. `.gitignore:11` ignores `.env`, but not `.env.local` or other
`.env.*` names.

## Dependency pinning

Pinned: `ultralytics==8.4.170` (`services/edge/requirements.txt:1`) and `torch==2.14.1`
(`services/edge/Dockerfile:7`) in the edge image, the pair the case studies' figures were measured on; the toxiproxy and
grafana images, by version tag, not digest (`docker-compose.yml:36`, `:83`). No pin carries a hash.

These pins are for reproducibility; no pinned or floating version here was checked against published advisories.
`grafana/grafana-oss:11.2.0` is affected by CVE-2025-4123, an XSS rated CVSS 7.6, which the advisory says is fixed on
the 11.2 line in 11.2.9+security-01 and works without a login when anonymous access is on, as it is here
(`docker-compose.yml:88`); the advisory notes that Grafana's default Content-Security-Policy blocks it, and this compose
file sets none (https://grafana.com/security/security-advisories/cve-2025-4123/). Move to a patched release before port
3000 is reachable by anyone else.

Not pinned: `torchvision` (`services/edge/Dockerfile:7`); the ranges in `services/edge/requirements.txt:2-6`,
`services/ingest/requirements.txt:1-2` and `services/sim/Dockerfile:3`; the harness extras in
`harness/pyproject.toml:9-12`, including `ultralytics>=8.3` in `[video]` (which the README's own-clip steps install) and
`[dev]` (which CI installs, `.github/workflows/ci.yml:10`); every transitive Python dependency (no lock file, no
`--require-hashes`); the apt packages `libgl1`, `libglib2.0-0` and `ffmpeg` (`services/edge/Dockerfile:2`); the base
`python:3.11-slim` (`services/*/Dockerfile:1`); the mediamtx and camera images at `:latest` (`docker-compose.yml:8`,
`:13`); emqx `5.8` and postgres `16-alpine`, which float within those lines (`:24`, `:44`); the model weights (no
checksum; see Lab-only defaults). CI pins its actions to major tags, not commit SHAs, and sets no `permissions:` block
(`.github/workflows/ci.yml:7-8`).
