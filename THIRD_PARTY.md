# Third-party components

What this repository pulls in, under which licence, and where each licence was read. The Apache-2.0 licence on this
repository's code (LICENSE, NOTICE) does not change any third-party licence. This is an inventory, not legal advice.

Unless the Source column says otherwise, a fact was read on 2026-10-05 from installed package metadata (`pip show`,
`METADATA`, the licence files in the `.dist-info`) inside the built images, from image labels (`docker image inspect`),
or from the host Python that `make test` uses. The edge image was an arm64 build; an amd64 build may bundle slightly
different native libraries. `pip` 24.0 does not print `License-Expression`, so where `pip show` is empty the licence
comes from that `METADATA` field. Copyleft licences are in **bold**.

## Edge image (`services/edge/Dockerfile`)

| Component | Version | Licence | Source | Shipping it commercially |
|---|---|---|---|---|
| `ultralytics` | 8.4.170, pinned (`services/edge/requirements.txt:1`) | **AGPL-3.0**, or the vendor's paid Enterprise licence | `pip show` (`AGPL-3.0`); classifier says "v3 or later", the `licenses/LICENSE` text is plain GNU AGPL v3; `METADATA:163`, `:406-411` state the dual licence and that the Enterprise licence covers software and AI models | Imported by the edge service in one process with our code. Distributing the image conveys `ultralytics` itself under AGPL-3.0 (s.4-6). Whether the combined program is a work based on it (s.0), and so under AGPL-3.0 as a whole, is a legal reading, not licence text. The network clause (s.13) covers a modified version that supports remote interaction; the edge program only publishes MQTT. The vendor offers its Enterprise licence for commercial use, including internal tools and production deployments (`METADATA:163`, `:411`). Take advice |
| `yolov8n.pt` weights | `/app/yolov8n.pt`, 6549796 bytes, fetched at build time (`services/edge/Dockerfile:10`), no checksum | **AGPL-3.0** | the checkpoint's own metadata (`yolov8n/data.pkl`, read without unpickling): `'license': 'AGPL-3.0 License (https://ultralytics.com/license)'`. The package writes the same string into every checkpoint it saves or exports. The licence file of the assets repository was not read | Treat as AGPL-3.0 unless you hold the Enterprise licence. Never committed (`.gitignore:10`) |
| `ultralytics-platform` | 0.1.80 (required by `ultralytics`) | **AGPL-3.0-only** | `METADATA` License-Expression; `licenses/LICENSE` is GNU AGPL v3 | As `ultralytics` |
| `ultralytics-thop` | 2.2.2 (required by `ultralytics`) | **AGPL-3.0** | `pip show`; `licenses/LICENSE` is GNU AGPL v3 | As `ultralytics` |
| `torch` | 2.14.1+cpu, pinned (`services/edge/Dockerfile:7`) | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | `METADATA` License-Expression; `licenses/LICENSE` and about 90 `third_party/*/LICENSE` files | Permissive; keep the notices. BSL-1.0 is the Boost Software License (`third_party/sleef`), not the Business Source License. No GPL, LGPL, AGPL, MPL or EPL text in its third-party licence files, but `torch/lib` bundles `libgfortran`, `libgomp` and OpenBLAS, which none of them names (see the bundled-libraries row) |
| `torchvision` | 0.29.1+cpu, unpinned | BSD-3-Clause | `pip show` (`BSD`); dist-info `LICENSE` | Permissive; keep the notice |
| `opencv-python-headless` | 5.0.0.93, unpinned (`services/edge/requirements.txt:3`) | Apache-2.0 (OpenCV), MIT (wheel packaging), bundled FFmpeg under **LGPL-2.1** | `pip show` (`Apache 2.0`); dist-info `LICENSE.txt` (MIT); `LICENSE-3RD-PARTY.txt:1-2`, `:243-252` (FFmpeg), `:208` (libvpx, BSD), `:924` (OpenSSL licence) | The FFmpeg shared libraries in `opencv_python_headless.libs` come under LGPL-2.1 terms when the image is distributed. In this image they ship but are not loaded: `cv2/cv2.abi3.so` is `opencv-python`'s binary (next row) |
| `opencv-python` | 5.0.0.93 (required by `ultralytics`) | as above, plus bundled Qt 5 under **LGPL-3.0** | `pip show` (`Apache 2.0`); `LICENSE-3RD-PARTY.txt:710-718` | As above, plus LGPL-3.0 for the Qt 5 libraries. Both wheels install into one `cv2/` directory. In this image `cv2/cv2.abi3.so` matches this wheel's `RECORD` sha256, not the headless one's, and `ldd` resolves Qt 5 (5.15.19) and the FFmpeg libraries from `opencv_python.libs`, so they load on every `import cv2`. The edge program does not import `cv2` itself; `ultralytics` does, unmodified, in the same process. A rebuild can change which wheel's binary ends up on disk |
| `certifi` | 2026.7.22 (via `requests`, `httpx`) | **MPL-2.0** | `pip show`; classifier; `licenses/LICENSE` | File-level copyleft. When the image is distributed, certifi's files stay under MPL-2.0: keep its licence notice (dist-info `licenses/LICENSE`, the only one; `cacert.pem` has none) and tell recipients (s.3.1, s.3.4). Changes to those files must be shared under MPL-2.0. Used unmodified, not imported by our code |
| `paho-mqtt` | 2.1.0 | **EPL-2.0** OR BSD-3-Clause | `pip show`; `licenses/LICENSE.txt` (EPL 2.0 and EDL 1.0) | Dual-licensed: the BSD-3-Clause (EDL 1.0) branch can be chosen. Also in ingest and sim |
| `python-ulid` | 4.0.1 | MIT | `METADATA` License-Expression; `LICENSE` | Permissive. Also in sim |
| `lap` | 0.5.13 | BSD-2-Clause | `pip show`; `LICENSE` | Permissive |
| `typing_extensions` | 4.16.0 | PSF-2.0 | `METADATA` License-Expression; `LICENSE` | Permissive. Also in ingest and sim |
| `numpy` | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | `METADATA` License-Expression; 17 licence files | Permissive; its bundled `libgfortran` and OpenBLAS are in the next row |
| Native libraries bundled inside the wheels | `libgfortran` (`numpy.libs`, `torch/lib`, both `opencv_python*.libs`); `libgomp` (`torch/lib`); OpenBLAS (`numpy.libs`, `torch/lib`, both `opencv_python*.libs`); `ld-linux-aarch64` (`torchvision.libs`) | **GPL-3.0-or-later WITH GCC-exception-3.1** (libgfortran, libgomp); BSD-3-Clause (OpenBLAS); **LGPL-2.1-or-later** (glibc loader) | numpy's `licenses/LICENSE.txt:40-44` (OpenBLAS) and `:131-135` (GCC runtime) cover numpy's copies. The torch, torchvision and both OpenCV wheels ship theirs with no licence file naming them (their dist-info licence files and `LICENSE-3RD-PARTY.txt` searched). For libgomp and the loader, the licences are as Debian states them for its own copies in the same image: `/usr/share/doc/libgomp1/copyright:98-109` (GCC runtime libraries, libgomp among them) and `/usr/share/doc/libc6/copyright:7-9` (glibc, `LGPL-2.1+`) | The GCC Runtime Library Exception lets GCC-compiled code use these without taking on GPL terms; keep the GPL-3.0 and exception texts with the image. LGPL-2.1 terms for the loader when the image is distributed. The torch and OpenCV copies of OpenBLAS ship without the BSD-3-Clause notice, which BSD-3-Clause asks binary redistributions to reproduce |
| `matplotlib` | 3.11.2 | Matplotlib License (PSF-based) | classifier; dist-info `LICENSE` | Permissive |
| `pillow` | 12.3.0 | MIT-CMU | `METADATA` License-Expression | Permissive |
| other transitive packages | polars 1.44.2 and polars-runtime-32 MIT; psutil 7.2.2 BSD-3-Clause; PyYAML 6.0.3 MIT; requests 2.34.2 Apache-2.0; cloudpickle 3.1.2, fsspec 2026.7.0, Jinja2 3.1.6, MarkupSafe 3.0.3, networkx 3.6.1, contourpy 1.3.3, kiwisolver 1.5.1, httpx 0.28.1, httpcore 1.0.9, idna 3.20 BSD-3-Clause; sympy 1.14.0, mpmath 1.3.0, cycler 0.12.1 BSD; filelock 3.32.3, fonttools 4.66.1, pyparsing 3.3.3, six 1.17.0, h11 0.16.0, anyio 4.15.1, charset-normalizer 3.5.2, urllib3 2.8.0 MIT; python-dateutil 2.9.0.post0 Apache-2.0 AND BSD-3-Clause (both apply: BSD-3-Clause to all code, Apache-2.0 to later and re-licensed contributions); nvidia-ml-py 13.615.71 BSD | permissive | `METADATA` and dist-info licence files; nvidia-ml-py from its `pip` field and classifier only (no licence file) | Permissive; keep the notices, including `requests`' NOTICE |
| `pip`, `setuptools`, `wheel`, `packaging` | 24.0, 79.0.1, 0.46.3, 26.3 | MIT, MIT, MIT, Apache-2.0 OR BSD-2-Clause | dist-info licence files and `METADATA`, in all three images | Permissive |
| CPython | 3.11.17 (`python:3.11-slim`, Debian 13) | PSF-2.0 | `/usr/local/lib/python3.11/LICENSE.txt` | Permissive. All three images |
| Debian `ffmpeg` | 7:7.1.5-0+deb13u1 (`services/edge/Dockerfile:2`) | **GPL-2.0-or-later** as built | `/usr/share/doc/ffmpeg/copyright`; `ffmpeg -buildconf` shows `--enable-gpl`, `--enable-libx264` | GPL terms for these binaries when the image is distributed. Our code never runs an `ffmpeg` binary (`cv2` uses its own); whether `ultralytics` does on our paths was not checked |
| Debian GPL libraries that `ffmpeg` links (`ldd /usr/bin/ffmpeg`), among them | `libx264-164` 2:0.164.3108+git31e19f9-2+b1, `libx265-215` 4.1-2, `libxvidcore4` 2:1.3.7-1+b2, `libvidstab1.1` 1.1.0-2+b2, `librubberband2` 3.3.0+dfsg-2+b3, `libpostproc58` 7:7.1.5-0+deb13u1, `libfftw3-double3` 3.3.10-2+b1, `libzvbi0t64` 0.2.44-1, `libdvdnav4` 6.1.1-3+b1, `libdvdread8t64` 6.1.3-2, `libslang2` 2.3.3-5+b2; `libcdio19t64` 2.2.0-4.1~deb13u1, `libcdio-paranoia2t64` and `libcdio-cdda2t64` 10.2+2.0.2-1+b1 (installed with `ffmpeg`) | **GPL-2.0-or-later**; **GPL-3** (`libcdio19t64`); **GPL-3.0-or-later** (`libcdio-paranoia2t64`, `libcdio-cdda2t64`) | `/usr/share/doc/<pkg>/copyright`, `Files: *` stanza (`libpostproc58`: the stanza listing `libpostproc/*`, `:490-634`) | As `ffmpeg`. The GPL-3 libraries load into the same `ffmpeg` process, so GPL-3.0 terms, not only GPL-2.0, apply to that binary when the image is distributed. Debian's "GPL v2+" sentence for `ffmpeg` does not cover them |
| Debian `libgl1`, `libglib2.0-0t64` | 1.7.0-1+b2, trixie | mostly MIT/Apache (libglvnd); **LGPL-2.1+** (glib) | `/usr/share/doc/<pkg>/copyright` | Keep the notices; LGPL terms for glib when distributed |
| other Debian packages | 289 in the edge image | each its own | `/usr/share/doc/*/copyright` (not listed here) | Ship their copyright files with the image |

The `ultralytics` package also ships sample images (`ultralytics/assets/bus.jpg`, `zidane.jpg`) with no separate
licence statement.

## Ingest and sim images (`services/ingest/Dockerfile`, `services/sim/Dockerfile`)

Both build from `python:3.11-slim` (Debian 13; CPython, `pip` and friends as above; 87 Debian packages in ingest).
`python:3.11-slim` was not present locally, so the base image itself was not inspected; its packaging is MIT per
https://github.com/docker-library/python (LICENSE).

| Component | Version | Licence | Source | Shipping it commercially |
|---|---|---|---|---|
| `psycopg` (ingest) | 3.3.6, from `psycopg[binary]>=3.2` (`services/ingest/requirements.txt:2`) | **LGPL-3.0-only** | `METADATA` License-Expression; `licenses/LICENSE.txt` is GNU LGPL v3 | Imported unmodified by `services/ingest/src/main.py:9`, loaded dynamically. LGPL-3.0 terms apply to the library when the image is distributed |
| `psycopg-binary` (ingest) | 3.3.6 | **LGPL-3.0-only**, plus bundled native libraries | `METADATA` License-Expression; `licenses/LICENSE.txt`; `psycopg_binary.libs` holds libpq 5.18, OpenSSL 3 and 1.1.1k, krb5, OpenLDAP, SASL, libcrypt, keyutils, libselinux, pcre2, com_err | The dist-info has no licence files for the bundled libraries, and their licences were not read |
| `paho-mqtt` (ingest, sim) | 2.1.0 | **EPL-2.0** OR BSD-3-Clause | as in the edge table | As in the edge table |
| `python-ulid`, `typing_extensions` (sim) | 4.0.1, 4.16.0 | MIT, PSF-2.0 | as in the edge table | Permissive. Sim installs unpinned ranges (`services/sim/Dockerfile:3`) |

## Harness optional extras (`harness/pyproject.toml`)

The `replay` package itself declares no dependencies. The extras are opt-in.

| Component | Version | Licence | Source | Shipping it commercially |
|---|---|---|---|---|
| `pytest` (`dev`) | 8.4.2 on the host | MIT | host `pip show` | Test-only |
| `numpy` (`dev`, `viz`, `trackeval`) | 2.2.6 on the host | as in the edge table | edge image `METADATA` (host `pip show` prints only the copyright line) | Permissive |
| `ultralytics` (`video`) | `>=8.3`; not installed on the host | **AGPL-3.0** | as in the edge table | As in the edge table: installing `[video]` brings in AGPL code |
| `opencv-python-headless` (`video`, `viz`) | 5.0.0.93 on the host | as in the edge table | host `pip show` (`Apache 2.0`); the macOS wheel also bundles libbluray, gnutls, nettle, lame and others under **LGPL** (`LICENSE-3RD-PARTY.txt:245-246`) | As in the edge table |
| `matplotlib` (`viz`) | not on the host | Matplotlib License | edge image (3.11.2) | Permissive |
| `imageio` (`viz`) | not installed | BSD-2-Clause | https://github.com/imageio/imageio (LICENSE), the project's repository | Permissive |
| `pillow` (`viz`) | 10.4.0 on the host | HPND | host `pip show` | Permissive |
| `scipy` (`trackeval`) | 1.17.1 on the host | BSD-3-Clause; the macOS wheel bundles libgfortran and libgcc_s (**GPL-3.0-or-later WITH GCC-exception-3.1**) and libquadmath (**LGPL-2.1-or-later**) | host `pip show` (the License field holds the copyright line); `scipy-1.17.1.dist-info/LICENSE.txt:129-133`, `:914-918` | Permissive core; the GCC runtime exception and LGPL-2.1 apply to the bundled libraries if you redistribute them |
| TrackEval (git clone, not a package) | commit `12c8791b303e0a0b50f753af204249e622d0281a`, in gitignored `.cache/` | MIT | `.cache/TrackEval/LICENSE` ("Copyright (c) 2020 Jonathon Luiten") | Used by `harness/scripts/trackeval_run.py`; nothing vendored into this tree |
| `boxmot` (no extra; optional) | not installed anywhere | **AGPL-3.0** | https://github.com/mikel-brostrom/boxmot (LICENSE), the project's repository | Only if you install it for `harness/replay/trackers/boxmot_adapter.py`: then AGPL-3.0 terms |

## Compose service images (`docker-compose.yml`)

Each runs as its own container, unmodified. None is linked into our code.

| Service | Image | Licence | Source | Shipping it commercially |
|---|---|---|---|---|
| mediamtx | `bluenviron/mediamtx:latest` (`:8`, unpinned) | MIT | https://github.com/bluenviron/mediamtx (LICENSE), the project's repository; image labels empty | Permissive |
| camera | `linuxserver/ffmpeg:latest` (`:13`, unpinned; 9.0-cli-ls83 locally) | **GPL-3.0-only** | image label `org.opencontainers.image.licenses` | GPL-3.0 terms if you distribute the image. Whether its FFmpeg build enables nonfree codecs was not verified |
| emqx | `emqx/emqx:5.8` (`:24`; 5.8.6 locally) | Apache-2.0 | image labels `licenses=Apache-2.0`, `edition=Opensource` | Permissive. Per the project's announcements (not read from an image), 5.9.0 and later ship under the **Business Source License 1.1**: an upgrade past 5.8 changes the licence |
| toxiproxy | `ghcr.io/shopify/toxiproxy:2.9.0` (`:36`) | MIT | image label | Permissive |
| postgres | `postgres:16-alpine` (`:44`) | PostgreSQL License (server); MIT (image packaging); Alpine base packages under their own licences, for example busybox **GPL-2.0-only**, bash and readline **GPL-3.0-or-later** | https://www.postgresql.org/about/licence/ and https://github.com/docker-library/postgres (LICENSE), the projects' repositories, not read from the image; image labels empty; Alpine package licences from the image's `/lib/apk/db/installed` (`L:` field) | Permissive server; the base packages' licences travel with the image |
| grafana | `grafana/grafana-oss:11.2.0` (`:83`) | **AGPL-3.0-only** | https://github.com/grafana/grafana (LICENSE; relicensed from Apache-2.0 at v8.0), the project's repository | We supply only provisioning YAML and dashboard JSON and make no API calls. AGPL-3.0 terms if you modify it and offer it over a network, or distribute the image |

## Datasets and footage

Every clip, with URL, author and licence, is logged in [media/SOURCES.md](media/SOURCES.md). No footage is committed.

| Source | Licence | Where it shows up | Shipping it commercially |
|---|---|---|---|
| MTID (`media/SOURCES.md:8`) | CC BY 4.0 | `docs/footage/real-compare.gif` and `docs/footage/real-after.gif` are rendered from it | CC BY 4.0 requires attribution (authors, licence, link, changes noted) wherever the GIFs go; this repository also adds a citation (`media/SOURCES.md:8`, "Shown publicly?"). The GIFs are also under the docs' CC BY-NC-SA 4.0 |
| Vecteezy 6434705 (`:9`) | Vecteezy Free License, attribution required | metrics only | Credit required |
| UA-DETRAC (`:11-12`), MOT17 (`:13`) | CC BY-NC-SA 3.0 (UA-DETRAC: academic use only) | metrics and citations only, no frames | Non-commercial: not for commercial use |

## What is encumbered in this repository

Our files that import AGPL code:

- `services/edge/src/main.py:15` imports `ultralytics` at module level, in the same process as `replay/`, which
  `services/edge/Dockerfile:11` copies into the image. Its default model is `yolov8n.pt` (`:70`).
- `harness/replay/trackers/bytetrack.py:25-28` imports `ultralytics` lazily, inside `make()`. The edge reaches it
  through `replay.trackers.create` when `CONTAIN_SHARE` is above 0. `replay.phantoms` and `replay.secondbox` (whose
  baseline is ByteTrack, so they run in the edge image), `replay.compare` and `replay.track` reach it by tracker name
  at runtime, never at import.
- `harness/scripts/dump_detections.py:23` and `harness/scripts/dump_tracks.py:14` import `ultralytics` at module level.
- `harness/replay/trackers/boxmot_adapter.py:22` imports `boxmot` (AGPL-3.0) lazily, inside `make()`. It is not
  installed anywhere, so the `boxmot_*` trackers fail at `create()`.

Distributing the edge service, or the optional ByteTrack adapter together with `ultralytics`, conveys AGPL-3.0 code.
Whether that combined program is a work based on `ultralytics`, and so under AGPL-3.0 as a whole, is a legal reading,
not licence text. The network clause (s.13) covers a modified version that supports remote interaction; the edge only
publishes MQTT. The vendor offers an `ultralytics` Enterprise licence for commercial use (`METADATA:163`, `:411`);
take advice before shipping. The replay harness core (`harness/replay/` without `trackers/bytetrack.py` and
`trackers/boxmot_adapter.py`), the other scripts, and the sim and ingest services import no AGPL code; in the core, only
choosing the `bytetrack` tracker loads it. `replay/trackers/__init__.py` imports both adapter modules to register them,
but neither loads `ultralytics` or `boxmot` until `make()` runs, and the tests stub `ultralytics`, so CI never installs
it. `torch` is never imported by our code; `services/edge/Dockerfile:7` installs it (CPU wheel, pinned) because
`ultralytics` needs it. Grafana runs as a separate service under its own licence and is not linked into our code.

## LICENSE and NOTICE in built artifacts

The builds here do not copy this repository's `LICENSE` or `NOTICE` into what they produce: the edge and sim images copy
only `harness/replay` and their `src/` (`services/edge/Dockerfile:11-12`, `services/sim/Dockerfile:4-5`), the ingest
image only its `src/` (`services/ingest/Dockerfile:5`), and `harness/` holds neither file, so a wheel built from it
carries no licence text. Apache-2.0 s.4(a) and s.4(d) require both with any redistribution, so anyone distributing a
built image or wheel must include `LICENSE` and `NOTICE` themselves.
