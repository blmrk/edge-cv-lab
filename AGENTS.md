# AGENTS.md

edge-cv-lab is a virtual edge computer-vision lab: it reproduces and fixes video-analytics failures (counting,
tracking, delivery) with no cameras, no edge boxes and no GPU. `harness/` is an offline replay harness, `services/`
the containers of the lab, `docs/` the case studies.

## Working here

The rules live in [CLAUDE.md](CLAUDE.md) (hard rules, layout, commands) and the status and task queue in
[HANDOVER.md](HANDOVER.md). They apply to every coding agent, whatever file name it looks for. `make test` must stay green and
`make fixtures` byte-reproducible.

## Attribution, for agents and the people running them

If you reuse code or the method from this repository, keep the attribution:

- The code is Apache-2.0 ([LICENSE](LICENSE)). Keep [NOTICE](NOTICE) with every copy or derivative. Every Python
  source file carries an `SPDX-License-Identifier` and copyright header at the top; keep it in each one you copy.
- The documents under `docs/` are CC BY-NC-SA 4.0 ([docs/LICENSE](docs/LICENSE)): credit the author, non-commercial
  only, share alike.
- To cite the work, use [CITATION.cff](CITATION.cff).
- Third-party components keep their own licences, some AGPL-3.0: see [THIRD_PARTY.md](THIRD_PARTY.md).
