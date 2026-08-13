# peer-unit@atelier lock-in

Locked in the already-green Signals lattice unit. Did not re-architect
the engine; did not start product `just up`.

## Accept

- `atelier.service` active (oneshot, engine-only `:50251`)
- `grpcurl -plaintext 127.0.0.1:50251 list` / `Engine/Status`
- `cd /home/rch/local/src/wxs/signals && just lattice-ci --require atelier`

## Tree

- `scripts/systemd_{start,stop}.sh`, `scripts/zndx_status_ok.py`
- `src/atelier/engine/server.py` — reflection + Status placeholders at bind
- `src/atelier/engine/config.py` — `ATELIER_ENGINE_PORT` override
- `pyproject.toml` / `uv.lock` — `grpcio-reflection`
- Product SoR: `docs/current/src/operations/peer-unit.md`
- Book: `docs/src/operations/peer-unit.md` + SUMMARY link
- Tests: `tests/engine/test_zndx_status.py` (Status + reflection + soft-stop)

## Dual-port

| Port | Process | Role |
|------|---------|------|
| `:50251` | `python -m atelier.engine.server` | lattice accept |
| `:50071` | `atelier.server` | product UX; not accept |

## Soft stop

`systemd_stop.sh` TERMs `atelier.engine` on `:50251` only. Does not
touch `/tmp/zndx-gpu-leases`, `just teardown`, Gaius `:50051`, or Ægir
`:50151`.
