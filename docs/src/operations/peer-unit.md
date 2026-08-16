# Signals peer unit (atelier.service)

Atelier joins the Signals lattice as peer id `atelier`. Process lifecycle is a
systemd oneshot under `signals.target`; the wire contract is
`zndx.engine.v1.Engine` on **:50251** (capability engine).

This is **not** the product gRPC servicer (**:50071** on co-tenant devenv),
gateway, or `just up` product stack. Lattice accept is Engine/Status at gRPC
bind, not product readiness and not vLLM cold-load.

| Fact | Value |
|------|--------|
| Peer id | `atelier` |
| Unit | `atelier.service` (`After=signals-ready.service`) |
| Wrappers | `scripts/systemd_start.sh` / `scripts/systemd_stop.sh` |
| Status probe | `scripts/zndx_status_ok.py` (codegen stubs) |
| gRPC lattice | `:50251` — `zndx.engine.v1.Engine` (+ native `AtelierEngine`) |
| Product servicer | `:50071` (devenv) — **not** lattice accept |
| Postgres | `:5533` — never Signals `:5455` / RustFS `:9010` |
| Capability | `referee` (and `instruct` when configured) |
| Status.project | `atelier` |

## Two gRPC ports (do not collapse)

| Port | Process | Role |
|------|---------|------|
| **:50251** | `python -m atelier.engine.server` | Lattice / `atelier.service` accept |
| **:50071** | `atelier.server` (devenv product servicer) | Workbench UX only |

`just up` / `devenv up` starts `:50071` (+ gateway / vite, and on a
laptop the llama.cpp classify backend on `:8080`). The unit never
starts that product stack, and `devenv up` never starts this engine
(dual-bind on `:50251` is a Gaius lesson).

Product `:50071` may be down while `just lattice-ci --require atelier` is green.
Do not wait for vLLM SERVING — Status at bind is enough.

Laptop cold-start (`just sdg-sample`, llama.cpp) is a **different**
runtime — it must keep working without this unit.

## Wrappers

`systemd_start.sh` is idempotent: if `Engine/Status` already succeeds it exits 0
without tearing down a live engine. Otherwise it starts
`python -m atelier.engine.server` in its own session and polls until Status
answers `project=atelier`.

It does **not** run `just up` (product servicer/gateway/vite).

`systemd_stop.sh` TERMs this checkout's `atelier.engine.server` on `:50251`.
It does **not**:

- stop the product servicer (`:50071`) / gateway / vite
- run `just teardown` or GPU-deep-cleanup
- write, delete, or scan `/tmp/zndx-gpu-leases` (Gaius `:50051` / Ægir `:50151` leases stay)

```bash
# Same as the unit (capability engine only)
./scripts/systemd_start.sh
grpcurl -plaintext 127.0.0.1:50251 list
grpcurl -plaintext 127.0.0.1:50251 zndx.engine.v1.Engine/Status
./scripts/systemd_stop.sh

# Optional product UX — not the accept gate
just up
# product servicer / gateway as documented in Atelier README
```

## Operator (after wrappers land)

From the Signals tree:

```bash
just install-systemd --peers atelier --enable
sudo systemctl start signals.target    # not bare "signals"
grpcurl -plaintext 127.0.0.1:50251 list
grpcurl -plaintext 127.0.0.1:50251 zndx.engine.v1.Engine/Status
just lattice-ci --require atelier
```

## Platform Metaflow / events (when federated)

```bash
export METAFLOW_SERVICE_URL=http://127.0.0.1:30180
# profile: signals config/metaflow/platform.json
```

## Health

Lattice-ci probes Status + reflection only. Product health stays Atelier-local.
GPU leases: `/tmp/zndx-gpu-leases` — keep lease-aware so stop does not wipe
Gaius / Ægir.
