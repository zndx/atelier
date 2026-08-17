# Signals peer unit (atelier.service)

Atelier joins the Signals lattice as peer id `atelier`. systemd is only
the `signals.target` membership hook. The wire contract is
`zndx.engine.v1.Engine` on **:50251**.

**Gaius-style wrap:** `ExecStart` is `devenv up -d`. The lattice engine
is the devenv process `capability-engine`, so `devenv processes
status|restart|stop capability-engine` works. The same graph starts on
Linux and macOS — devenv already gates linux-vs-darwin (CUDA libs, d2);
there is no laptop-vs-server switch.

Lattice accept is Engine/Status at gRPC bind, not vLLM cold-load and
not product HTTP. Status also advertises `surfaces` (`kind=primary`,
devenv Vite `:3300`, never Metaflow `:3000`) and answers `ServerQuery` (REMOTES / PEERS /
SURFACES) for the chrome waffle.

| Fact | Value |
|------|--------|
| Peer id | `atelier` |
| Unit | `atelier.service` (`After=signals-ready.service`) |
| Wrappers | `scripts/systemd_{start,stop,unit}.sh` |
| Engine process | devenv `capability-engine` → `scripts/processes/capability-engine.sh` |
| Status probe | `scripts/zndx_status_ok.py` (codegen stubs) |
| gRPC lattice | `:50251` — `zndx.engine.v1.Engine` (+ native `AtelierEngine`) |
| Product servicer | `:50071` (same devenv graph; not the accept gate) |
| Postgres | `:5533` — never Signals `:5455` / RustFS `:9010` |
| Capability | `referee` (and `instruct` when configured) |
| Status.project | `atelier` |

## Two gRPC ports (same graph, different accept)

| Port | Process | Role |
|------|---------|------|
| **:50251** | `capability-engine` | Lattice accept |
| **:50071** | `grpc-server` | Workbench product gRPC |

Do not wait for vLLM SERVING — Status at bind is enough.

## Wrappers

`systemd_start.sh` is idempotent: if a **process-compose-owned** listener
already answers `Engine/Status`, it exits 0. Foreign/old `setsid` engines
are reaped, then `devenv up -d` (never foreground `just up`).

`systemd_stop.sh` runs `just down` / `devenv processes down` for this
checkout and reaps leftover `atelier.engine` PIDs. It does **not**:

- run `just teardown` or GPU-deep-cleanup
- write, delete, or scan `/tmp/zndx-gpu-leases` (Gaius `:50051` / Ægir `:50151` leases stay)

```bash
# Same graph as the unit
devenv up -d
devenv processes status          # includes capability-engine
grpcurl -plaintext 127.0.0.1:50251 list
grpcurl -plaintext 127.0.0.1:50251 zndx.engine.v1.Engine/Status
devenv processes restart capability-engine
just down
```

## Operator (from Signals)

```bash
just install-systemd --peers atelier --enable
sudo systemctl start signals.target    # not bare "signals"
grpcurl -plaintext 127.0.0.1:50251 list
grpcurl -plaintext 127.0.0.1:50251 zndx.engine.v1.Engine/Status
just lattice-ci --require atelier
```

## Platform Metaflow

```bash
export METAFLOW_SERVICE_URL=http://127.0.0.1:30180
```

## Health

Lattice-ci probes Status + reflection only. Product health stays
Atelier-local. GPU leases: `/tmp/zndx-gpu-leases` — stop must stay
lease-aware so it does not wipe Gaius / Ægir.
