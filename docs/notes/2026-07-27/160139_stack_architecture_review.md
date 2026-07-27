# Stack architecture & connections — review for `devenv up -d` management

**Date:** 2026-07-27
**Context (RH):** the Atelier stack will be managed with `devenv up -d`
(detached process-compose — solves the supervisor-ownership/orphan problem:
no session-lifetime coupling; `devenv processes down` stops it cleanly).

## The tinybox constellation (shared 6×4090 host)

Three projects + bindings:

| | Gaius (Knowledge) | Ægir (Ontology) | Atelier (Classification) |
|---|---|---|---|
| Engine gRPC | **50051** (live) | 50151 (live) | 50251 (live, idle) |
| vLLM range | 8080–8095 | 8100+ | 8200+ |
| UI | TUI | :5173 vite · :8080 caddy · :5006 bokeh · :21000 Atlas | :3000 vite · :8090 gateway |

Bindings: `/tmp/zndx-gpu-leases` (zero-share, `min_mib=64`) ·
`zndx.engine.v1` federation face (Ægir+Atelier; Gaius=OIP, reconciliation
open) · engine port lattice above.

## Atelier managed set (`devenv up -d` owns exactly these)

```
postgres :5533 ──(healthy)──▶ grpc-server :50071 ──(healthy)──▶ gateway :8090
qdrant  :6333/6334 (healthz probe)                              vite-dev :3000
```
- **postgres** — `services.postgres`, PG16+pgvector.
- **qdrant** — process w/ `/healthz` readiness probe.
- **grpc-server** — db-bootstrap inline, then `atelier.server`; readiness =
  TCP connect probe. **Now on :50071** (see finding below).
- **gateway** — uvicorn :8090, REST→gRPC + serves ui/dist in prod;
  depends on grpc-server healthy.
- **vite-dev** — :3000, `/api`+`/ws` proxy → :8090.

## Outside the supervisor (deliberate)

- **Atelier engine** :50251 (+vLLM :8200+ per capability) — operator-run
  (`just engine-serve`); idle-safe by construction (GPU lease claimed only
  at `ensure()`); lifecycle tracks GPU windows, not app restarts — so NOT
  a devenv process for now.
- **PR-1 reference UI** :3001 (worktree vite; PR on hold).

## Connection graph

```
browser ─▶ :3000 vite ─/api,/ws─▶ :8090 gateway ─gRPC─▶ :50071 servicer
                                      │                      ├─▶ :5533 postgres
                                      │ (pipeline in-proc)   └─▶ :6333 qdrant
                                      └─▶ classify.llm → engine vLLM :8200/v1
                                          (window-gated: refuses under Ægir lease)
:50251 engine ◀─ just engine-* / future gateway bridge; ATELIER_ENGINE_FEDERATE seam
```

## FINDING (defused today): silent 50051 collision

Gaius's engine (restarted post-reboot) holds :50051 — the servicer's
config default AND the hardcoded readiness-probe port. Failure mode was a
trap: `add_insecure_port` returns 0 on bind failure and `start()` proceeds
**portless**; the TCP probe would then hit *Gaius's* socket → false
healthy → gateway dials 50051 → wrong project's engine → every RPC
UNIMPLEMENTED while the UI shows "connected".

Fixes (all landed, 155 tests green, port pin verified via devenv shell):
1. `server.py` — **fail-fast on bind failure** (raise, naming the port and
   the fix) per the required-critical-path directive.
2. `devenv.nix` — `env.ATELIER_GRPC_PORT = "50071"` (devenv-only pin;
   CAI keeps 50051 — no Gaius there) + readiness probe moved to 50071.
3. CLAUDE.md port table updated (devenv 50071 / CAI 50051).

Note: the servicer sat on 50071 six days ago with no override on record —
the old session likely had a manual export. It is now pinned declaratively.

## Ops quickref

- Start: `devenv up -d` · stop: `devenv processes down` · attach TUI:
  `devenv up` (foreground) or process-compose attach.
- Engine: `just engine-serve` / `engine-status` / `engine-ping` (will
  correctly refuse capabilities while Ægir holds the GPU window).
- First `devenv up -d` after this note boots: keiretsu UI, context-free
  acceleration probe, and the 50071 pin — all for the first time together.
```
