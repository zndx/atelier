# MiNiFi engine sidecar — telemetry + C2 design (signals umbrella)

**Date:** 2026-07-03
**Direction (RH):** each engine (Atelier, Ægir) runs its own MiNiFi C++
instance providing a non-OTel telemetry channel + remote C&C; one NiFi
operator on the local RKE2 cluster manages the engine-bearing projects under
the signals umbrella. Fork: `weathership/oss-minifi-cpp` at
`external/nifi-minifi-cpp` (tracks upstream main, no custom commits yet —
pointer-control fork, same pattern as oss-embedding-atlas).

## Ground truth (verified)

- **Gaius's Metaflow→NiFi prototype used NO MiNiFi and NO C2.** It was full
  JVM NiFi as a local devenv process (`:8450`, HTTP-only, single-user, S2S
  disabled), with three channels: OTel traces → collector → `ListenOTLP`
  (:4319, fragile — silently degrades to a placeholder when the OTLP NAR is
  absent, and there's a gRPC-vs-HTTP transport mismatch in its config);
  NiFi→engine triggers via `ExecuteStreamCommand` invoking the gaius CLI;
  Python httpx `NiFiClient` for REST canvas control. "Heartbeats" there are
  OTel span events with z-score anomaly detection, not C2.
- **RKE2 is live**: single node `tinybox` v1.34.3+rke2r3 (140d), `zarf`
  namespace present (Zarf packaging is the cluster's established deploy
  pattern; Ægir has a `zarf/` tree). No NiFi anywhere on the cluster yet.
- **MiNiFi C++ C2 is first-class** in the fork (`C2.md`): heartbeats
  (DeviceInfo/AgentInformation/FlowInformation/ConfigurationChecksums),
  flow updates, acknowledged operations, agent classes.

## Architecture

```
tinybox (host)                          RKE2 (same box)
┌──────────────────────────────┐        ┌──────────────────────────┐
│ atelier-engine :50251        │        │ nifi ns (Zarf package)   │
│   └ vLLM children :8200+     │        │  NiFi (operator-managed) │
│   └ events.jsonl (value-free)│        │  + C2 endpoint           │
│ MiNiFi agent                 │──S2S──▶│  RouteOnAttribute→sinks  │
│   class=atelier-engine       │◀──C2──│  ops / flow updates      │
│   (tails events.jsonl;       │        └──────────────────────────┘
│    bridges ops→127.0.0.1     │                 ▲
│    native gRPC admin)        │                 │ same for
├──────────────────────────────┤                 │ class=aegir-engine
│ aegir-engine :50151 (mirror) │─────────────────┘
└──────────────────────────────┘
```

Key decisions:

1. **MiNiFi is a supervisor-level sibling, NOT an engine child.** The C&C
   channel must survive engine crashes to be useful ("restart the engine"
   needs a live agent precisely when the engine is dead). The agent watches
   the engine from outside: tails its event stream, heartbeats
   independently, and can exec remediation.
2. **Telemetry substrate = the engine event stream** (LANDED, increment 1):
   `atelier/engine/events.py` — append-only JSONL in `engine.log_dir`,
   **value-free by construction** (key allowlist enforced at emit; content
   -shaped keys refused loudly — the egress-membrane invariant mechanized).
   Events: engine_start, endpoint_launch/healthy/fatal/stop, complete
   (capability, model, token counts, latency, finish_reason,
   schema_constrained, reasoning_retained — never text). Correlation:
   per-process `run_id` + monotonic `seq` (the Gaius correlation-id
   pattern).
3. **C&C lands via the native service on loopback only.** C2 delivers an
   operation → the MiNiFi flow bridges it to `127.0.0.1:50251`
   (`atelier.engine.AtelierEngine` admin surface / a small
   `atelier.engine.ctl` CLI). The engine is never exposed off-box; the
   MiNiFi agent is the only outward-facing component, mTLS'd to the
   operator. Agent classes `atelier-engine` / `aegir-engine` under one C2
   server.
4. **NiFi deploy = Zarf package into RKE2** (`nifi` namespace), operator-
   managed per RH's direction (nifikop-class operator; single-node
   NiFi + C2). Host↔cluster reachability via NodePort/hostPort (the
   cluster node IS the engines' host — same pattern Gaius used for
   Metaflow's Postgres/MinIO endpoints, in reverse).
5. **Security posture inverts Gaius's dev stance**: mTLS agent↔C2,
   per-agent identity, `ConfigurationChecksums` in heartbeats (tamper
   evidence), signed flow updates. Gaius's HTTP-only/S2S-off/no-TLS was a
   devbox stance; a C&C channel doesn't get that luxury.

## Reused from Gaius (proven)

- RouteOnAttribute taxonomy for classifying agent telemetry (theirs:
  step/event/phase/anomaly → ours: event/capability/severity/project).
- Statistical heartbeat payload semantics (sequence, elapsed, baseline
  mean/stddev, z-score, anomaly flag) — as C2 heartbeat enrichment.
- Semantic desired-vs-actual flow diff (their `FlowState`/`SemanticDiff` +
  RASE NiFi state model) — the reconciliation shape C2 flow-updates need.
- Guru-code + `/health fix` remediation convention (`#NF.*`-style codes).
- kubeconfig-sync pattern (root rke2.yaml → user copy) for any tooling that
  talks to the cluster.

## Build increments

1. **[LANDED] Engine event stream** — `events.py`, hooks at all lifecycle
   points, allowlist tests (13 engine tests green).
2. **MiNiFi binary** — build from the fork (cmake/conan; heavy) or start
   from an upstream release binary while the fork carries no changes;
   minimal local flow first: TailFile(events.jsonl) → LogAttribute, no
   network. Proves agent lifecycle on the host.
3. **NiFi + C2 on RKE2** — Zarf package, `nifi` namespace, operator, mTLS
   bootstrap, NodePort ingress for S2S + C2.
4. **Wire the channel** — agent flow: TailFile → S2S to NiFi; C2 heartbeat
   on (`nifi.c2.enable=true`, `agent.class=atelier-engine`, 30s period);
   NiFi routing flow per the taxonomy.
5. **C&C bridge** — C2 operations → loopback engine ctl; start with
   Status/EnsureEndpoint, add Drain/Stop RPCs as needed.
6. **Ægir mirror** — via the running-observations channel; same package,
   `agent.class=aegir-engine`.

## Open questions (for RH / the Ægir session)

- Operator choice: nifikop (konpyutaika) vs plain StatefulSet chart under
  Zarf — single-node cluster makes the operator's value mostly upgrade
  management; RH said "operator", so nifikop unless it fights RKE2.
- Where MiNiFi supervision lives: devenv process (dev) + systemd unit
  (always-on)? An engine-adjacent supervisor entry is the natural devenv
  shape; always-on C&C wants systemd.
- Does the fork intend local modifications (why a fork?) — e.g. a custom
  processor for the engine bridge, or provenance-stamped telemetry?
