# Imperative classification declares its Airflow, YuniKorn, and vLLM effects

2026-10-11. Follow-up to `174245_d3-aperture-publish-and-tp4-swap.md`.
The genus Choice and the single unquantized TP-4 swap stand. The
reconfiguration is part of the user trigger.

## Classification is imperative

Most runs start because a person asked. `POST /api/fsm/start` still
spawns the pipeline in-process. The catalogue entry `task.sdg_classify`
is a paused cron (`0 8 * * *`, `enabled=False`, runner `metaflow`,
`source=airflow`) whose only YuniKorn claim is one `light` GPU.
`classify.clock = airflow` already suppresses boot dispatch. A cron
tick is the wrong clock for a button.

The trigger has to articulate the whole reconfiguration, then Airflow
runs the Metaflow flow under it. The UI talks to the Atelier engine.
The engine declares to Signals. Only Signals talks to Airflow and
YuniKorn. Metaflow pods do not hold an Airflow URL. That is the same
two-hop rule as a Hermes session.

## What Hermes already does, and where it stops

`hsengine` `interactive.enter` declares before it starts Moshi:

1. `DeclareActivity(kind=interactive_session)` with the YuniKorn claims
   (`agent-rtc` GPU 1, Cerebras and compute at GPU 0).
2. Deny the session unless Signals echoes those claims and the
   activity is in force.
3. Start Moshi. Heartbeat the lease (60 s, TTL 180 s).
4. `leave` releases. Peers restore their desired sets.

The catalogue entry `interactive.agent_rtc` is `source=engine`: Signals
lists it and materialises no workload DAG. The run is
`coord_interactive_session` (pool `agent_rtc`, one slot). Postures are
empty. `yk_sentinel` says heavy thinking on GPUs 0–3 is Gaius's
standing claim and is not ceded. Dialog for that session is Cerebras.

`postures` means `hold-uptime`: cede an endpoint, do not restart it.
That is the right effect for a session that leaves Qwen standing. It
does not say "stop Qwen, serve `vllm-sr/d3` on the same four cards,
then put Qwen back."

## What a classification trigger declares

`POST /api/fsm/start` asks the Atelier engine to declare. It does not
start a pipeline thread. The declaration carries the user's
`source_id` and three effects. Signals refuses the start, and Atelier
denies it, unless the echo matches.

**Airflow.** One run of `atelier_sdg_classify`, id `act-<activity_id>`,
conf = the declaration. Pool `sdg_classify`, one slot, deferred-
inclusive, so a second user run waits rather than swapping the same
process twice. The DAG exists because the runner is Metaflow on
platform Kubernetes. The cron stays empty: a run exists when a user
declares it. `SyncWorkloads` still publishes the catalogue
(`runner=metaflow`, `source=airflow`) so the DAG is visible before it
is live.

**YuniKorn.** The activity's `claims[]` are the queue configuration
for as long as the run is `RUNNING`, judged by the capacity gate, and
retracted when the run ends. Heavy stays the standing claim: guaranteed
4, fenced, `maxapplications: 1`, GPUs 0–3. The light claim (one GPU,
ColBERT-Zero and the ModernBERT head taking turns) stays the claim
already on `task.sdg_classify`. Cards 4 and 5 stay with the Gaius
process and Moshi. `federation-queues.yaml` is not rewritten. An
in-force `interactive_session` is not precluded and is not preempted:
AgentRTC dialog is Cerebras, and Moshi keeps the agent-rtc token.

**vLLM.** A new repeated effect, not a posture string. Postures stay
`hold-uptime`.

```
endpoint:         gaius.endpoint.heavy
action:           replace
model:            vllm-sr/d3
tensor_parallel:  4
capability:       decision
route:            /v1/decisions
```

While the sweep phase is `RUNNING`, Gaius — which already watches
activities — stops the standing `Qwen/Qwen3.8-27B` process and starts
one unquantized process on the same four GPUs,
`vllm serve` with `--tensor-parallel-size 4` and the decision model
class. The YuniKorn application does not change. The command line
does. Leaving `RUNNING` (released, lapsed, or failed) restores
`DEPLOYMENT_PROFILES["thinking"]` even when the DAG died. Restore is
the protocol default. Gaius records the standing profile at replace
time and applies it itself, so a dead Metaflow pod is not what puts
Qwen back.

## The DAG around the flow

Airflow orders the effects. Metaflow does not swap vLLM.

```
assert_yk       claims → arbiter
apply_vllm      sensor: capability decision is SERVING on /v1/decisions
sweep           ClassificationFlow genus Choice + walk
restore_vllm    sensor: thinking is SERVING          trigger_rule=all_done
revisit         ClassificationFlow thinking revisit, disagreement set
close           release the lease; retract the activity's share
```

`restore_vllm` runs when the sweep fails. The sweep step starts only
after `decision` is healthy. The revisit starts only after `thinking`
is healthy. `instruct` is an operating profile of the Qwen process, so
it is absent during the sweep and present again after the restore.

Platform Metaflow is the compute: `@kubernetes` pods, YuniKorn
admission, artifacts on RustFS, the Signals metadata service. The pod
learns that `decision` is serving from the Atelier engine's
`ServerQuery`. It never opens the vLLM port and never opens Airflow.

## `/v1/decisions` is a vLLM capability, not a chat profile

Local vLLM, the process Gaius already launches, gains a model
capability named `decision`. The private route is `POST /v1/decisions`.
The body is the Decision contract: model, state, questions (Choice,
Noul, Score), optional images and video. The response is a distribution
over the caller's option keys. No generated tokens. The 255-option cap
stays. The published `d3_runtime.py` one-device loader is not this
process; the readout is the model class on the TP-4 server.

`thinking` and `instruct` stay operating profiles of `Qwen/Qwen3.8-27B`
on `/v1/chat/completions`. `decision` is not a third profile of that
checkpoint. It is advertised on `WorkloadOffer` only while the replace
effect is serving. Requesting it while Qwen is the served model fails
closed (`FAILED_PRECONDITION`), the same rule as asking for a
capability the peer does not host. `Engine/Complete` stays the chat
contract. The hosting engine is the only caller of `/v1/decisions`.
The lattice names the capability `decision` and never sees the URL.

The sweep maps top-1 probability to the code and the confidence, and
the next three to alternatives, into `llm_to_mass`. Column text is the
question state. The other DST sources stay out of that state. The
thinking revisit stays on Qwen after the restore.

## Deny, then restore

- Declare refused, or the echo omits the heavy effect or the light
  claim: the user start fails. Qwen keeps running. No Metaflow run.
- `decision` never becomes healthy: release, restore Qwen, FSM
  `ERROR`. No sweep.
- Sweep fails after the swap: `restore_vllm` still runs, then close.
- Heartbeats stop (lease TTL 180 s) or the horizon passes: the activity
  lapses, Gaius restores, the Backlog records the coordination miss.
- A second start while one activity is in force: the pool holds it.
  The FSM still refuses a second in-process run; the pool is the
  federation-level form of that refusal.
