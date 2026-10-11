# An imperative workload articulates the platform

2026-10-11. The agent-rtc trace is the pattern. The same five steps
cover every Airflow + Metaflow workload in the federation. Classification
is the first chain that also changes the vLLM weights.

Signals already says this in `workloads_catalog.py` (2026-09-07): ordering,
with the YuniKorn configuration Signals applies for the duration of the
workflow, Metaflow or otherwise, including the interactive session.
`SyncWorkloads` publishes each engine's catalogue. `make_workload_dag`
is one activity: declare, hold, close. `make_chain_dag` runs the next
declare only after the previous close, and that next declare is what
tells the arbiter to assert the new configuration.

What the catalogue carries today is the queue (`claims_for`), the
runner label, the horizon, and postures that only cede. `runner=metaflow`
is already set on article curation, ambient synthesis, and others.
`workload_pass` does not read it. On a `RUNNING` activity whose peer is
Gaius it inserts a `scheduled_tasks` row. Nothing in that path calls
`start_model`.

## The five steps

Every workload, imperative or on a clock, takes the same path. A user
request and a cron tick differ only in who fires the declare.

1. **Request.** The product engine receives it. A person, a cron, or an
   ended asset. The engine is the only hop. Signals is the only process
   that talks to Airflow.
2. **Airflow.** One activity run. Imperative: `DeclareActivity`, the
   DAG has `schedule=None`, as `coord_interactive_session` does.
   Scheduled: the catalogue DAG's declare task calls
   `declare_scheduled`. Tasks are declare, hold, close. A chain is
   several of those in one DAG. The pool is one slot.
3. **YuniKorn.** While the activity is `RUNNING`, `_reconcile_shares`
   asserts the claims as the floor, owned by the activity. `precludes`
   retracts the mutually exclusive leaf. Close retracts the floor.
   Peers do not write `federation-queues.yaml`.
4. **Weights.** The engine that owns the process sees the activity
   `RUNNING` and performs the side effect the workload names, then
   heartbeats. Hermes owns Moshi and starts it. Gaius owns the one
   tensor-parallel-4 vLLM process. The workload names the model. Gaius
   deploys it and advertises the capability mix that model serves.
   `Qwen/Qwen3.8-27B` serves `thinking` and `instruct`. `vllm-sr/d3`
   serves `decisions`. The hold does not start compute until that
   capability is healthy. A workload that names the standing model
   skips the deploy.
5. **Metaflow.** `runner=metaflow` starts the `@kubernetes` flow on the
   Signals RKE2 platform. The pods call the capability. They do not
   load weights and they do not open Airflow. The owner heartbeats
   until the flow ends, then releases. `runner=task` keeps today's
   `scheduled_tasks` row. Same activity either way.

Close, lapse, and failure restore the standing deployment when no
successor activity is in force. That is `leave` on the agent-rtc path:
the owner puts its process back, and the floor retracts.

## Classification is one chain

A user `POST /api/fsm/start` declares a chain, not a pipeline thread.

- `decisions`. Claims `root.internal.inference.decisions` at 4,
  precludes `heavy`. Model `vllm-sr/d3`. Capability `decisions`.
  Runner metaflow: the genus sweep.
- `thinking`. Claims `heavy` at 4. Model `Qwen/Qwen3.8-27B`. Capability
  `thinking`. Runner metaflow: the revisit.

The queues stay mutually exclusive because both models want the same
four cards. The chain's close of the first link is what asserts the
second. Cards 4 and 5 stay outside the pair.

Article curation, ambient synthesis, theta cycle, and an Ægir run use
this same chain shape. Most of them name the standing model, so step 4
is a health check. A workload that needs a different checkpoint names
that checkpoint. The platform does not grow a second way to say it.
