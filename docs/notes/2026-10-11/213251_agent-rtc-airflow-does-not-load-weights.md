# Agent-rtc: Airflow observes, the owner starts the process

2026-10-11. Trace of the live path. It does not change Gaius weights.

## What runs

`interactive.enter` (`hsengine`) declares before it starts anything.

1. `declare_interactive` sends `DeclareActivity` to Signals.
   Peer `hermes`, kind `interactive_session`. Claims are
   `agent-rtc` GPU 1, Cerebras GPU 0, compute GPU 0.
   `interactive_postures()` is empty unless config says otherwise.
   The reason string says Gaius thinking stays up.
2. Signals `ActivityService.declare` checks the claims against the
   leaf max, writes the lease, and triggers Airflow run `act-<id>` of
   `coord_interactive_session`. Peers never call Airflow.
3. That DAG is `schedule=None`. Tasks are `declare` → `hold` → `close`.
   `hold` is a `SignalsActivitySensor` on the lease and occupies pool
   `agent_rtc` (one slot). Renew is a heartbeat. Release completes
   the sensor. Nobody patches the run state.
4. When the run is `RUNNING`, `_reconcile_shares` turns each claim
   into a `QueueShareRequest` whose floor is the claimed GPU, owned by
   the activity. When the run leaves force, the same call retracts the
   floor to 0. The share names a queue and a GPU count. It does not
   name a model.
5. Hermes denies the session unless Signals echoed the agent-rtc claim
   and the activity is in force, then `_moshi_on` posts to Hermes's own
   supervisor. Those weights are Moshi's. `leave` releases the lease.

## What Gaius does with that activity

Gaius watches `WatchActivities`. Three reactions exist, and none of
them load a checkpoint:

- **Cede.** Only a posture `gaius.endpoint.<alias>: hold-uptime`.
  `cede_endpoint` leaves a healthy process serving and stops chasing
  its uptime. Agent-rtc sends no such posture.
- **Preclude.** `precludes[]` defers new admits into named leaves.
  Agent-rtc's default list is empty. Heavy keeps admitting.
- **Run our own.** `workload_pass` inserts a `scheduled_tasks` row only
  when `peer=gaius` and `kind` is in the Gaius catalogue
  (`article_curate`, `theta_cycle`, …). `interactive_session` is not
  in that map. The row is work for an already-running model.

## Where weights actually change

The standing Qwen process is the boot preload (`vllm serve`, about
240s, `InitController`). A later load is in-process:
`ensure_capability` / `ensure_endpoint("thinking")` →
`vllm_controller.start_model(model_id=...)`, which starts a process
for an endpoint name. Nothing in the activity watcher or the
coordination DAG calls it. `plan_swap_for_reasoning` is a local plan
on the resource manager, not an Airflow task.

## What classification has to copy

Copy the activity shape: declare, echo, queue floor while `RUNNING`,
owner performs the side effect, release retracts the floor. Airflow
does not deploy weights. Hermes can start Moshi because Hermes owns
Moshi. Gaius owns the vLLM process, and today no activity field tells
it to call `start_model`. A classification activity becomes a weight
change only when Gaius, on seeing that activity `RUNNING`, deploys
the model the workload names and releases when the sweep is done.
The queue share remains the floor, not the checkpoint.
