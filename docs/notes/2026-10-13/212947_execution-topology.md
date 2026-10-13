# Classify runs are an Airflow activity

2026-10-13. Signals is the only process that talks to Airflow. An
imperative Metaflow workflow is declared as one activity, and the run
configuration is that activity.

`classify.clock` stays `airflow`. `POST /api/fsm/start` builds an
`ExecutionTopology` from the catalogue entry plus the trigger arguments
and calls `DeclareActivity`. It does not start
`run_classification_pipeline`. A dark or refusing Scheduler returns
`{started: false, scheduled: false, error}`. `clock=gateway` is still
the CAI in-process thread.

The activity `reason` is the configuration JSON: peer `atelier`, kind
`sdg_classify`, runner `metaflow`, trigger `imperative`, empty cron,
source, target, reference, sweep capability `decisions` model
`vllm-sr/d3-flash` width 2, revisit capability `thinking` model
`Qwen/Qwen3.8-27B`, claims one light GPU, precludes
`agent-rtc` / `medium` / `extract`, horizon 4h. There is no free-form
parameter bag on `DeclareActivityRequest`. Postures are not used as a
parameter bag.

`task.sdg_classify` stays `enabled=False`. Cron changed from
`0 8 * * *` to empty, so the next `SyncWorkloads` publishes an
imperative paused DAG. Precludes are on the `ScheduleHint`. This slice
did not run `airflow create` and did not restart the engine.

The engine starts an activity owner beside workload sync. The owner
lists this peer's `sdg_classify` activities. When one is
`ACTIVITY_RUNNING` it starts `python -m atelier.flows.run_classify`
with argv taken from the activity reason, including
`--run-configuration`. `run_classify` still inserts `--with kubernetes`.
The flow reads that document in `start` and does not re-read the
catalogue for model or capability. A hand-started flow with an empty
parameter builds the same document from the catalogue. While the child
is alive the owner calls `RenewActivity` with the activity's existing
horizon. Signals' lease TTL is 180s and the poll is 30s. A horizon that
has already passed is not extended. Exit 0 releases
`completed: metaflow exit 0`. A nonzero exit releases `failed:`, which
Signals records as `#CO.0000000F.NOEFFECT`. A refused release stays
registered and is retried. It does not start a second flow. A Signals
with no `ListActivities` is fail-open: the owner logs once and starts
nothing. A vanished activity is not SIGTERM'd in this slice.

`trigger_pipeline_run` and `apply_and_rerun` use the same dispatch
reading. A declaration is scheduled, not an in-process start.
`apply_and_rerun` prints `applied_and_scheduled`. A refusal fails the
task and the CLI returns 5.

Declare refusal is `#CO.0000000A.NODECLARE`. `#CO.0000000D` is already
Gaius `MISSTICK`. Renew and release reuse `#CO.00000006.RENEWFAIL` and
`#CO.00000007.RELEASEFAIL`.

Tests on the fake Scheduler at `127.0.0.1:0`: topology, declare, owner
(start, renew, release, refused release, unimplemented list), dispatch
callers, catalogue, workload sync, airflow clock, classification flow,
platform Metaflow. 62 passed. The running atelier engine was left as it
was, so it will not poll until the next process start.
