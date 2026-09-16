# Live: atelier_sdg_classify paused on Airflow

After atelier.service recycle (03:41 UTC) and `fs.inotify.max_user_instances=8192`:

- Engine SCHEDULES serves `task.sdg_classify` (light×1, not heavy, not embedding).
- First SyncWorkloads: embedding leaf GPU max 0 (live YK); then Airflow
  Variable PATCH 500 (`PendingRollbackError` on api-server).
- Claims fixed to `root.internal.inference.light` × 1. `source=airflow`
  even while `enabled=False` (engine_declared was the agent-rtc shape).
- Recycled `airflow-api-server`. SyncWorkloads accepted: state=paused,
  dag_id=`atelier_sdg_classify`. Dag-processor created the ORM DAG.
- Do **not** unpause until a K8s Metaflow run has a YK app-id.
  Dag-processor logged `next_dagrun=2026-09-15 08:00` — catch-up risk
  if unpaused; keep paused.
- Platform Metaflow discovery: scheduler healthy, metadata ping ok.
