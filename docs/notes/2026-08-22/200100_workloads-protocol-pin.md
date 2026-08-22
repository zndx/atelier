# WORKLOADS pin a2a8d2b + queue-share UUIDv7

Protocol `9649662` → `a2a8d2b` is a fast-forward (`ServerQuery
WORKLOADS`). Additive: enum 7, `WorkloadHint` field 9. No wire conflict
with QueueHint.

Merge requirement: engine stubs now register `RecordLineage`. The
servicer must define that method (UNIMPLEMENTED → Signals Atlas SoR)
or `add_EngineServicer_to_server` AttributeErrors.

Queue share matches Aegir: RFC 9562 UUIDv7 on every RequestQueueShare;
admit occupancy; WRK end zero-floor + valid_until; REJECTED does not
admit; heavy/medium/light from tp×pp; model/tp/pp on WORKLOADS, never
in the queue name. Never write queues.yaml.
