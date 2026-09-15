from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProcessKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROCESS_KIND_UNSPECIFIED: _ClassVar[ProcessKind]
    PROCESS_KIND_SYSTEMD_UNIT: _ClassVar[ProcessKind]
    PROCESS_KIND_ENGINE_DAEMON: _ClassVar[ProcessKind]
    PROCESS_KIND_SERVING_ENDPOINT: _ClassVar[ProcessKind]
    PROCESS_KIND_SCHEDULED_TASK: _ClassVar[ProcessKind]
    PROCESS_KIND_METAFLOW_FLOW: _ClassVar[ProcessKind]
    PROCESS_KIND_AIRFLOW_DAG: _ClassVar[ProcessKind]
    PROCESS_KIND_PG_CRON_JOB: _ClassVar[ProcessKind]
    PROCESS_KIND_EXTERNAL_SERVICE: _ClassVar[ProcessKind]

class SourceKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SOURCE_KIND_UNSPECIFIED: _ClassVar[SourceKind]
    SOURCE_KIND_ENGINE_GRPC: _ClassVar[SourceKind]
    SOURCE_KIND_METAFLOW_STORE: _ClassVar[SourceKind]
    SOURCE_KIND_AIRFLOW_METADATA: _ClassVar[SourceKind]
    SOURCE_KIND_PG_CRON: _ClassVar[SourceKind]
    SOURCE_KIND_SYSTEMD: _ClassVar[SourceKind]
    SOURCE_KIND_TASK_QUEUE: _ClassVar[SourceKind]
    SOURCE_KIND_HTTP_PROBE: _ClassVar[SourceKind]
    SOURCE_KIND_REPORTED: _ClassVar[SourceKind]

class RestartStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESTART_STRATEGY_UNSPECIFIED: _ClassVar[RestartStrategy]
    RESTART_STRATEGY_ONE_FOR_ONE: _ClassVar[RestartStrategy]
    RESTART_STRATEGY_ONE_FOR_ALL: _ClassVar[RestartStrategy]
    RESTART_STRATEGY_REST_FOR_ONE: _ClassVar[RestartStrategy]
    RESTART_STRATEGY_NONE: _ClassVar[RestartStrategy]

class ExpectationCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXPECTATION_CATEGORY_UNSPECIFIED: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_TICK: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_HOURLY_SETTLED: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_SLOT: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_DAILY_DEPENDENT: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_JUDGED: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_CHRONIC_AUDIT: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_OPERATOR_WINDOW: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_TRANSIENT: _ClassVar[ExpectationCategory]
    EXPECTATION_CATEGORY_COORDINATION: _ClassVar[ExpectationCategory]

class Channel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHANNEL_UNSPECIFIED: _ClassVar[Channel]
    CHANNEL_LOG: _ClassVar[Channel]
    CHANNEL_AGENDA_EVENT: _ClassVar[Channel]
    CHANNEL_REMINDER: _ClassVar[Channel]
    CHANNEL_BRIEFING: _ClassVar[Channel]
    CHANNEL_DISCUSSION: _ClassVar[Channel]

class GateKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GATE_KIND_UNSPECIFIED: _ClassVar[GateKind]
    GATE_KIND_FLOW_REACHED_STEP: _ClassVar[GateKind]
    GATE_KIND_FOOTPRINT_ADVANCED: _ClassVar[GateKind]
    GATE_KIND_FRESHNESS_WITHIN: _ClassVar[GateKind]
    GATE_KIND_SURFACE_WELLFORMED: _ClassVar[GateKind]
    GATE_KIND_JUDGE_RUBRIC: _ClassVar[GateKind]
    GATE_KIND_PROBE_PASS: _ClassVar[GateKind]
    GATE_KIND_POSITION_REPORTED: _ClassVar[GateKind]
    GATE_KIND_SHARE_APPLIED: _ClassVar[GateKind]
    GATE_KIND_BACKLOG_SURFACED: _ClassVar[GateKind]

class Tier(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TIER_UNSPECIFIED: _ClassVar[Tier]
    TIER_GOLD: _ClassVar[Tier]
    TIER_SILVER: _ClassVar[Tier]

class PositionSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    POSITION_SOURCE_UNSPECIFIED: _ClassVar[PositionSource]
    POSITION_SOURCE_REPORTED: _ClassVar[PositionSource]
    POSITION_SOURCE_OBSERVED: _ClassVar[PositionSource]
    POSITION_SOURCE_INFERRED: _ClassVar[PositionSource]

class ProcessHealth(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROCESS_HEALTH_UNSPECIFIED: _ClassVar[ProcessHealth]
    PROCESS_HEALTH_SERVING: _ClassVar[ProcessHealth]
    PROCESS_HEALTH_STARTING: _ClassVar[ProcessHealth]
    PROCESS_HEALTH_MISSING: _ClassVar[ProcessHealth]
    PROCESS_HEALTH_STALLED: _ClassVar[ProcessHealth]
    PROCESS_HEALTH_UNSUPERVISED: _ClassVar[ProcessHealth]
    PROCESS_HEALTH_UNOBSERVABLE: _ClassVar[ProcessHealth]

class BacklogState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BACKLOG_STATE_UNSPECIFIED: _ClassVar[BacklogState]
    BACKLOG_STATE_OK: _ClassVar[BacklogState]
    BACKLOG_STATE_DEFERRED: _ClassVar[BacklogState]
    BACKLOG_STATE_MISSED: _ClassVar[BacklogState]
    BACKLOG_STATE_FAILED: _ClassVar[BacklogState]
    BACKLOG_STATE_JUDGED_FAIL: _ClassVar[BacklogState]
    BACKLOG_STATE_AWAITING: _ClassVar[BacklogState]
    BACKLOG_STATE_UNKNOWN: _ClassVar[BacklogState]

class TaskState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_STATE_UNSPECIFIED: _ClassVar[TaskState]
    TASK_STATE_QUEUED: _ClassVar[TaskState]
    TASK_STATE_CLAIMED: _ClassVar[TaskState]
    TASK_STATE_HEARTBEAT: _ClassVar[TaskState]
    TASK_STATE_COMPLETED: _ClassVar[TaskState]
    TASK_STATE_DEFERRED: _ClassVar[TaskState]
    TASK_STATE_FAILED: _ClassVar[TaskState]
    TASK_STATE_ERROR: _ClassVar[TaskState]
    TASK_STATE_STALLED: _ClassVar[TaskState]
    TASK_STATE_YIELDED: _ClassVar[TaskState]
    TASK_STATE_SKIPPED: _ClassVar[TaskState]
    TASK_STATE_RESET: _ClassVar[TaskState]

class AdmissionPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ADMISSION_PHASE_UNSPECIFIED: _ClassVar[AdmissionPhase]
    ADMISSION_PHASE_PENDING: _ClassVar[AdmissionPhase]
    ADMISSION_PHASE_ADMITTED: _ClassVar[AdmissionPhase]
    ADMISSION_PHASE_NOTADMITTED: _ClassVar[AdmissionPhase]
    ADMISSION_PHASE_RETIRED: _ClassVar[AdmissionPhase]

class Verdict(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VERDICT_UNSPECIFIED: _ClassVar[Verdict]
    VERDICT_PASS: _ClassVar[Verdict]
    VERDICT_FAIL: _ClassVar[Verdict]
    VERDICT_INCONCLUSIVE: _ClassVar[Verdict]
    VERDICT_ERROR: _ClassVar[Verdict]

class IncidentState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INCIDENT_STATE_UNSPECIFIED: _ClassVar[IncidentState]
    INCIDENT_STATE_ACTIVE: _ClassVar[IncidentState]
    INCIDENT_STATE_HEALING: _ClassVar[IncidentState]
    INCIDENT_STATE_RECOVERING: _ClassVar[IncidentState]
    INCIDENT_STATE_RESOLVED: _ClassVar[IncidentState]
    INCIDENT_STATE_MANUAL_REQUIRED: _ClassVar[IncidentState]

class ServingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVING_STATUS_UNSPECIFIED: _ClassVar[ServingStatus]
    SERVING_STATUS_SERVING: _ClassVar[ServingStatus]
    SERVING_STATUS_STARTING: _ClassVar[ServingStatus]
    SERVING_STATUS_DEGRADED: _ClassVar[ServingStatus]
    SERVING_STATUS_FAILED: _ClassVar[ServingStatus]
    SERVING_STATUS_ABSENT: _ClassVar[ServingStatus]

class ServingPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVING_PHASE_UNSPECIFIED: _ClassVar[ServingPhase]
    SERVING_PHASE_SETTLED: _ClassVar[ServingPhase]
    SERVING_PHASE_TRANSITIONING: _ClassVar[ServingPhase]

class GoodbyeReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GOODBYE_REASON_UNSPECIFIED: _ClassVar[GoodbyeReason]
    GOODBYE_REASON_SUPERSEDED: _ClassVar[GoodbyeReason]
    GOODBYE_REASON_SHUTDOWN: _ClassVar[GoodbyeReason]
    GOODBYE_REASON_PROJECT_MISMATCH: _ClassVar[GoodbyeReason]
PROCESS_KIND_UNSPECIFIED: ProcessKind
PROCESS_KIND_SYSTEMD_UNIT: ProcessKind
PROCESS_KIND_ENGINE_DAEMON: ProcessKind
PROCESS_KIND_SERVING_ENDPOINT: ProcessKind
PROCESS_KIND_SCHEDULED_TASK: ProcessKind
PROCESS_KIND_METAFLOW_FLOW: ProcessKind
PROCESS_KIND_AIRFLOW_DAG: ProcessKind
PROCESS_KIND_PG_CRON_JOB: ProcessKind
PROCESS_KIND_EXTERNAL_SERVICE: ProcessKind
SOURCE_KIND_UNSPECIFIED: SourceKind
SOURCE_KIND_ENGINE_GRPC: SourceKind
SOURCE_KIND_METAFLOW_STORE: SourceKind
SOURCE_KIND_AIRFLOW_METADATA: SourceKind
SOURCE_KIND_PG_CRON: SourceKind
SOURCE_KIND_SYSTEMD: SourceKind
SOURCE_KIND_TASK_QUEUE: SourceKind
SOURCE_KIND_HTTP_PROBE: SourceKind
SOURCE_KIND_REPORTED: SourceKind
RESTART_STRATEGY_UNSPECIFIED: RestartStrategy
RESTART_STRATEGY_ONE_FOR_ONE: RestartStrategy
RESTART_STRATEGY_ONE_FOR_ALL: RestartStrategy
RESTART_STRATEGY_REST_FOR_ONE: RestartStrategy
RESTART_STRATEGY_NONE: RestartStrategy
EXPECTATION_CATEGORY_UNSPECIFIED: ExpectationCategory
EXPECTATION_CATEGORY_TICK: ExpectationCategory
EXPECTATION_CATEGORY_HOURLY_SETTLED: ExpectationCategory
EXPECTATION_CATEGORY_SLOT: ExpectationCategory
EXPECTATION_CATEGORY_DAILY_DEPENDENT: ExpectationCategory
EXPECTATION_CATEGORY_JUDGED: ExpectationCategory
EXPECTATION_CATEGORY_CHRONIC_AUDIT: ExpectationCategory
EXPECTATION_CATEGORY_OPERATOR_WINDOW: ExpectationCategory
EXPECTATION_CATEGORY_TRANSIENT: ExpectationCategory
EXPECTATION_CATEGORY_COORDINATION: ExpectationCategory
CHANNEL_UNSPECIFIED: Channel
CHANNEL_LOG: Channel
CHANNEL_AGENDA_EVENT: Channel
CHANNEL_REMINDER: Channel
CHANNEL_BRIEFING: Channel
CHANNEL_DISCUSSION: Channel
GATE_KIND_UNSPECIFIED: GateKind
GATE_KIND_FLOW_REACHED_STEP: GateKind
GATE_KIND_FOOTPRINT_ADVANCED: GateKind
GATE_KIND_FRESHNESS_WITHIN: GateKind
GATE_KIND_SURFACE_WELLFORMED: GateKind
GATE_KIND_JUDGE_RUBRIC: GateKind
GATE_KIND_PROBE_PASS: GateKind
GATE_KIND_POSITION_REPORTED: GateKind
GATE_KIND_SHARE_APPLIED: GateKind
GATE_KIND_BACKLOG_SURFACED: GateKind
TIER_UNSPECIFIED: Tier
TIER_GOLD: Tier
TIER_SILVER: Tier
POSITION_SOURCE_UNSPECIFIED: PositionSource
POSITION_SOURCE_REPORTED: PositionSource
POSITION_SOURCE_OBSERVED: PositionSource
POSITION_SOURCE_INFERRED: PositionSource
PROCESS_HEALTH_UNSPECIFIED: ProcessHealth
PROCESS_HEALTH_SERVING: ProcessHealth
PROCESS_HEALTH_STARTING: ProcessHealth
PROCESS_HEALTH_MISSING: ProcessHealth
PROCESS_HEALTH_STALLED: ProcessHealth
PROCESS_HEALTH_UNSUPERVISED: ProcessHealth
PROCESS_HEALTH_UNOBSERVABLE: ProcessHealth
BACKLOG_STATE_UNSPECIFIED: BacklogState
BACKLOG_STATE_OK: BacklogState
BACKLOG_STATE_DEFERRED: BacklogState
BACKLOG_STATE_MISSED: BacklogState
BACKLOG_STATE_FAILED: BacklogState
BACKLOG_STATE_JUDGED_FAIL: BacklogState
BACKLOG_STATE_AWAITING: BacklogState
BACKLOG_STATE_UNKNOWN: BacklogState
TASK_STATE_UNSPECIFIED: TaskState
TASK_STATE_QUEUED: TaskState
TASK_STATE_CLAIMED: TaskState
TASK_STATE_HEARTBEAT: TaskState
TASK_STATE_COMPLETED: TaskState
TASK_STATE_DEFERRED: TaskState
TASK_STATE_FAILED: TaskState
TASK_STATE_ERROR: TaskState
TASK_STATE_STALLED: TaskState
TASK_STATE_YIELDED: TaskState
TASK_STATE_SKIPPED: TaskState
TASK_STATE_RESET: TaskState
ADMISSION_PHASE_UNSPECIFIED: AdmissionPhase
ADMISSION_PHASE_PENDING: AdmissionPhase
ADMISSION_PHASE_ADMITTED: AdmissionPhase
ADMISSION_PHASE_NOTADMITTED: AdmissionPhase
ADMISSION_PHASE_RETIRED: AdmissionPhase
VERDICT_UNSPECIFIED: Verdict
VERDICT_PASS: Verdict
VERDICT_FAIL: Verdict
VERDICT_INCONCLUSIVE: Verdict
VERDICT_ERROR: Verdict
INCIDENT_STATE_UNSPECIFIED: IncidentState
INCIDENT_STATE_ACTIVE: IncidentState
INCIDENT_STATE_HEALING: IncidentState
INCIDENT_STATE_RECOVERING: IncidentState
INCIDENT_STATE_RESOLVED: IncidentState
INCIDENT_STATE_MANUAL_REQUIRED: IncidentState
SERVING_STATUS_UNSPECIFIED: ServingStatus
SERVING_STATUS_SERVING: ServingStatus
SERVING_STATUS_STARTING: ServingStatus
SERVING_STATUS_DEGRADED: ServingStatus
SERVING_STATUS_FAILED: ServingStatus
SERVING_STATUS_ABSENT: ServingStatus
SERVING_PHASE_UNSPECIFIED: ServingPhase
SERVING_PHASE_SETTLED: ServingPhase
SERVING_PHASE_TRANSITIONING: ServingPhase
GOODBYE_REASON_UNSPECIFIED: GoodbyeReason
GOODBYE_REASON_SUPERSEDED: GoodbyeReason
GOODBYE_REASON_SHUTDOWN: GoodbyeReason
GOODBYE_REASON_PROJECT_MISMATCH: GoodbyeReason

class Observation(_message.Message):
    __slots__ = ("source", "locator", "poll_seconds", "via")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    LOCATOR_FIELD_NUMBER: _ClassVar[int]
    POLL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    VIA_FIELD_NUMBER: _ClassVar[int]
    source: SourceKind
    locator: str
    poll_seconds: int
    via: str
    def __init__(self, source: _Optional[_Union[SourceKind, str]] = ..., locator: _Optional[str] = ..., poll_seconds: _Optional[int] = ..., via: _Optional[str] = ...) -> None: ...

class Cadence(_message.Message):
    __slots__ = ("cron", "expected_period_seconds", "net_seconds", "net_slot")
    CRON_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_PERIOD_SECONDS_FIELD_NUMBER: _ClassVar[int]
    NET_SECONDS_FIELD_NUMBER: _ClassVar[int]
    NET_SLOT_FIELD_NUMBER: _ClassVar[int]
    cron: str
    expected_period_seconds: int
    net_seconds: int
    net_slot: int
    def __init__(self, cron: _Optional[str] = ..., expected_period_seconds: _Optional[int] = ..., net_seconds: _Optional[int] = ..., net_slot: _Optional[int] = ...) -> None: ...

class Expectation(_message.Message):
    __slots__ = ("category", "horizon_slot", "channel", "rationale", "feeds", "backfill_depth", "run_net_slot", "awaits")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    HORIZON_SLOT_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    FEEDS_FIELD_NUMBER: _ClassVar[int]
    BACKFILL_DEPTH_FIELD_NUMBER: _ClassVar[int]
    RUN_NET_SLOT_FIELD_NUMBER: _ClassVar[int]
    AWAITS_FIELD_NUMBER: _ClassVar[int]
    category: ExpectationCategory
    horizon_slot: int
    channel: Channel
    rationale: str
    feeds: str
    backfill_depth: int
    run_net_slot: int
    awaits: str
    def __init__(self, category: _Optional[_Union[ExpectationCategory, str]] = ..., horizon_slot: _Optional[int] = ..., channel: _Optional[_Union[Channel, str]] = ..., rationale: _Optional[str] = ..., feeds: _Optional[str] = ..., backfill_depth: _Optional[int] = ..., run_net_slot: _Optional[int] = ..., awaits: _Optional[str] = ...) -> None: ...

class Process(_message.Message):
    __slots__ = ("id", "kind", "parent", "restart", "observation", "warmup_seconds", "cadence", "depends_on", "reports_positions", "attrs", "resources", "expectation")
    class AttrsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    RESTART_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    WARMUP_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CADENCE_FIELD_NUMBER: _ClassVar[int]
    DEPENDS_ON_FIELD_NUMBER: _ClassVar[int]
    REPORTS_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    ATTRS_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    EXPECTATION_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: ProcessKind
    parent: str
    restart: RestartStrategy
    observation: Observation
    warmup_seconds: int
    cadence: Cadence
    depends_on: _containers.RepeatedScalarFieldContainer[str]
    reports_positions: bool
    attrs: _containers.ScalarMap[str, str]
    resources: _containers.RepeatedCompositeFieldContainer[ResourceIntent]
    expectation: Expectation
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[_Union[ProcessKind, str]] = ..., parent: _Optional[str] = ..., restart: _Optional[_Union[RestartStrategy, str]] = ..., observation: _Optional[_Union[Observation, _Mapping]] = ..., warmup_seconds: _Optional[int] = ..., cadence: _Optional[_Union[Cadence, _Mapping]] = ..., depends_on: _Optional[_Iterable[str]] = ..., reports_positions: bool = ..., attrs: _Optional[_Mapping[str, str]] = ..., resources: _Optional[_Iterable[_Union[ResourceIntent, _Mapping]]] = ..., expectation: _Optional[_Union[Expectation, _Mapping]] = ...) -> None: ...

class Gate(_message.Message):
    __slots__ = ("name", "kind", "params", "judge_final_call", "tier", "rationale")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    JUDGE_FINAL_CALL_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: GateKind
    params: _containers.ScalarMap[str, str]
    judge_final_call: bool
    tier: Tier
    rationale: str
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[_Union[GateKind, str]] = ..., params: _Optional[_Mapping[str, str]] = ..., judge_final_call: bool = ..., tier: _Optional[_Union[Tier, str]] = ..., rationale: _Optional[str] = ...) -> None: ...

class Horizon(_message.Message):
    __slots__ = ("window_seconds", "resolve_within_phase_only")
    WINDOW_SECONDS_FIELD_NUMBER: _ClassVar[int]
    RESOLVE_WITHIN_PHASE_ONLY_FIELD_NUMBER: _ClassVar[int]
    window_seconds: int
    resolve_within_phase_only: bool
    def __init__(self, window_seconds: _Optional[int] = ..., resolve_within_phase_only: bool = ...) -> None: ...

class ResourceIntent(_message.Message):
    __slots__ = ("leaf", "workload", "occupancy", "floor", "priority", "applications", "rationale")
    LEAF_FIELD_NUMBER: _ClassVar[int]
    WORKLOAD_FIELD_NUMBER: _ClassVar[int]
    OCCUPANCY_FIELD_NUMBER: _ClassVar[int]
    FLOOR_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    APPLICATIONS_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    leaf: str
    workload: str
    occupancy: int
    floor: int
    priority: int
    applications: int
    rationale: str
    def __init__(self, leaf: _Optional[str] = ..., workload: _Optional[str] = ..., occupancy: _Optional[int] = ..., floor: _Optional[int] = ..., priority: _Optional[int] = ..., applications: _Optional[int] = ..., rationale: _Optional[str] = ...) -> None: ...

class Phase(_message.Message):
    __slots__ = ("id", "order", "steps", "gate", "horizon", "resources")
    ID_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    GATE_FIELD_NUMBER: _ClassVar[int]
    HORIZON_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    id: str
    order: int
    steps: _containers.RepeatedScalarFieldContainer[str]
    gate: Gate
    horizon: Horizon
    resources: _containers.RepeatedCompositeFieldContainer[ResourceIntent]
    def __init__(self, id: _Optional[str] = ..., order: _Optional[int] = ..., steps: _Optional[_Iterable[str]] = ..., gate: _Optional[_Union[Gate, _Mapping]] = ..., horizon: _Optional[_Union[Horizon, _Mapping]] = ..., resources: _Optional[_Iterable[_Union[ResourceIntent, _Mapping]]] = ...) -> None: ...

class Machine(_message.Message):
    __slots__ = ("id", "process", "phases", "default_horizon", "surfaced_result")
    ID_FIELD_NUMBER: _ClassVar[int]
    PROCESS_FIELD_NUMBER: _ClassVar[int]
    PHASES_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_HORIZON_FIELD_NUMBER: _ClassVar[int]
    SURFACED_RESULT_FIELD_NUMBER: _ClassVar[int]
    id: str
    process: str
    phases: _containers.RepeatedCompositeFieldContainer[Phase]
    default_horizon: Horizon
    surfaced_result: str
    def __init__(self, id: _Optional[str] = ..., process: _Optional[str] = ..., phases: _Optional[_Iterable[_Union[Phase, _Mapping]]] = ..., default_horizon: _Optional[_Union[Horizon, _Mapping]] = ..., surfaced_result: _Optional[str] = ...) -> None: ...

class Objective(_message.Message):
    __slots__ = ("name", "surfaced_result", "machines", "gates", "cadence_cron", "horizon", "resolves", "attrs", "expectation")
    class AttrsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    SURFACED_RESULT_FIELD_NUMBER: _ClassVar[int]
    MACHINES_FIELD_NUMBER: _ClassVar[int]
    GATES_FIELD_NUMBER: _ClassVar[int]
    CADENCE_CRON_FIELD_NUMBER: _ClassVar[int]
    HORIZON_FIELD_NUMBER: _ClassVar[int]
    RESOLVES_FIELD_NUMBER: _ClassVar[int]
    ATTRS_FIELD_NUMBER: _ClassVar[int]
    EXPECTATION_FIELD_NUMBER: _ClassVar[int]
    name: str
    surfaced_result: str
    machines: _containers.RepeatedScalarFieldContainer[str]
    gates: _containers.RepeatedCompositeFieldContainer[Gate]
    cadence_cron: str
    horizon: Horizon
    resolves: _containers.RepeatedScalarFieldContainer[str]
    attrs: _containers.ScalarMap[str, str]
    expectation: Expectation
    def __init__(self, name: _Optional[str] = ..., surfaced_result: _Optional[str] = ..., machines: _Optional[_Iterable[str]] = ..., gates: _Optional[_Iterable[_Union[Gate, _Mapping]]] = ..., cadence_cron: _Optional[str] = ..., horizon: _Optional[_Union[Horizon, _Mapping]] = ..., resolves: _Optional[_Iterable[str]] = ..., attrs: _Optional[_Mapping[str, str]] = ..., expectation: _Optional[_Union[Expectation, _Mapping]] = ...) -> None: ...

class Budgets(_message.Message):
    __slots__ = ("recycle_k", "recycle_window_seconds", "recycle_cooldown_seconds", "judge_consults_per_day", "rubric_judgments_per_day")
    RECYCLE_K_FIELD_NUMBER: _ClassVar[int]
    RECYCLE_WINDOW_SECONDS_FIELD_NUMBER: _ClassVar[int]
    RECYCLE_COOLDOWN_SECONDS_FIELD_NUMBER: _ClassVar[int]
    JUDGE_CONSULTS_PER_DAY_FIELD_NUMBER: _ClassVar[int]
    RUBRIC_JUDGMENTS_PER_DAY_FIELD_NUMBER: _ClassVar[int]
    recycle_k: int
    recycle_window_seconds: int
    recycle_cooldown_seconds: int
    judge_consults_per_day: int
    rubric_judgments_per_day: int
    def __init__(self, recycle_k: _Optional[int] = ..., recycle_window_seconds: _Optional[int] = ..., recycle_cooldown_seconds: _Optional[int] = ..., judge_consults_per_day: _Optional[int] = ..., rubric_judgments_per_day: _Optional[int] = ...) -> None: ...

class Supervisor(_message.Message):
    __slots__ = ("project", "spec_version", "engine_build", "processes", "machines", "objectives", "budgets", "engine_grpc", "attrs", "supervisor_grpc")
    class AttrsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENGINE_BUILD_FIELD_NUMBER: _ClassVar[int]
    PROCESSES_FIELD_NUMBER: _ClassVar[int]
    MACHINES_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVES_FIELD_NUMBER: _ClassVar[int]
    BUDGETS_FIELD_NUMBER: _ClassVar[int]
    ENGINE_GRPC_FIELD_NUMBER: _ClassVar[int]
    ATTRS_FIELD_NUMBER: _ClassVar[int]
    SUPERVISOR_GRPC_FIELD_NUMBER: _ClassVar[int]
    project: str
    spec_version: str
    engine_build: str
    processes: _containers.RepeatedCompositeFieldContainer[Process]
    machines: _containers.RepeatedCompositeFieldContainer[Machine]
    objectives: _containers.RepeatedCompositeFieldContainer[Objective]
    budgets: Budgets
    engine_grpc: str
    attrs: _containers.ScalarMap[str, str]
    supervisor_grpc: str
    def __init__(self, project: _Optional[str] = ..., spec_version: _Optional[str] = ..., engine_build: _Optional[str] = ..., processes: _Optional[_Iterable[_Union[Process, _Mapping]]] = ..., machines: _Optional[_Iterable[_Union[Machine, _Mapping]]] = ..., objectives: _Optional[_Iterable[_Union[Objective, _Mapping]]] = ..., budgets: _Optional[_Union[Budgets, _Mapping]] = ..., engine_grpc: _Optional[str] = ..., attrs: _Optional[_Mapping[str, str]] = ..., supervisor_grpc: _Optional[str] = ...) -> None: ...

class Position(_message.Message):
    __slots__ = ("process", "machine", "phase", "run_id", "step", "momentum", "source", "at_unix_ms", "epoch")
    PROCESS_FIELD_NUMBER: _ClassVar[int]
    MACHINE_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    process: str
    machine: str
    phase: str
    run_id: str
    step: str
    momentum: int
    source: PositionSource
    at_unix_ms: int
    epoch: str
    def __init__(self, process: _Optional[str] = ..., machine: _Optional[str] = ..., phase: _Optional[str] = ..., run_id: _Optional[str] = ..., step: _Optional[str] = ..., momentum: _Optional[int] = ..., source: _Optional[_Union[PositionSource, str]] = ..., at_unix_ms: _Optional[int] = ..., epoch: _Optional[str] = ...) -> None: ...

class PositionReport(_message.Message):
    __slots__ = ("position", "detail")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    position: Position
    detail: str
    def __init__(self, position: _Optional[_Union[Position, _Mapping]] = ..., detail: _Optional[str] = ...) -> None: ...

class Ack(_message.Message):
    __slots__ = ("accepted", "note")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    note: str
    def __init__(self, accepted: bool = ..., note: _Optional[str] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("process", "at_unix_ms")
    PROCESS_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    process: str
    at_unix_ms: int
    def __init__(self, process: _Optional[str] = ..., at_unix_ms: _Optional[int] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ProcessStatus(_message.Message):
    __slots__ = ("process", "health", "last_position", "last_seen_unix_ms", "detail")
    PROCESS_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    LAST_POSITION_FIELD_NUMBER: _ClassVar[int]
    LAST_SEEN_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    process: str
    health: ProcessHealth
    last_position: Position
    last_seen_unix_ms: int
    detail: str
    def __init__(self, process: _Optional[str] = ..., health: _Optional[_Union[ProcessHealth, str]] = ..., last_position: _Optional[_Union[Position, _Mapping]] = ..., last_seen_unix_ms: _Optional[int] = ..., detail: _Optional[str] = ...) -> None: ...

class ArmedTrigger(_message.Message):
    __slots__ = ("trigger", "scope", "signature", "armed_unix_ms")
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    ARMED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    trigger: str
    scope: str
    signature: str
    armed_unix_ms: int
    def __init__(self, trigger: _Optional[str] = ..., scope: _Optional[str] = ..., signature: _Optional[str] = ..., armed_unix_ms: _Optional[int] = ...) -> None: ...

class SupervisorStatus(_message.Message):
    __slots__ = ("project", "epoch", "spec_loaded_unix_ms", "processes", "armed", "recycles_in_window", "breaker_tripped", "buffered_records", "engine_connected", "engine_session", "last_event_seq", "last_tick_unix_ms", "open_backlog_items")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    SPEC_LOADED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESSES_FIELD_NUMBER: _ClassVar[int]
    ARMED_FIELD_NUMBER: _ClassVar[int]
    RECYCLES_IN_WINDOW_FIELD_NUMBER: _ClassVar[int]
    BREAKER_TRIPPED_FIELD_NUMBER: _ClassVar[int]
    BUFFERED_RECORDS_FIELD_NUMBER: _ClassVar[int]
    ENGINE_CONNECTED_FIELD_NUMBER: _ClassVar[int]
    ENGINE_SESSION_FIELD_NUMBER: _ClassVar[int]
    LAST_EVENT_SEQ_FIELD_NUMBER: _ClassVar[int]
    LAST_TICK_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    OPEN_BACKLOG_ITEMS_FIELD_NUMBER: _ClassVar[int]
    project: str
    epoch: str
    spec_loaded_unix_ms: int
    processes: _containers.RepeatedCompositeFieldContainer[ProcessStatus]
    armed: _containers.RepeatedCompositeFieldContainer[ArmedTrigger]
    recycles_in_window: int
    breaker_tripped: bool
    buffered_records: int
    engine_connected: bool
    engine_session: int
    last_event_seq: int
    last_tick_unix_ms: int
    open_backlog_items: int
    def __init__(self, project: _Optional[str] = ..., epoch: _Optional[str] = ..., spec_loaded_unix_ms: _Optional[int] = ..., processes: _Optional[_Iterable[_Union[ProcessStatus, _Mapping]]] = ..., armed: _Optional[_Iterable[_Union[ArmedTrigger, _Mapping]]] = ..., recycles_in_window: _Optional[int] = ..., breaker_tripped: bool = ..., buffered_records: _Optional[int] = ..., engine_connected: bool = ..., engine_session: _Optional[int] = ..., last_event_seq: _Optional[int] = ..., last_tick_unix_ms: _Optional[int] = ..., open_backlog_items: _Optional[int] = ...) -> None: ...

class WatchEscalationsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Escalation(_message.Message):
    __slots__ = ("trigger", "scope", "detail", "position", "snapshot_markdown", "at_unix_ms", "signature", "report_only", "backlog_slot")
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_MARKDOWN_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    REPORT_ONLY_FIELD_NUMBER: _ClassVar[int]
    BACKLOG_SLOT_FIELD_NUMBER: _ClassVar[int]
    trigger: str
    scope: str
    detail: str
    position: Position
    snapshot_markdown: str
    at_unix_ms: int
    signature: str
    report_only: bool
    backlog_slot: int
    def __init__(self, trigger: _Optional[str] = ..., scope: _Optional[str] = ..., detail: _Optional[str] = ..., position: _Optional[_Union[Position, _Mapping]] = ..., snapshot_markdown: _Optional[str] = ..., at_unix_ms: _Optional[int] = ..., signature: _Optional[str] = ..., report_only: bool = ..., backlog_slot: _Optional[int] = ...) -> None: ...

class LoadSpecRequest(_message.Message):
    __slots__ = ("supervisor", "textproto")
    SUPERVISOR_FIELD_NUMBER: _ClassVar[int]
    TEXTPROTO_FIELD_NUMBER: _ClassVar[int]
    supervisor: bytes
    textproto: bool
    def __init__(self, supervisor: _Optional[bytes] = ..., textproto: bool = ...) -> None: ...

class BacklogRequest(_message.Message):
    __slots__ = ("project", "workflow", "as_of_unix_ms", "include_ok")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_FIELD_NUMBER: _ClassVar[int]
    AS_OF_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_OK_FIELD_NUMBER: _ClassVar[int]
    project: str
    workflow: str
    as_of_unix_ms: int
    include_ok: bool
    def __init__(self, project: _Optional[str] = ..., workflow: _Optional[str] = ..., as_of_unix_ms: _Optional[int] = ..., include_ok: bool = ...) -> None: ...

class BacklogSlot(_message.Message):
    __slots__ = ("slot", "state", "window_start_unix_ms", "filled_unix_ms", "evidence_json")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    WINDOW_START_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    FILLED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_JSON_FIELD_NUMBER: _ClassVar[int]
    slot: int
    state: BacklogState
    window_start_unix_ms: int
    filled_unix_ms: int
    evidence_json: str
    def __init__(self, slot: _Optional[int] = ..., state: _Optional[_Union[BacklogState, str]] = ..., window_start_unix_ms: _Optional[int] = ..., filled_unix_ms: _Optional[int] = ..., evidence_json: _Optional[str] = ...) -> None: ...

class BacklogRow(_message.Message):
    __slots__ = ("workflow", "category", "horizon_slot", "slots", "escalation_level", "channel", "item_key", "first_miss_unix_ms", "horizon_unix_ms", "surfaced_unix_ms", "resolved_unix_ms")
    WORKFLOW_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    HORIZON_SLOT_FIELD_NUMBER: _ClassVar[int]
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    ESCALATION_LEVEL_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    ITEM_KEY_FIELD_NUMBER: _ClassVar[int]
    FIRST_MISS_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    HORIZON_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SURFACED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    workflow: str
    category: ExpectationCategory
    horizon_slot: int
    slots: _containers.RepeatedCompositeFieldContainer[BacklogSlot]
    escalation_level: int
    channel: Channel
    item_key: str
    first_miss_unix_ms: int
    horizon_unix_ms: int
    surfaced_unix_ms: int
    resolved_unix_ms: int
    def __init__(self, workflow: _Optional[str] = ..., category: _Optional[_Union[ExpectationCategory, str]] = ..., horizon_slot: _Optional[int] = ..., slots: _Optional[_Iterable[_Union[BacklogSlot, _Mapping]]] = ..., escalation_level: _Optional[int] = ..., channel: _Optional[_Union[Channel, str]] = ..., item_key: _Optional[str] = ..., first_miss_unix_ms: _Optional[int] = ..., horizon_unix_ms: _Optional[int] = ..., surfaced_unix_ms: _Optional[int] = ..., resolved_unix_ms: _Optional[int] = ...) -> None: ...

class BacklogResponse(_message.Message):
    __slots__ = ("project", "as_of_unix_ms", "last_tick_unix_ms", "supervisor_connected", "epoch", "rows")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    AS_OF_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_TICK_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SUPERVISOR_CONNECTED_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    project: str
    as_of_unix_ms: int
    last_tick_unix_ms: int
    supervisor_connected: bool
    epoch: str
    rows: _containers.RepeatedCompositeFieldContainer[BacklogRow]
    def __init__(self, project: _Optional[str] = ..., as_of_unix_ms: _Optional[int] = ..., last_tick_unix_ms: _Optional[int] = ..., supervisor_connected: bool = ..., epoch: _Optional[str] = ..., rows: _Optional[_Iterable[_Union[BacklogRow, _Mapping]]] = ...) -> None: ...

class TickRequest(_message.Message):
    __slots__ = ("window_start_unix_ms", "source", "force")
    WINDOW_START_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    window_start_unix_ms: int
    source: str
    force: bool
    def __init__(self, window_start_unix_ms: _Optional[int] = ..., source: _Optional[str] = ..., force: bool = ...) -> None: ...

class TickResponse(_message.Message):
    __slots__ = ("ran", "window_start_unix_ms", "rows_filled", "note")
    RAN_FIELD_NUMBER: _ClassVar[int]
    WINDOW_START_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FILLED_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    ran: bool
    window_start_unix_ms: int
    rows_filled: int
    note: str
    def __init__(self, ran: bool = ..., window_start_unix_ms: _Optional[int] = ..., rows_filled: _Optional[int] = ..., note: _Optional[str] = ...) -> None: ...

class TaskLifecycle(_message.Message):
    __slots__ = ("task_id", "task_type", "state", "reason", "error", "source", "scheduled_for_unix_ms", "picked_up_unix_ms", "heartbeat_unix_ms", "completed_unix_ms", "workload_id", "evidence")
    class EvidenceEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SCHEDULED_FOR_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    PICKED_UP_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    WORKLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    task_type: str
    state: TaskState
    reason: str
    error: str
    source: str
    scheduled_for_unix_ms: int
    picked_up_unix_ms: int
    heartbeat_unix_ms: int
    completed_unix_ms: int
    workload_id: str
    evidence: _containers.ScalarMap[str, str]
    def __init__(self, task_id: _Optional[int] = ..., task_type: _Optional[str] = ..., state: _Optional[_Union[TaskState, str]] = ..., reason: _Optional[str] = ..., error: _Optional[str] = ..., source: _Optional[str] = ..., scheduled_for_unix_ms: _Optional[int] = ..., picked_up_unix_ms: _Optional[int] = ..., heartbeat_unix_ms: _Optional[int] = ..., completed_unix_ms: _Optional[int] = ..., workload_id: _Optional[str] = ..., evidence: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AdmissionEvent(_message.Message):
    __slots__ = ("workload_id", "kind", "leaf", "phase", "wait_ms", "net_ms", "priority", "holders", "task_id")
    WORKLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LEAF_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    WAIT_MS_FIELD_NUMBER: _ClassVar[int]
    NET_MS_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    HOLDERS_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    workload_id: str
    kind: str
    leaf: str
    phase: AdmissionPhase
    wait_ms: int
    net_ms: int
    priority: int
    holders: _containers.RepeatedScalarFieldContainer[str]
    task_id: int
    def __init__(self, workload_id: _Optional[str] = ..., kind: _Optional[str] = ..., leaf: _Optional[str] = ..., phase: _Optional[_Union[AdmissionPhase, str]] = ..., wait_ms: _Optional[int] = ..., net_ms: _Optional[int] = ..., priority: _Optional[int] = ..., holders: _Optional[_Iterable[str]] = ..., task_id: _Optional[int] = ...) -> None: ...

class GateResult(_message.Message):
    __slots__ = ("name", "verdict", "detail", "tier")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERDICT_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    name: str
    verdict: Verdict
    detail: str
    tier: Tier
    def __init__(self, name: _Optional[str] = ..., verdict: _Optional[_Union[Verdict, str]] = ..., detail: _Optional[str] = ..., tier: _Optional[_Union[Tier, str]] = ...) -> None: ...

class ObjectiveVerdict(_message.Message):
    __slots__ = ("objective", "run_id", "verdict", "gates_passed", "gates_total", "judge_rendered", "awaits", "started_unix_ms", "completed_unix_ms", "gates")
    OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    VERDICT_FIELD_NUMBER: _ClassVar[int]
    GATES_PASSED_FIELD_NUMBER: _ClassVar[int]
    GATES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    JUDGE_RENDERED_FIELD_NUMBER: _ClassVar[int]
    AWAITS_FIELD_NUMBER: _ClassVar[int]
    STARTED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    GATES_FIELD_NUMBER: _ClassVar[int]
    objective: str
    run_id: str
    verdict: Verdict
    gates_passed: int
    gates_total: int
    judge_rendered: bool
    awaits: str
    started_unix_ms: int
    completed_unix_ms: int
    gates: _containers.RepeatedCompositeFieldContainer[GateResult]
    def __init__(self, objective: _Optional[str] = ..., run_id: _Optional[str] = ..., verdict: _Optional[_Union[Verdict, str]] = ..., gates_passed: _Optional[int] = ..., gates_total: _Optional[int] = ..., judge_rendered: bool = ..., awaits: _Optional[str] = ..., started_unix_ms: _Optional[int] = ..., completed_unix_ms: _Optional[int] = ..., gates: _Optional[_Iterable[_Union[GateResult, _Mapping]]] = ...) -> None: ...

class IncidentEvent(_message.Message):
    __slots__ = ("fingerprint", "state", "tier", "endpoint", "failure_mode", "sequence_id", "sequence_num", "event_type")
    FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_MODE_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUM_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    fingerprint: str
    state: IncidentState
    tier: int
    endpoint: str
    failure_mode: str
    sequence_id: str
    sequence_num: int
    event_type: str
    def __init__(self, fingerprint: _Optional[str] = ..., state: _Optional[_Union[IncidentState, str]] = ..., tier: _Optional[int] = ..., endpoint: _Optional[str] = ..., failure_mode: _Optional[str] = ..., sequence_id: _Optional[str] = ..., sequence_num: _Optional[int] = ..., event_type: _Optional[str] = ...) -> None: ...

class ServingEvent(_message.Message):
    __slots__ = ("phase", "generation", "capability", "alias", "model", "actual", "warmup_seconds", "settled_unix_ms")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_FIELD_NUMBER: _ClassVar[int]
    WARMUP_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SETTLED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    phase: ServingPhase
    generation: int
    capability: str
    alias: str
    model: str
    actual: ServingStatus
    warmup_seconds: int
    settled_unix_ms: int
    def __init__(self, phase: _Optional[_Union[ServingPhase, str]] = ..., generation: _Optional[int] = ..., capability: _Optional[str] = ..., alias: _Optional[str] = ..., model: _Optional[str] = ..., actual: _Optional[_Union[ServingStatus, str]] = ..., warmup_seconds: _Optional[int] = ..., settled_unix_ms: _Optional[int] = ...) -> None: ...

class EngineHello(_message.Message):
    __slots__ = ("project", "engine_build", "spec_version", "spec_sha256", "boot_unix_ms", "session", "engine_grpc", "capabilities")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    ENGINE_BUILD_FIELD_NUMBER: _ClassVar[int]
    SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    SPEC_SHA256_FIELD_NUMBER: _ClassVar[int]
    BOOT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SESSION_FIELD_NUMBER: _ClassVar[int]
    ENGINE_GRPC_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    project: str
    engine_build: str
    spec_version: str
    spec_sha256: str
    boot_unix_ms: int
    session: int
    engine_grpc: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, project: _Optional[str] = ..., engine_build: _Optional[str] = ..., spec_version: _Optional[str] = ..., spec_sha256: _Optional[str] = ..., boot_unix_ms: _Optional[int] = ..., session: _Optional[int] = ..., engine_grpc: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ...) -> None: ...

class EngineHeartbeat(_message.Message):
    __slots__ = ("at_unix_ms", "last_seq", "queued_tasks", "claimed_tasks", "oldest_silence_ms", "daemons_running", "daemons_total", "supervisor_acked_seq")
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_SEQ_FIELD_NUMBER: _ClassVar[int]
    QUEUED_TASKS_FIELD_NUMBER: _ClassVar[int]
    CLAIMED_TASKS_FIELD_NUMBER: _ClassVar[int]
    OLDEST_SILENCE_MS_FIELD_NUMBER: _ClassVar[int]
    DAEMONS_RUNNING_FIELD_NUMBER: _ClassVar[int]
    DAEMONS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    SUPERVISOR_ACKED_SEQ_FIELD_NUMBER: _ClassVar[int]
    at_unix_ms: int
    last_seq: int
    queued_tasks: int
    claimed_tasks: int
    oldest_silence_ms: int
    daemons_running: int
    daemons_total: int
    supervisor_acked_seq: int
    def __init__(self, at_unix_ms: _Optional[int] = ..., last_seq: _Optional[int] = ..., queued_tasks: _Optional[int] = ..., claimed_tasks: _Optional[int] = ..., oldest_silence_ms: _Optional[int] = ..., daemons_running: _Optional[int] = ..., daemons_total: _Optional[int] = ..., supervisor_acked_seq: _Optional[int] = ...) -> None: ...

class DirectiveResult(_message.Message):
    __slots__ = ("directive_id", "accepted", "applied", "note", "forecast_id", "judge_status", "at_unix_ms")
    DIRECTIVE_ID_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    APPLIED_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    FORECAST_ID_FIELD_NUMBER: _ClassVar[int]
    JUDGE_STATUS_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    directive_id: str
    accepted: bool
    applied: bool
    note: str
    forecast_id: str
    judge_status: str
    at_unix_ms: int
    def __init__(self, directive_id: _Optional[str] = ..., accepted: bool = ..., applied: bool = ..., note: _Optional[str] = ..., forecast_id: _Optional[str] = ..., judge_status: _Optional[str] = ..., at_unix_ms: _Optional[int] = ...) -> None: ...

class ReplayComplete(_message.Message):
    __slots__ = ("since_unix_ms", "through_seq", "events", "tables", "clamped")
    SINCE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    THROUGH_SEQ_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    TABLES_FIELD_NUMBER: _ClassVar[int]
    CLAMPED_FIELD_NUMBER: _ClassVar[int]
    since_unix_ms: int
    through_seq: int
    events: int
    tables: _containers.RepeatedScalarFieldContainer[str]
    clamped: bool
    def __init__(self, since_unix_ms: _Optional[int] = ..., through_seq: _Optional[int] = ..., events: _Optional[int] = ..., tables: _Optional[_Iterable[str]] = ..., clamped: bool = ...) -> None: ...

class Goodbye(_message.Message):
    __slots__ = ("reason", "note")
    REASON_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    reason: GoodbyeReason
    note: str
    def __init__(self, reason: _Optional[_Union[GoodbyeReason, str]] = ..., note: _Optional[str] = ...) -> None: ...

class ActivityEvent(_message.Message):
    __slots__ = ("activity_id", "kind", "peer", "owner", "dag_id", "run_id", "state", "declared_ns", "horizon_ns", "ended_ns", "precludes", "postures", "reason", "transition", "ceded")
    class PosturesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ACTIVITY_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PEER_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    DAG_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    DECLARED_NS_FIELD_NUMBER: _ClassVar[int]
    HORIZON_NS_FIELD_NUMBER: _ClassVar[int]
    ENDED_NS_FIELD_NUMBER: _ClassVar[int]
    PRECLUDES_FIELD_NUMBER: _ClassVar[int]
    POSTURES_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_FIELD_NUMBER: _ClassVar[int]
    CEDED_FIELD_NUMBER: _ClassVar[int]
    activity_id: str
    kind: str
    peer: str
    owner: str
    dag_id: str
    run_id: str
    state: str
    declared_ns: int
    horizon_ns: int
    ended_ns: int
    precludes: _containers.RepeatedScalarFieldContainer[str]
    postures: _containers.ScalarMap[str, str]
    reason: str
    transition: str
    ceded: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, activity_id: _Optional[str] = ..., kind: _Optional[str] = ..., peer: _Optional[str] = ..., owner: _Optional[str] = ..., dag_id: _Optional[str] = ..., run_id: _Optional[str] = ..., state: _Optional[str] = ..., declared_ns: _Optional[int] = ..., horizon_ns: _Optional[int] = ..., ended_ns: _Optional[int] = ..., precludes: _Optional[_Iterable[str]] = ..., postures: _Optional[_Mapping[str, str]] = ..., reason: _Optional[str] = ..., transition: _Optional[str] = ..., ceded: _Optional[_Iterable[str]] = ...) -> None: ...

class EngineEvent(_message.Message):
    __slots__ = ("seq", "at_unix_ms", "epoch", "replayed", "hello", "heartbeat", "task", "admission", "position", "objective", "incident", "serving", "directive_result", "replay_complete", "goodbye", "activity")
    SEQ_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    REPLAYED_FIELD_NUMBER: _ClassVar[int]
    HELLO_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
    INCIDENT_FIELD_NUMBER: _ClassVar[int]
    SERVING_FIELD_NUMBER: _ClassVar[int]
    DIRECTIVE_RESULT_FIELD_NUMBER: _ClassVar[int]
    REPLAY_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    GOODBYE_FIELD_NUMBER: _ClassVar[int]
    ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    seq: int
    at_unix_ms: int
    epoch: str
    replayed: bool
    hello: EngineHello
    heartbeat: EngineHeartbeat
    task: TaskLifecycle
    admission: AdmissionEvent
    position: PositionReport
    objective: ObjectiveVerdict
    incident: IncidentEvent
    serving: ServingEvent
    directive_result: DirectiveResult
    replay_complete: ReplayComplete
    goodbye: Goodbye
    activity: ActivityEvent
    def __init__(self, seq: _Optional[int] = ..., at_unix_ms: _Optional[int] = ..., epoch: _Optional[str] = ..., replayed: bool = ..., hello: _Optional[_Union[EngineHello, _Mapping]] = ..., heartbeat: _Optional[_Union[EngineHeartbeat, _Mapping]] = ..., task: _Optional[_Union[TaskLifecycle, _Mapping]] = ..., admission: _Optional[_Union[AdmissionEvent, _Mapping]] = ..., position: _Optional[_Union[PositionReport, _Mapping]] = ..., objective: _Optional[_Union[ObjectiveVerdict, _Mapping]] = ..., incident: _Optional[_Union[IncidentEvent, _Mapping]] = ..., serving: _Optional[_Union[ServingEvent, _Mapping]] = ..., directive_result: _Optional[_Union[DirectiveResult, _Mapping]] = ..., replay_complete: _Optional[_Union[ReplayComplete, _Mapping]] = ..., goodbye: _Optional[_Union[Goodbye, _Mapping]] = ..., activity: _Optional[_Union[ActivityEvent, _Mapping]] = ...) -> None: ...

class Subscribe(_message.Message):
    __slots__ = ("supervisor_id", "project", "spec_version", "spec_sha256", "since_unix_ms", "resume_session", "resume_seq", "capabilities")
    SUPERVISOR_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    SPEC_SHA256_FIELD_NUMBER: _ClassVar[int]
    SINCE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    RESUME_SESSION_FIELD_NUMBER: _ClassVar[int]
    RESUME_SEQ_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    supervisor_id: str
    project: str
    spec_version: str
    spec_sha256: str
    since_unix_ms: int
    resume_session: int
    resume_seq: int
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, supervisor_id: _Optional[str] = ..., project: _Optional[str] = ..., spec_version: _Optional[str] = ..., spec_sha256: _Optional[str] = ..., since_unix_ms: _Optional[int] = ..., resume_session: _Optional[int] = ..., resume_seq: _Optional[int] = ..., capabilities: _Optional[_Iterable[str]] = ...) -> None: ...

class SupervisorAck(_message.Message):
    __slots__ = ("through_seq",)
    THROUGH_SEQ_FIELD_NUMBER: _ClassVar[int]
    through_seq: int
    def __init__(self, through_seq: _Optional[int] = ...) -> None: ...

class SupervisorHeartbeat(_message.Message):
    __slots__ = ("at_unix_ms", "buffered_records", "last_tick_unix_ms")
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    BUFFERED_RECORDS_FIELD_NUMBER: _ClassVar[int]
    LAST_TICK_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    at_unix_ms: int
    buffered_records: int
    last_tick_unix_ms: int
    def __init__(self, at_unix_ms: _Optional[int] = ..., buffered_records: _Optional[int] = ..., last_tick_unix_ms: _Optional[int] = ...) -> None: ...

class ReclaimOrphan(_message.Message):
    __slots__ = ("task_id", "task_type", "silence_ms", "net_ms", "position", "evidence_json", "dry_run")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_TYPE_FIELD_NUMBER: _ClassVar[int]
    SILENCE_MS_FIELD_NUMBER: _ClassVar[int]
    NET_MS_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_JSON_FIELD_NUMBER: _ClassVar[int]
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    task_id: int
    task_type: str
    silence_ms: int
    net_ms: int
    position: Position
    evidence_json: str
    dry_run: bool
    def __init__(self, task_id: _Optional[int] = ..., task_type: _Optional[str] = ..., silence_ms: _Optional[int] = ..., net_ms: _Optional[int] = ..., position: _Optional[_Union[Position, _Mapping]] = ..., evidence_json: _Optional[str] = ..., dry_run: bool = ...) -> None: ...

class BacklogTransition(_message.Message):
    __slots__ = ("workflow", "item_key", "slot", "state", "category", "channel", "window_start_unix_ms", "first_miss_unix_ms", "horizon_unix_ms", "evidence_json", "resolved")
    WORKFLOW_FIELD_NUMBER: _ClassVar[int]
    ITEM_KEY_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    WINDOW_START_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    FIRST_MISS_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    HORIZON_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_JSON_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_FIELD_NUMBER: _ClassVar[int]
    workflow: str
    item_key: str
    slot: int
    state: BacklogState
    category: ExpectationCategory
    channel: Channel
    window_start_unix_ms: int
    first_miss_unix_ms: int
    horizon_unix_ms: int
    evidence_json: str
    resolved: bool
    def __init__(self, workflow: _Optional[str] = ..., item_key: _Optional[str] = ..., slot: _Optional[int] = ..., state: _Optional[_Union[BacklogState, str]] = ..., category: _Optional[_Union[ExpectationCategory, str]] = ..., channel: _Optional[_Union[Channel, str]] = ..., window_start_unix_ms: _Optional[int] = ..., first_miss_unix_ms: _Optional[int] = ..., horizon_unix_ms: _Optional[int] = ..., evidence_json: _Optional[str] = ..., resolved: bool = ...) -> None: ...

class SupervisorMessage(_message.Message):
    __slots__ = ("directive_id", "at_unix_ms", "subscribe", "ack", "heartbeat", "reclaim_orphan", "escalation", "backlog_transition")
    DIRECTIVE_ID_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SUBSCRIBE_FIELD_NUMBER: _ClassVar[int]
    ACK_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    RECLAIM_ORPHAN_FIELD_NUMBER: _ClassVar[int]
    ESCALATION_FIELD_NUMBER: _ClassVar[int]
    BACKLOG_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    directive_id: str
    at_unix_ms: int
    subscribe: Subscribe
    ack: SupervisorAck
    heartbeat: SupervisorHeartbeat
    reclaim_orphan: ReclaimOrphan
    escalation: Escalation
    backlog_transition: BacklogTransition
    def __init__(self, directive_id: _Optional[str] = ..., at_unix_ms: _Optional[int] = ..., subscribe: _Optional[_Union[Subscribe, _Mapping]] = ..., ack: _Optional[_Union[SupervisorAck, _Mapping]] = ..., heartbeat: _Optional[_Union[SupervisorHeartbeat, _Mapping]] = ..., reclaim_orphan: _Optional[_Union[ReclaimOrphan, _Mapping]] = ..., escalation: _Optional[_Union[Escalation, _Mapping]] = ..., backlog_transition: _Optional[_Union[BacklogTransition, _Mapping]] = ...) -> None: ...

class BacklogCell(_message.Message):
    __slots__ = ("project", "workflow", "item_key", "category", "slot", "state", "at_unix_ms", "first_miss_unix_ms", "horizon_slot", "horizon_unix_ms", "escalation_level", "resolved", "evidence_json", "spec_version", "engine_build")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_FIELD_NUMBER: _ClassVar[int]
    ITEM_KEY_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    FIRST_MISS_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    HORIZON_SLOT_FIELD_NUMBER: _ClassVar[int]
    HORIZON_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    ESCALATION_LEVEL_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_JSON_FIELD_NUMBER: _ClassVar[int]
    SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENGINE_BUILD_FIELD_NUMBER: _ClassVar[int]
    project: str
    workflow: str
    item_key: str
    category: ExpectationCategory
    slot: int
    state: BacklogState
    at_unix_ms: int
    first_miss_unix_ms: int
    horizon_slot: int
    horizon_unix_ms: int
    escalation_level: int
    resolved: bool
    evidence_json: str
    spec_version: str
    engine_build: str
    def __init__(self, project: _Optional[str] = ..., workflow: _Optional[str] = ..., item_key: _Optional[str] = ..., category: _Optional[_Union[ExpectationCategory, str]] = ..., slot: _Optional[int] = ..., state: _Optional[_Union[BacklogState, str]] = ..., at_unix_ms: _Optional[int] = ..., first_miss_unix_ms: _Optional[int] = ..., horizon_slot: _Optional[int] = ..., horizon_unix_ms: _Optional[int] = ..., escalation_level: _Optional[int] = ..., resolved: bool = ..., evidence_json: _Optional[str] = ..., spec_version: _Optional[str] = ..., engine_build: _Optional[str] = ...) -> None: ...

class SupervisorEvent(_message.Message):
    __slots__ = ("project", "event_id", "kind", "trigger", "scope", "signature", "observer", "call_site", "proposition", "verdict", "p", "side_effect", "task_class", "task_id", "position", "directive_id", "forecast_id", "outcome_known", "outcome", "tier", "detail", "evidence_json", "at_unix_ms", "spec_version", "engine_build")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    OBSERVER_FIELD_NUMBER: _ClassVar[int]
    CALL_SITE_FIELD_NUMBER: _ClassVar[int]
    PROPOSITION_FIELD_NUMBER: _ClassVar[int]
    VERDICT_FIELD_NUMBER: _ClassVar[int]
    P_FIELD_NUMBER: _ClassVar[int]
    SIDE_EFFECT_FIELD_NUMBER: _ClassVar[int]
    TASK_CLASS_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    DIRECTIVE_ID_FIELD_NUMBER: _ClassVar[int]
    FORECAST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_KNOWN_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_JSON_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENGINE_BUILD_FIELD_NUMBER: _ClassVar[int]
    project: str
    event_id: str
    kind: str
    trigger: str
    scope: str
    signature: str
    observer: str
    call_site: str
    proposition: str
    verdict: Verdict
    p: float
    side_effect: str
    task_class: str
    task_id: int
    position: Position
    directive_id: str
    forecast_id: str
    outcome_known: bool
    outcome: bool
    tier: Tier
    detail: str
    evidence_json: str
    at_unix_ms: int
    spec_version: str
    engine_build: str
    def __init__(self, project: _Optional[str] = ..., event_id: _Optional[str] = ..., kind: _Optional[str] = ..., trigger: _Optional[str] = ..., scope: _Optional[str] = ..., signature: _Optional[str] = ..., observer: _Optional[str] = ..., call_site: _Optional[str] = ..., proposition: _Optional[str] = ..., verdict: _Optional[_Union[Verdict, str]] = ..., p: _Optional[float] = ..., side_effect: _Optional[str] = ..., task_class: _Optional[str] = ..., task_id: _Optional[int] = ..., position: _Optional[_Union[Position, _Mapping]] = ..., directive_id: _Optional[str] = ..., forecast_id: _Optional[str] = ..., outcome_known: bool = ..., outcome: bool = ..., tier: _Optional[_Union[Tier, str]] = ..., detail: _Optional[str] = ..., evidence_json: _Optional[str] = ..., at_unix_ms: _Optional[int] = ..., spec_version: _Optional[str] = ..., engine_build: _Optional[str] = ...) -> None: ...

class PositionRecord(_message.Message):
    __slots__ = ("project", "process", "kind", "machine", "phase", "run_id", "step", "momentum", "source", "health", "silence_s", "detail", "at_unix_ms")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    PROCESS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    MACHINE_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    SILENCE_S_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    project: str
    process: str
    kind: str
    machine: str
    phase: str
    run_id: str
    step: str
    momentum: int
    source: PositionSource
    health: ProcessHealth
    silence_s: int
    detail: str
    at_unix_ms: int
    def __init__(self, project: _Optional[str] = ..., process: _Optional[str] = ..., kind: _Optional[str] = ..., machine: _Optional[str] = ..., phase: _Optional[str] = ..., run_id: _Optional[str] = ..., step: _Optional[str] = ..., momentum: _Optional[int] = ..., source: _Optional[_Union[PositionSource, str]] = ..., health: _Optional[_Union[ProcessHealth, str]] = ..., silence_s: _Optional[int] = ..., detail: _Optional[str] = ..., at_unix_ms: _Optional[int] = ...) -> None: ...

class SupervisorRecord(_message.Message):
    __slots__ = ("project", "backlog_cell", "event", "position")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    BACKLOG_CELL_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    project: str
    backlog_cell: BacklogCell
    event: SupervisorEvent
    position: PositionRecord
    def __init__(self, project: _Optional[str] = ..., backlog_cell: _Optional[_Union[BacklogCell, _Mapping]] = ..., event: _Optional[_Union[SupervisorEvent, _Mapping]] = ..., position: _Optional[_Union[PositionRecord, _Mapping]] = ...) -> None: ...
