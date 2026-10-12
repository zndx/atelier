from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SignalKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SIGNAL_KIND_UNSPECIFIED: _ClassVar[SignalKind]
    EXTERNAL_NAMESPACE_VIOLATION: _ClassVar[SignalKind]
    UNSATISFIABLE: _ClassVar[SignalKind]
    UNGROUNDED: _ClassVar[SignalKind]
    VERSION_DRIFT: _ClassVar[SignalKind]
    TX_ID_NOT_UUIDV7: _ClassVar[SignalKind]

class Disposition(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DISPOSITION_UNSPECIFIED: _ClassVar[Disposition]
    CORRECTED: _ClassVar[Disposition]
    COINED_LOCAL: _ClassVar[Disposition]
    UNRESOLVABLE: _ClassVar[Disposition]

class YieldReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    YIELD_REASON_UNSPECIFIED: _ClassVar[YieldReason]
    YIELD_REASON_PREEMPTED: _ClassVar[YieldReason]
    YIELD_REASON_COMPLETED: _ClassVar[YieldReason]
    YIELD_REASON_ORPHAN: _ClassVar[YieldReason]
    YIELD_REASON_UNIT_STOP: _ClassVar[YieldReason]

class ServerQueryKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVER_QUERY_KIND_UNSPECIFIED: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_REMOTES: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_SCHEDULES: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_PEERS: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_NOTE: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_SURFACES: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_QUEUES: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_WORKLOADS: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_SOURCE_POSTURE: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_PRODUCTS: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_COGNITION: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_CONTRIBUTIONS: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_ACTIVITIES: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_THOUGHTS: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_AGENDA: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_SEARCH: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_AGENTS: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_FMP: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_RESOURCES: _ClassVar[ServerQueryKind]
    SERVER_QUERY_KIND_ASPECTS: _ClassVar[ServerQueryKind]

class ActivityState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVITY_STATE_UNSPECIFIED: _ClassVar[ActivityState]
    ACTIVITY_QUEUED: _ClassVar[ActivityState]
    ACTIVITY_RUNNING: _ClassVar[ActivityState]
    ACTIVITY_RELEASED: _ClassVar[ActivityState]
    ACTIVITY_EXPIRED: _ClassVar[ActivityState]
    ACTIVITY_FAILED: _ClassVar[ActivityState]
    ACTIVITY_SUPERSEDED: _ClassVar[ActivityState]

class ShaclNodeKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SHACL_NODE_KIND_UNSPECIFIED: _ClassVar[ShaclNodeKind]
    SHACL_NODE_KIND_IRI: _ClassVar[ShaclNodeKind]
    SHACL_NODE_KIND_BLANK_NODE: _ClassVar[ShaclNodeKind]
    SHACL_NODE_KIND_LITERAL: _ClassVar[ShaclNodeKind]
    SHACL_NODE_KIND_BLANK_NODE_OR_IRI: _ClassVar[ShaclNodeKind]
    SHACL_NODE_KIND_BLANK_NODE_OR_LITERAL: _ClassVar[ShaclNodeKind]
    SHACL_NODE_KIND_IRI_OR_LITERAL: _ClassVar[ShaclNodeKind]

class ShaclSeverity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SHACL_SEVERITY_UNSPECIFIED: _ClassVar[ShaclSeverity]
    SHACL_SEVERITY_INFO: _ClassVar[ShaclSeverity]
    SHACL_SEVERITY_WARNING: _ClassVar[ShaclSeverity]
    SHACL_SEVERITY_VIOLATION: _ClassVar[ShaclSeverity]

class ServingBackend(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVING_BACKEND_UNSPECIFIED: _ClassVar[ServingBackend]
    SERVING_BACKEND_VLLM_LOCAL: _ClassVar[ServingBackend]
    SERVING_BACKEND_KSERVE_REMOTE: _ClassVar[ServingBackend]
    SERVING_BACKEND_CPU_PROXY: _ClassVar[ServingBackend]

class ResourceClass(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_CLASS_UNSPECIFIED: _ClassVar[ResourceClass]
    RESOURCE_CLASS_HEAVY: _ClassVar[ResourceClass]
    RESOURCE_CLASS_MEDIUM: _ClassVar[ResourceClass]
    RESOURCE_CLASS_LIGHT: _ClassVar[ResourceClass]
    RESOURCE_CLASS_COMPUTE: _ClassVar[ResourceClass]

class WorkloadPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKLOAD_PHASE_UNSPECIFIED: _ClassVar[WorkloadPhase]
    WORKLOAD_PHASE_SETTLED: _ClassVar[WorkloadPhase]
    WORKLOAD_PHASE_TRANSITIONING: _ClassVar[WorkloadPhase]

class WorkloadStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKLOAD_STATUS_UNSPECIFIED: _ClassVar[WorkloadStatus]
    WORKLOAD_STATUS_SERVING: _ClassVar[WorkloadStatus]
    WORKLOAD_STATUS_STARTING: _ClassVar[WorkloadStatus]
    WORKLOAD_STATUS_DEGRADED: _ClassVar[WorkloadStatus]
    WORKLOAD_STATUS_FAILED: _ClassVar[WorkloadStatus]
    WORKLOAD_STATUS_ABSENT: _ClassVar[WorkloadStatus]
SIGNAL_KIND_UNSPECIFIED: SignalKind
EXTERNAL_NAMESPACE_VIOLATION: SignalKind
UNSATISFIABLE: SignalKind
UNGROUNDED: SignalKind
VERSION_DRIFT: SignalKind
TX_ID_NOT_UUIDV7: SignalKind
DISPOSITION_UNSPECIFIED: Disposition
CORRECTED: Disposition
COINED_LOCAL: Disposition
UNRESOLVABLE: Disposition
YIELD_REASON_UNSPECIFIED: YieldReason
YIELD_REASON_PREEMPTED: YieldReason
YIELD_REASON_COMPLETED: YieldReason
YIELD_REASON_ORPHAN: YieldReason
YIELD_REASON_UNIT_STOP: YieldReason
SERVER_QUERY_KIND_UNSPECIFIED: ServerQueryKind
SERVER_QUERY_KIND_REMOTES: ServerQueryKind
SERVER_QUERY_KIND_SCHEDULES: ServerQueryKind
SERVER_QUERY_KIND_PEERS: ServerQueryKind
SERVER_QUERY_KIND_NOTE: ServerQueryKind
SERVER_QUERY_KIND_SURFACES: ServerQueryKind
SERVER_QUERY_KIND_QUEUES: ServerQueryKind
SERVER_QUERY_KIND_WORKLOADS: ServerQueryKind
SERVER_QUERY_KIND_SOURCE_POSTURE: ServerQueryKind
SERVER_QUERY_KIND_PRODUCTS: ServerQueryKind
SERVER_QUERY_KIND_COGNITION: ServerQueryKind
SERVER_QUERY_KIND_CONTRIBUTIONS: ServerQueryKind
SERVER_QUERY_KIND_ACTIVITIES: ServerQueryKind
SERVER_QUERY_KIND_THOUGHTS: ServerQueryKind
SERVER_QUERY_KIND_AGENDA: ServerQueryKind
SERVER_QUERY_KIND_SEARCH: ServerQueryKind
SERVER_QUERY_KIND_AGENTS: ServerQueryKind
SERVER_QUERY_KIND_FMP: ServerQueryKind
SERVER_QUERY_KIND_RESOURCES: ServerQueryKind
SERVER_QUERY_KIND_ASPECTS: ServerQueryKind
ACTIVITY_STATE_UNSPECIFIED: ActivityState
ACTIVITY_QUEUED: ActivityState
ACTIVITY_RUNNING: ActivityState
ACTIVITY_RELEASED: ActivityState
ACTIVITY_EXPIRED: ActivityState
ACTIVITY_FAILED: ActivityState
ACTIVITY_SUPERSEDED: ActivityState
SHACL_NODE_KIND_UNSPECIFIED: ShaclNodeKind
SHACL_NODE_KIND_IRI: ShaclNodeKind
SHACL_NODE_KIND_BLANK_NODE: ShaclNodeKind
SHACL_NODE_KIND_LITERAL: ShaclNodeKind
SHACL_NODE_KIND_BLANK_NODE_OR_IRI: ShaclNodeKind
SHACL_NODE_KIND_BLANK_NODE_OR_LITERAL: ShaclNodeKind
SHACL_NODE_KIND_IRI_OR_LITERAL: ShaclNodeKind
SHACL_SEVERITY_UNSPECIFIED: ShaclSeverity
SHACL_SEVERITY_INFO: ShaclSeverity
SHACL_SEVERITY_WARNING: ShaclSeverity
SHACL_SEVERITY_VIOLATION: ShaclSeverity
SERVING_BACKEND_UNSPECIFIED: ServingBackend
SERVING_BACKEND_VLLM_LOCAL: ServingBackend
SERVING_BACKEND_KSERVE_REMOTE: ServingBackend
SERVING_BACKEND_CPU_PROXY: ServingBackend
RESOURCE_CLASS_UNSPECIFIED: ResourceClass
RESOURCE_CLASS_HEAVY: ResourceClass
RESOURCE_CLASS_MEDIUM: ResourceClass
RESOURCE_CLASS_LIGHT: ResourceClass
RESOURCE_CLASS_COMPUTE: ResourceClass
WORKLOAD_PHASE_UNSPECIFIED: WorkloadPhase
WORKLOAD_PHASE_SETTLED: WorkloadPhase
WORKLOAD_PHASE_TRANSITIONING: WorkloadPhase
WORKLOAD_STATUS_UNSPECIFIED: WorkloadStatus
WORKLOAD_STATUS_SERVING: WorkloadStatus
WORKLOAD_STATUS_STARTING: WorkloadStatus
WORKLOAD_STATUS_DEGRADED: WorkloadStatus
WORKLOAD_STATUS_FAILED: WorkloadStatus
WORKLOAD_STATUS_ABSENT: WorkloadStatus

class Candidate(_message.Message):
    __slots__ = ("iri", "label", "kind", "score")
    IRI_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    iri: str
    label: str
    kind: str
    score: float
    def __init__(self, iri: _Optional[str] = ..., label: _Optional[str] = ..., kind: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class BoundarySignal(_message.Message):
    __slots__ = ("kind", "subject", "offending", "reason", "authority")
    KIND_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    OFFENDING_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    AUTHORITY_FIELD_NUMBER: _ClassVar[int]
    kind: SignalKind
    subject: str
    offending: str
    reason: str
    authority: str
    def __init__(self, kind: _Optional[_Union[SignalKind, str]] = ..., subject: _Optional[str] = ..., offending: _Optional[str] = ..., reason: _Optional[str] = ..., authority: _Optional[str] = ...) -> None: ...

class SignalContext(_message.Message):
    __slots__ = ("candidates", "justification", "anchors", "rules")
    CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    JUSTIFICATION_FIELD_NUMBER: _ClassVar[int]
    ANCHORS_FIELD_NUMBER: _ClassVar[int]
    RULES_FIELD_NUMBER: _ClassVar[int]
    candidates: _containers.RepeatedCompositeFieldContainer[Candidate]
    justification: _containers.RepeatedScalarFieldContainer[str]
    anchors: _containers.RepeatedCompositeFieldContainer[Candidate]
    rules: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, candidates: _Optional[_Iterable[_Union[Candidate, _Mapping]]] = ..., justification: _Optional[_Iterable[str]] = ..., anchors: _Optional[_Iterable[_Union[Candidate, _Mapping]]] = ..., rules: _Optional[_Iterable[str]] = ...) -> None: ...

class RemediationRequest(_message.Message):
    __slots__ = ("capability", "signal", "context", "max_tokens", "temperature")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    capability: str
    signal: BoundarySignal
    context: SignalContext
    max_tokens: int
    temperature: float
    def __init__(self, capability: _Optional[str] = ..., signal: _Optional[_Union[BoundarySignal, _Mapping]] = ..., context: _Optional[_Union[SignalContext, _Mapping]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ...) -> None: ...

class RemediationResponse(_message.Message):
    __slots__ = ("correction", "disposition", "rationale", "model", "reasoning_content", "completion_tokens", "latency_ms")
    CORRECTION_FIELD_NUMBER: _ClassVar[int]
    DISPOSITION_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    REASONING_CONTENT_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    correction: str
    disposition: Disposition
    rationale: str
    model: str
    reasoning_content: str
    completion_tokens: int
    latency_ms: float
    def __init__(self, correction: _Optional[str] = ..., disposition: _Optional[_Union[Disposition, str]] = ..., rationale: _Optional[str] = ..., model: _Optional[str] = ..., reasoning_content: _Optional[str] = ..., completion_tokens: _Optional[int] = ..., latency_ms: _Optional[float] = ...) -> None: ...

class CompleteRequest(_message.Message):
    __slots__ = ("capability", "prompt", "system_prompt", "max_tokens", "temperature", "json_schema", "timezone", "clock_json", "tools_json", "tool_choice", "messages_json", "capabilities")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    CLOCK_JSON_FIELD_NUMBER: _ClassVar[int]
    TOOLS_JSON_FIELD_NUMBER: _ClassVar[int]
    TOOL_CHOICE_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_JSON_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    capability: str
    prompt: str
    system_prompt: str
    max_tokens: int
    temperature: float
    json_schema: str
    timezone: str
    clock_json: str
    tools_json: str
    tool_choice: str
    messages_json: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, capability: _Optional[str] = ..., prompt: _Optional[str] = ..., system_prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., json_schema: _Optional[str] = ..., timezone: _Optional[str] = ..., clock_json: _Optional[str] = ..., tools_json: _Optional[str] = ..., tool_choice: _Optional[str] = ..., messages_json: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ...) -> None: ...

class ToolCall(_message.Message):
    __slots__ = ("id", "name", "arguments_json")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_JSON_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    arguments_json: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., arguments_json: _Optional[str] = ...) -> None: ...

class ReasoningLayer(_message.Message):
    __slots__ = ("layer", "producer", "text", "tokens")
    LAYER_FIELD_NUMBER: _ClassVar[int]
    PRODUCER_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    layer: str
    producer: str
    text: str
    tokens: int
    def __init__(self, layer: _Optional[str] = ..., producer: _Optional[str] = ..., text: _Optional[str] = ..., tokens: _Optional[int] = ...) -> None: ...

class CompleteResponse(_message.Message):
    __slots__ = ("text", "model", "prompt_tokens", "completion_tokens", "latency_ms", "reasoning_content", "finish_reason", "tool_calls", "reasoning", "fulfilled_by", "profile")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    REASONING_CONTENT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    FULFILLED_BY_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    reasoning_content: str
    finish_reason: str
    tool_calls: _containers.RepeatedCompositeFieldContainer[ToolCall]
    reasoning: _containers.RepeatedCompositeFieldContainer[ReasoningLayer]
    fulfilled_by: str
    profile: OperatingProfile
    def __init__(self, text: _Optional[str] = ..., model: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., latency_ms: _Optional[float] = ..., reasoning_content: _Optional[str] = ..., finish_reason: _Optional[str] = ..., tool_calls: _Optional[_Iterable[_Union[ToolCall, _Mapping]]] = ..., reasoning: _Optional[_Iterable[_Union[ReasoningLayer, _Mapping]]] = ..., fulfilled_by: _Optional[str] = ..., profile: _Optional[_Union[OperatingProfile, _Mapping]] = ...) -> None: ...

class DecisionQuestion(_message.Message):
    __slots__ = ("id", "type", "instructions", "criteria_json")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    CRITERIA_JSON_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    instructions: str
    criteria_json: str
    def __init__(self, id: _Optional[str] = ..., type: _Optional[str] = ..., instructions: _Optional[str] = ..., criteria_json: _Optional[str] = ...) -> None: ...

class DecideRequest(_message.Message):
    __slots__ = ("capability", "state_json", "questions", "instructions", "cache_salt")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    STATE_JSON_FIELD_NUMBER: _ClassVar[int]
    QUESTIONS_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    CACHE_SALT_FIELD_NUMBER: _ClassVar[int]
    capability: str
    state_json: str
    questions: _containers.RepeatedCompositeFieldContainer[DecisionQuestion]
    instructions: str
    cache_salt: str
    def __init__(self, capability: _Optional[str] = ..., state_json: _Optional[str] = ..., questions: _Optional[_Iterable[_Union[DecisionQuestion, _Mapping]]] = ..., instructions: _Optional[str] = ..., cache_salt: _Optional[str] = ...) -> None: ...

class DecideResponse(_message.Message):
    __slots__ = ("id", "model", "answers_json", "diagnostics_json", "input_tokens", "output_tokens", "latency_ms", "fulfilled_by")
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ANSWERS_JSON_FIELD_NUMBER: _ClassVar[int]
    DIAGNOSTICS_JSON_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    FULFILLED_BY_FIELD_NUMBER: _ClassVar[int]
    id: str
    model: str
    answers_json: str
    diagnostics_json: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    fulfilled_by: str
    def __init__(self, id: _Optional[str] = ..., model: _Optional[str] = ..., answers_json: _Optional[str] = ..., diagnostics_json: _Optional[str] = ..., input_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., latency_ms: _Optional[float] = ..., fulfilled_by: _Optional[str] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Endpoint(_message.Message):
    __slots__ = ("capability", "model", "healthy", "gpu_ids", "detail")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    GPU_IDS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    capability: str
    model: str
    healthy: bool
    gpu_ids: _containers.RepeatedScalarFieldContainer[int]
    detail: str
    def __init__(self, capability: _Optional[str] = ..., model: _Optional[str] = ..., healthy: bool = ..., gpu_ids: _Optional[_Iterable[int]] = ..., detail: _Optional[str] = ...) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("project", "endpoints", "total_gpus", "surfaces")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_GPUS_FIELD_NUMBER: _ClassVar[int]
    SURFACES_FIELD_NUMBER: _ClassVar[int]
    project: str
    endpoints: _containers.RepeatedCompositeFieldContainer[Endpoint]
    total_gpus: int
    surfaces: _containers.RepeatedCompositeFieldContainer[Surface]
    def __init__(self, project: _Optional[str] = ..., endpoints: _Optional[_Iterable[_Union[Endpoint, _Mapping]]] = ..., total_gpus: _Optional[int] = ..., surfaces: _Optional[_Iterable[_Union[Surface, _Mapping]]] = ...) -> None: ...

class Surface(_message.Message):
    __slots__ = ("kind", "url", "healthy")
    KIND_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    kind: str
    url: str
    healthy: bool
    def __init__(self, kind: _Optional[str] = ..., url: _Optional[str] = ..., healthy: bool = ...) -> None: ...

class YieldRequest(_message.Message):
    __slots__ = ("workload_id", "reason", "sentinel_id", "detail")
    WORKLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    SENTINEL_ID_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    workload_id: str
    reason: YieldReason
    sentinel_id: str
    detail: str
    def __init__(self, workload_id: _Optional[str] = ..., reason: _Optional[_Union[YieldReason, str]] = ..., sentinel_id: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class YieldResponse(_message.Message):
    __slots__ = ("ok", "process_ended", "restore_started", "message")
    OK_FIELD_NUMBER: _ClassVar[int]
    PROCESS_ENDED_FIELD_NUMBER: _ClassVar[int]
    RESTORE_STARTED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    process_ended: bool
    restore_started: bool
    message: str
    def __init__(self, ok: bool = ..., process_ended: bool = ..., restore_started: bool = ..., message: _Optional[str] = ...) -> None: ...

class GitRemote(_message.Message):
    __slots__ = ("name", "url")
    NAME_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    name: str
    url: str
    def __init__(self, name: _Optional[str] = ..., url: _Optional[str] = ...) -> None: ...

class PeerHint(_message.Message):
    __slots__ = ("project", "target")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    project: str
    target: str
    def __init__(self, project: _Optional[str] = ..., target: _Optional[str] = ...) -> None: ...

class ScheduleHint(_message.Message):
    __slots__ = ("id", "cron", "airflow_dag_id", "source", "enabled", "kind", "claims", "precludes", "postures", "horizon_s", "after", "timezone", "description", "runner", "after_mode")
    class PosturesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    CRON_FIELD_NUMBER: _ClassVar[int]
    AIRFLOW_DAG_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    CLAIMS_FIELD_NUMBER: _ClassVar[int]
    PRECLUDES_FIELD_NUMBER: _ClassVar[int]
    POSTURES_FIELD_NUMBER: _ClassVar[int]
    HORIZON_S_FIELD_NUMBER: _ClassVar[int]
    AFTER_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    RUNNER_FIELD_NUMBER: _ClassVar[int]
    AFTER_MODE_FIELD_NUMBER: _ClassVar[int]
    id: str
    cron: str
    airflow_dag_id: str
    source: str
    enabled: bool
    kind: str
    claims: _containers.RepeatedCompositeFieldContainer[ActivityClaim]
    precludes: _containers.RepeatedScalarFieldContainer[str]
    postures: _containers.ScalarMap[str, str]
    horizon_s: int
    after: _containers.RepeatedScalarFieldContainer[str]
    timezone: str
    description: str
    runner: str
    after_mode: str
    def __init__(self, id: _Optional[str] = ..., cron: _Optional[str] = ..., airflow_dag_id: _Optional[str] = ..., source: _Optional[str] = ..., enabled: bool = ..., kind: _Optional[str] = ..., claims: _Optional[_Iterable[_Union[ActivityClaim, _Mapping]]] = ..., precludes: _Optional[_Iterable[str]] = ..., postures: _Optional[_Mapping[str, str]] = ..., horizon_s: _Optional[int] = ..., after: _Optional[_Iterable[str]] = ..., timezone: _Optional[str] = ..., description: _Optional[str] = ..., runner: _Optional[str] = ..., after_mode: _Optional[str] = ...) -> None: ...

class WikiNote(_message.Message):
    __slots__ = ("id", "title", "body", "links", "origin_project")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_PROJECT_FIELD_NUMBER: _ClassVar[int]
    id: str
    title: str
    body: str
    links: _containers.RepeatedScalarFieldContainer[str]
    origin_project: str
    def __init__(self, id: _Optional[str] = ..., title: _Optional[str] = ..., body: _Optional[str] = ..., links: _Optional[_Iterable[str]] = ..., origin_project: _Optional[str] = ...) -> None: ...

class SubmodulePosture(_message.Message):
    __slots__ = ("path", "name", "pinned_sha", "checked_out_sha", "dirty")
    PATH_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PINNED_SHA_FIELD_NUMBER: _ClassVar[int]
    CHECKED_OUT_SHA_FIELD_NUMBER: _ClassVar[int]
    DIRTY_FIELD_NUMBER: _ClassVar[int]
    path: str
    name: str
    pinned_sha: str
    checked_out_sha: str
    dirty: bool
    def __init__(self, path: _Optional[str] = ..., name: _Optional[str] = ..., pinned_sha: _Optional[str] = ..., checked_out_sha: _Optional[str] = ..., dirty: bool = ...) -> None: ...

class MigrationPosture(_message.Message):
    __slots__ = ("source", "current", "unapplied")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_FIELD_NUMBER: _ClassVar[int]
    UNAPPLIED_FIELD_NUMBER: _ClassVar[int]
    source: str
    current: str
    unapplied: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, source: _Optional[str] = ..., current: _Optional[str] = ..., unapplied: _Optional[_Iterable[str]] = ...) -> None: ...

class SourcePosture(_message.Message):
    __slots__ = ("project", "checkout", "branch", "head", "running_sha", "dirty", "upstream", "ahead", "behind", "submodules", "migrations")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    CHECKOUT_FIELD_NUMBER: _ClassVar[int]
    BRANCH_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    RUNNING_SHA_FIELD_NUMBER: _ClassVar[int]
    DIRTY_FIELD_NUMBER: _ClassVar[int]
    UPSTREAM_FIELD_NUMBER: _ClassVar[int]
    AHEAD_FIELD_NUMBER: _ClassVar[int]
    BEHIND_FIELD_NUMBER: _ClassVar[int]
    SUBMODULES_FIELD_NUMBER: _ClassVar[int]
    MIGRATIONS_FIELD_NUMBER: _ClassVar[int]
    project: str
    checkout: str
    branch: str
    head: str
    running_sha: str
    dirty: bool
    upstream: str
    ahead: int
    behind: int
    submodules: _containers.RepeatedCompositeFieldContainer[SubmodulePosture]
    migrations: _containers.RepeatedCompositeFieldContainer[MigrationPosture]
    def __init__(self, project: _Optional[str] = ..., checkout: _Optional[str] = ..., branch: _Optional[str] = ..., head: _Optional[str] = ..., running_sha: _Optional[str] = ..., dirty: bool = ..., upstream: _Optional[str] = ..., ahead: _Optional[int] = ..., behind: _Optional[int] = ..., submodules: _Optional[_Iterable[_Union[SubmodulePosture, _Mapping]]] = ..., migrations: _Optional[_Iterable[_Union[MigrationPosture, _Mapping]]] = ...) -> None: ...

class ServerQueryRequest(_message.Message):
    __slots__ = ("kind", "ttl", "nonce", "origin_project", "note_id", "limit", "since_ms", "stream", "query")
    KIND_FIELD_NUMBER: _ClassVar[int]
    TTL_FIELD_NUMBER: _ClassVar[int]
    NONCE_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_PROJECT_FIELD_NUMBER: _ClassVar[int]
    NOTE_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    SINCE_MS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    kind: ServerQueryKind
    ttl: int
    nonce: str
    origin_project: str
    note_id: str
    limit: int
    since_ms: int
    stream: str
    query: str
    def __init__(self, kind: _Optional[_Union[ServerQueryKind, str]] = ..., ttl: _Optional[int] = ..., nonce: _Optional[str] = ..., origin_project: _Optional[str] = ..., note_id: _Optional[str] = ..., limit: _Optional[int] = ..., since_ms: _Optional[int] = ..., stream: _Optional[str] = ..., query: _Optional[str] = ...) -> None: ...

class ServerQueryResponse(_message.Message):
    __slots__ = ("project", "remotes", "head", "peers", "schedules", "note", "surfaces", "queues", "workloads", "posture", "products", "cognition", "contributions", "activities", "thoughts_hint", "agenda_hint", "search_hint", "agents", "fmp_hint", "resources_hint", "aspect_catalog", "product_specs")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    REMOTES_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    PEERS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULES_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    SURFACES_FIELD_NUMBER: _ClassVar[int]
    QUEUES_FIELD_NUMBER: _ClassVar[int]
    WORKLOADS_FIELD_NUMBER: _ClassVar[int]
    POSTURE_FIELD_NUMBER: _ClassVar[int]
    PRODUCTS_FIELD_NUMBER: _ClassVar[int]
    COGNITION_FIELD_NUMBER: _ClassVar[int]
    CONTRIBUTIONS_FIELD_NUMBER: _ClassVar[int]
    ACTIVITIES_FIELD_NUMBER: _ClassVar[int]
    THOUGHTS_HINT_FIELD_NUMBER: _ClassVar[int]
    AGENDA_HINT_FIELD_NUMBER: _ClassVar[int]
    SEARCH_HINT_FIELD_NUMBER: _ClassVar[int]
    AGENTS_FIELD_NUMBER: _ClassVar[int]
    FMP_HINT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_HINT_FIELD_NUMBER: _ClassVar[int]
    ASPECT_CATALOG_FIELD_NUMBER: _ClassVar[int]
    PRODUCT_SPECS_FIELD_NUMBER: _ClassVar[int]
    project: str
    remotes: _containers.RepeatedCompositeFieldContainer[GitRemote]
    head: str
    peers: _containers.RepeatedCompositeFieldContainer[PeerHint]
    schedules: _containers.RepeatedCompositeFieldContainer[ScheduleHint]
    note: WikiNote
    surfaces: _containers.RepeatedCompositeFieldContainer[Surface]
    queues: _containers.RepeatedCompositeFieldContainer[QueueHint]
    workloads: _containers.RepeatedCompositeFieldContainer[WorkloadOffer]
    posture: SourcePosture
    products: _containers.RepeatedCompositeFieldContainer[ProductHint]
    cognition: CognitionHint
    contributions: ContributionsHint
    activities: _containers.RepeatedCompositeFieldContainer[Activity]
    thoughts_hint: ThoughtsHint
    agenda_hint: AgendaHint
    search_hint: SearchHint
    agents: _containers.RepeatedCompositeFieldContainer[AgentHint]
    fmp_hint: FmpHint
    resources_hint: ResourcesHint
    aspect_catalog: _containers.RepeatedCompositeFieldContainer[AspectSpec]
    product_specs: _containers.RepeatedCompositeFieldContainer[ProductSpec]
    def __init__(self, project: _Optional[str] = ..., remotes: _Optional[_Iterable[_Union[GitRemote, _Mapping]]] = ..., head: _Optional[str] = ..., peers: _Optional[_Iterable[_Union[PeerHint, _Mapping]]] = ..., schedules: _Optional[_Iterable[_Union[ScheduleHint, _Mapping]]] = ..., note: _Optional[_Union[WikiNote, _Mapping]] = ..., surfaces: _Optional[_Iterable[_Union[Surface, _Mapping]]] = ..., queues: _Optional[_Iterable[_Union[QueueHint, _Mapping]]] = ..., workloads: _Optional[_Iterable[_Union[WorkloadOffer, _Mapping]]] = ..., posture: _Optional[_Union[SourcePosture, _Mapping]] = ..., products: _Optional[_Iterable[_Union[ProductHint, _Mapping]]] = ..., cognition: _Optional[_Union[CognitionHint, _Mapping]] = ..., contributions: _Optional[_Union[ContributionsHint, _Mapping]] = ..., activities: _Optional[_Iterable[_Union[Activity, _Mapping]]] = ..., thoughts_hint: _Optional[_Union[ThoughtsHint, _Mapping]] = ..., agenda_hint: _Optional[_Union[AgendaHint, _Mapping]] = ..., search_hint: _Optional[_Union[SearchHint, _Mapping]] = ..., agents: _Optional[_Iterable[_Union[AgentHint, _Mapping]]] = ..., fmp_hint: _Optional[_Union[FmpHint, _Mapping]] = ..., resources_hint: _Optional[_Union[ResourcesHint, _Mapping]] = ..., aspect_catalog: _Optional[_Iterable[_Union[AspectSpec, _Mapping]]] = ..., product_specs: _Optional[_Iterable[_Union[ProductSpec, _Mapping]]] = ...) -> None: ...

class AgentHint(_message.Message):
    __slots__ = ("agent_id", "project", "name", "version", "transport", "model_capabilities", "billing")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TRANSPORT_FIELD_NUMBER: _ClassVar[int]
    MODEL_CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    BILLING_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    project: str
    name: str
    version: str
    transport: str
    model_capabilities: _containers.RepeatedScalarFieldContainer[str]
    billing: str
    def __init__(self, agent_id: _Optional[str] = ..., project: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[str] = ..., transport: _Optional[str] = ..., model_capabilities: _Optional[_Iterable[str]] = ..., billing: _Optional[str] = ...) -> None: ...

class SearchHit(_message.Message):
    __slots__ = ("title", "url", "snippet", "source", "score")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    SNIPPET_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    title: str
    url: str
    snippet: str
    source: str
    score: float
    def __init__(self, title: _Optional[str] = ..., url: _Optional[str] = ..., snippet: _Optional[str] = ..., source: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class SearchHint(_message.Message):
    __slots__ = ("project", "query", "stream", "hits", "note")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    project: str
    query: str
    stream: str
    hits: _containers.RepeatedCompositeFieldContainer[SearchHit]
    note: str
    def __init__(self, project: _Optional[str] = ..., query: _Optional[str] = ..., stream: _Optional[str] = ..., hits: _Optional[_Iterable[_Union[SearchHit, _Mapping]]] = ..., note: _Optional[str] = ...) -> None: ...

class FmpHit(_message.Message):
    __slots__ = ("symbol", "title", "snippet", "url", "exchange", "as_of", "source")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    SNIPPET_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    AS_OF_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    title: str
    snippet: str
    url: str
    exchange: str
    as_of: str
    source: str
    def __init__(self, symbol: _Optional[str] = ..., title: _Optional[str] = ..., snippet: _Optional[str] = ..., url: _Optional[str] = ..., exchange: _Optional[str] = ..., as_of: _Optional[str] = ..., source: _Optional[str] = ...) -> None: ...

class FmpHint(_message.Message):
    __slots__ = ("project", "query", "stream", "hits", "note", "spoken")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    SPOKEN_FIELD_NUMBER: _ClassVar[int]
    project: str
    query: str
    stream: str
    hits: _containers.RepeatedCompositeFieldContainer[FmpHit]
    note: str
    spoken: str
    def __init__(self, project: _Optional[str] = ..., query: _Optional[str] = ..., stream: _Optional[str] = ..., hits: _Optional[_Iterable[_Union[FmpHit, _Mapping]]] = ..., note: _Optional[str] = ..., spoken: _Optional[str] = ...) -> None: ...

class AgendaHintItem(_message.Message):
    __slots__ = ("id", "starts_ms", "ends_ms", "kind", "intent", "title", "summary", "tags", "pinned", "open_checks", "with_whom", "body", "created_ms", "day", "origin_project", "session_prompt", "session_materials", "origin_agent", "attachments", "attachments_allowed")
    ID_FIELD_NUMBER: _ClassVar[int]
    STARTS_MS_FIELD_NUMBER: _ClassVar[int]
    ENDS_MS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    PINNED_FIELD_NUMBER: _ClassVar[int]
    OPEN_CHECKS_FIELD_NUMBER: _ClassVar[int]
    WITH_WHOM_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    CREATED_MS_FIELD_NUMBER: _ClassVar[int]
    DAY_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_PROJECT_FIELD_NUMBER: _ClassVar[int]
    SESSION_PROMPT_FIELD_NUMBER: _ClassVar[int]
    SESSION_MATERIALS_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_AGENT_FIELD_NUMBER: _ClassVar[int]
    ATTACHMENTS_FIELD_NUMBER: _ClassVar[int]
    ATTACHMENTS_ALLOWED_FIELD_NUMBER: _ClassVar[int]
    id: str
    starts_ms: int
    ends_ms: int
    kind: str
    intent: str
    title: str
    summary: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    pinned: bool
    open_checks: int
    with_whom: str
    body: str
    created_ms: int
    day: str
    origin_project: str
    session_prompt: str
    session_materials: str
    origin_agent: str
    attachments: _containers.RepeatedCompositeFieldContainer[Attachment]
    attachments_allowed: bool
    def __init__(self, id: _Optional[str] = ..., starts_ms: _Optional[int] = ..., ends_ms: _Optional[int] = ..., kind: _Optional[str] = ..., intent: _Optional[str] = ..., title: _Optional[str] = ..., summary: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., pinned: bool = ..., open_checks: _Optional[int] = ..., with_whom: _Optional[str] = ..., body: _Optional[str] = ..., created_ms: _Optional[int] = ..., day: _Optional[str] = ..., origin_project: _Optional[str] = ..., session_prompt: _Optional[str] = ..., session_materials: _Optional[str] = ..., origin_agent: _Optional[str] = ..., attachments: _Optional[_Iterable[_Union[Attachment, _Mapping]]] = ..., attachments_allowed: bool = ...) -> None: ...

class AgendaHint(_message.Message):
    __slots__ = ("project", "brief", "spoken", "brief_at_ms", "brief_id", "timezone", "today", "items", "item", "total_in_window", "note")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    BRIEF_FIELD_NUMBER: _ClassVar[int]
    SPOKEN_FIELD_NUMBER: _ClassVar[int]
    BRIEF_AT_MS_FIELD_NUMBER: _ClassVar[int]
    BRIEF_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    TODAY_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    TOTAL_IN_WINDOW_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    project: str
    brief: str
    spoken: str
    brief_at_ms: int
    brief_id: str
    timezone: str
    today: str
    items: _containers.RepeatedCompositeFieldContainer[AgendaHintItem]
    item: AgendaHintItem
    total_in_window: int
    note: str
    def __init__(self, project: _Optional[str] = ..., brief: _Optional[str] = ..., spoken: _Optional[str] = ..., brief_at_ms: _Optional[int] = ..., brief_id: _Optional[str] = ..., timezone: _Optional[str] = ..., today: _Optional[str] = ..., items: _Optional[_Iterable[_Union[AgendaHintItem, _Mapping]]] = ..., item: _Optional[_Union[AgendaHintItem, _Mapping]] = ..., total_in_window: _Optional[int] = ..., note: _Optional[str] = ...) -> None: ...

class PutAgendaItemRequest(_message.Message):
    __slots__ = ("origin_project", "origin_agent", "item")
    ORIGIN_PROJECT_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_AGENT_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    origin_project: str
    origin_agent: str
    item: AgendaHintItem
    def __init__(self, origin_project: _Optional[str] = ..., origin_agent: _Optional[str] = ..., item: _Optional[_Union[AgendaHintItem, _Mapping]] = ...) -> None: ...

class PutAgendaItemResponse(_message.Message):
    __slots__ = ("ok", "item", "note")
    OK_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    item: AgendaHintItem
    note: str
    def __init__(self, ok: bool = ..., item: _Optional[_Union[AgendaHintItem, _Mapping]] = ..., note: _Optional[str] = ...) -> None: ...

class ResourceObject(_message.Message):
    __slots__ = ("name", "text", "uri")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    name: str
    text: str
    uri: str
    def __init__(self, name: _Optional[str] = ..., text: _Optional[str] = ..., uri: _Optional[str] = ...) -> None: ...

class Attachment(_message.Message):
    __slots__ = ("name", "uri", "role", "media_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    MEDIA_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    uri: str
    role: str
    media_type: str
    def __init__(self, name: _Optional[str] = ..., uri: _Optional[str] = ..., role: _Optional[str] = ..., media_type: _Optional[str] = ...) -> None: ...

class ResourcesHint(_message.Message):
    __slots__ = ("project", "note_id", "objects", "note")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    NOTE_ID_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    project: str
    note_id: str
    objects: _containers.RepeatedCompositeFieldContainer[ResourceObject]
    note: str
    def __init__(self, project: _Optional[str] = ..., note_id: _Optional[str] = ..., objects: _Optional[_Iterable[_Union[ResourceObject, _Mapping]]] = ..., note: _Optional[str] = ...) -> None: ...

class Thought(_message.Message):
    __slots__ = ("id", "at_ms", "kind", "title", "summary", "excerpt", "domains", "salience", "chain_id", "generation", "note_path", "profile", "model")
    ID_FIELD_NUMBER: _ClassVar[int]
    AT_MS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    EXCERPT_FIELD_NUMBER: _ClassVar[int]
    DOMAINS_FIELD_NUMBER: _ClassVar[int]
    SALIENCE_FIELD_NUMBER: _ClassVar[int]
    CHAIN_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    NOTE_PATH_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    id: str
    at_ms: int
    kind: str
    title: str
    summary: str
    excerpt: str
    domains: _containers.RepeatedScalarFieldContainer[str]
    salience: float
    chain_id: str
    generation: int
    note_path: str
    profile: str
    model: str
    def __init__(self, id: _Optional[str] = ..., at_ms: _Optional[int] = ..., kind: _Optional[str] = ..., title: _Optional[str] = ..., summary: _Optional[str] = ..., excerpt: _Optional[str] = ..., domains: _Optional[_Iterable[str]] = ..., salience: _Optional[float] = ..., chain_id: _Optional[str] = ..., generation: _Optional[int] = ..., note_path: _Optional[str] = ..., profile: _Optional[str] = ..., model: _Optional[str] = ...) -> None: ...

class ThoughtsHint(_message.Message):
    __slots__ = ("project", "thoughts", "total_in_window", "newest_ms", "window_ms", "cycles_in_window", "note", "brief", "spoken", "brief_at_ms", "brief_id", "brief_thoughts")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    THOUGHTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_IN_WINDOW_FIELD_NUMBER: _ClassVar[int]
    NEWEST_MS_FIELD_NUMBER: _ClassVar[int]
    WINDOW_MS_FIELD_NUMBER: _ClassVar[int]
    CYCLES_IN_WINDOW_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    BRIEF_FIELD_NUMBER: _ClassVar[int]
    SPOKEN_FIELD_NUMBER: _ClassVar[int]
    BRIEF_AT_MS_FIELD_NUMBER: _ClassVar[int]
    BRIEF_ID_FIELD_NUMBER: _ClassVar[int]
    BRIEF_THOUGHTS_FIELD_NUMBER: _ClassVar[int]
    project: str
    thoughts: _containers.RepeatedCompositeFieldContainer[Thought]
    total_in_window: int
    newest_ms: int
    window_ms: int
    cycles_in_window: int
    note: str
    brief: str
    spoken: str
    brief_at_ms: int
    brief_id: str
    brief_thoughts: int
    def __init__(self, project: _Optional[str] = ..., thoughts: _Optional[_Iterable[_Union[Thought, _Mapping]]] = ..., total_in_window: _Optional[int] = ..., newest_ms: _Optional[int] = ..., window_ms: _Optional[int] = ..., cycles_in_window: _Optional[int] = ..., note: _Optional[str] = ..., brief: _Optional[str] = ..., spoken: _Optional[str] = ..., brief_at_ms: _Optional[int] = ..., brief_id: _Optional[str] = ..., brief_thoughts: _Optional[int] = ...) -> None: ...

class ActivityClaim(_message.Message):
    __slots__ = ("leaf", "gpu")
    LEAF_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    leaf: str
    gpu: int
    def __init__(self, leaf: _Optional[str] = ..., gpu: _Optional[int] = ...) -> None: ...

class Activity(_message.Message):
    __slots__ = ("activity_id", "kind", "peer", "owner", "dag_id", "run_id", "state", "declared_ns", "horizon_ns", "renewed_ns", "ended_ns", "claims", "precludes", "postures", "reason", "note")
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
    RENEWED_NS_FIELD_NUMBER: _ClassVar[int]
    ENDED_NS_FIELD_NUMBER: _ClassVar[int]
    CLAIMS_FIELD_NUMBER: _ClassVar[int]
    PRECLUDES_FIELD_NUMBER: _ClassVar[int]
    POSTURES_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    activity_id: str
    kind: str
    peer: str
    owner: str
    dag_id: str
    run_id: str
    state: ActivityState
    declared_ns: int
    horizon_ns: int
    renewed_ns: int
    ended_ns: int
    claims: _containers.RepeatedCompositeFieldContainer[ActivityClaim]
    precludes: _containers.RepeatedScalarFieldContainer[str]
    postures: _containers.ScalarMap[str, str]
    reason: str
    note: str
    def __init__(self, activity_id: _Optional[str] = ..., kind: _Optional[str] = ..., peer: _Optional[str] = ..., owner: _Optional[str] = ..., dag_id: _Optional[str] = ..., run_id: _Optional[str] = ..., state: _Optional[_Union[ActivityState, str]] = ..., declared_ns: _Optional[int] = ..., horizon_ns: _Optional[int] = ..., renewed_ns: _Optional[int] = ..., ended_ns: _Optional[int] = ..., claims: _Optional[_Iterable[_Union[ActivityClaim, _Mapping]]] = ..., precludes: _Optional[_Iterable[str]] = ..., postures: _Optional[_Mapping[str, str]] = ..., reason: _Optional[str] = ..., note: _Optional[str] = ...) -> None: ...

class CognitionActivityBucket(_message.Message):
    __slots__ = ("start_ms", "end_ms", "thoughts", "cycles")
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    THOUGHTS_FIELD_NUMBER: _ClassVar[int]
    CYCLES_FIELD_NUMBER: _ClassVar[int]
    start_ms: int
    end_ms: int
    thoughts: int
    cycles: int
    def __init__(self, start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., thoughts: _Optional[int] = ..., cycles: _Optional[int] = ...) -> None: ...

class CognitionStreamHint(_message.Message):
    __slots__ = ("id", "thoughts")
    ID_FIELD_NUMBER: _ClassVar[int]
    THOUGHTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    thoughts: int
    def __init__(self, id: _Optional[str] = ..., thoughts: _Optional[int] = ...) -> None: ...

class CognitionHint(_message.Message):
    __slots__ = ("project", "unit", "running", "thoughts", "cycles", "last_cycle_ms", "interval", "range_start_ms", "range_end_ms", "buckets", "streams")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_FIELD_NUMBER: _ClassVar[int]
    THOUGHTS_FIELD_NUMBER: _ClassVar[int]
    CYCLES_FIELD_NUMBER: _ClassVar[int]
    LAST_CYCLE_MS_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    RANGE_START_MS_FIELD_NUMBER: _ClassVar[int]
    RANGE_END_MS_FIELD_NUMBER: _ClassVar[int]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    STREAMS_FIELD_NUMBER: _ClassVar[int]
    project: str
    unit: str
    running: bool
    thoughts: int
    cycles: int
    last_cycle_ms: int
    interval: str
    range_start_ms: int
    range_end_ms: int
    buckets: _containers.RepeatedCompositeFieldContainer[CognitionActivityBucket]
    streams: _containers.RepeatedCompositeFieldContainer[CognitionStreamHint]
    def __init__(self, project: _Optional[str] = ..., unit: _Optional[str] = ..., running: bool = ..., thoughts: _Optional[int] = ..., cycles: _Optional[int] = ..., last_cycle_ms: _Optional[int] = ..., interval: _Optional[str] = ..., range_start_ms: _Optional[int] = ..., range_end_ms: _Optional[int] = ..., buckets: _Optional[_Iterable[_Union[CognitionActivityBucket, _Mapping]]] = ..., streams: _Optional[_Iterable[_Union[CognitionStreamHint, _Mapping]]] = ...) -> None: ...

class ContributionItem(_message.Message):
    __slots__ = ("group", "id", "system", "total", "series")
    GROUP_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    SERIES_FIELD_NUMBER: _ClassVar[int]
    group: str
    id: str
    system: str
    total: int
    series: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, group: _Optional[str] = ..., id: _Optional[str] = ..., system: _Optional[str] = ..., total: _Optional[int] = ..., series: _Optional[_Iterable[int]] = ...) -> None: ...

class ContributionsHint(_message.Message):
    __slots__ = ("project", "interval", "range_start_ms", "range_end_ms", "buckets", "items")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    RANGE_START_MS_FIELD_NUMBER: _ClassVar[int]
    RANGE_END_MS_FIELD_NUMBER: _ClassVar[int]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    project: str
    interval: str
    range_start_ms: int
    range_end_ms: int
    buckets: _containers.RepeatedCompositeFieldContainer[CognitionActivityBucket]
    items: _containers.RepeatedCompositeFieldContainer[ContributionItem]
    def __init__(self, project: _Optional[str] = ..., interval: _Optional[str] = ..., range_start_ms: _Optional[int] = ..., range_end_ms: _Optional[int] = ..., buckets: _Optional[_Iterable[_Union[CognitionActivityBucket, _Mapping]]] = ..., items: _Optional[_Iterable[_Union[ContributionItem, _Mapping]]] = ...) -> None: ...

class PropertyPath(_message.Message):
    __slots__ = ("predicate",)
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    predicate: str
    def __init__(self, predicate: _Optional[str] = ...) -> None: ...

class PropertyShape(_message.Message):
    __slots__ = ("id", "path", "name", "description", "min_count", "max_count", "datatype", "node_kind", "pattern", "flags", "has_value", "min_length", "max_length", "node", "severity", "message", "deactivated")
    ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MIN_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_COUNT_FIELD_NUMBER: _ClassVar[int]
    DATATYPE_FIELD_NUMBER: _ClassVar[int]
    CLASS_FIELD_NUMBER: _ClassVar[int]
    NODE_KIND_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    IN_FIELD_NUMBER: _ClassVar[int]
    HAS_VALUE_FIELD_NUMBER: _ClassVar[int]
    MIN_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_LENGTH_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DEACTIVATED_FIELD_NUMBER: _ClassVar[int]
    id: str
    path: PropertyPath
    name: str
    description: str
    min_count: int
    max_count: int
    datatype: str
    node_kind: ShaclNodeKind
    pattern: str
    flags: str
    has_value: str
    min_length: int
    max_length: int
    node: str
    severity: ShaclSeverity
    message: str
    deactivated: bool
    def __init__(self, id: _Optional[str] = ..., path: _Optional[_Union[PropertyPath, _Mapping]] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., min_count: _Optional[int] = ..., max_count: _Optional[int] = ..., datatype: _Optional[str] = ..., node_kind: _Optional[_Union[ShaclNodeKind, str]] = ..., pattern: _Optional[str] = ..., flags: _Optional[str] = ..., has_value: _Optional[str] = ..., min_length: _Optional[int] = ..., max_length: _Optional[int] = ..., node: _Optional[str] = ..., severity: _Optional[_Union[ShaclSeverity, str]] = ..., message: _Optional[str] = ..., deactivated: bool = ..., **kwargs) -> None: ...

class NodeShape(_message.Message):
    __slots__ = ("id", "target_class", "target_node", "closed", "ignored_properties", "property", "xone", "severity", "message", "deactivated")
    ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_CLASS_FIELD_NUMBER: _ClassVar[int]
    TARGET_NODE_FIELD_NUMBER: _ClassVar[int]
    CLOSED_FIELD_NUMBER: _ClassVar[int]
    IGNORED_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    AND_FIELD_NUMBER: _ClassVar[int]
    OR_FIELD_NUMBER: _ClassVar[int]
    NOT_FIELD_NUMBER: _ClassVar[int]
    XONE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DEACTIVATED_FIELD_NUMBER: _ClassVar[int]
    id: str
    target_class: _containers.RepeatedScalarFieldContainer[str]
    target_node: _containers.RepeatedScalarFieldContainer[str]
    closed: bool
    ignored_properties: _containers.RepeatedScalarFieldContainer[str]
    property: _containers.RepeatedCompositeFieldContainer[PropertyShape]
    xone: _containers.RepeatedScalarFieldContainer[str]
    severity: ShaclSeverity
    message: str
    deactivated: bool
    def __init__(self, id: _Optional[str] = ..., target_class: _Optional[_Iterable[str]] = ..., target_node: _Optional[_Iterable[str]] = ..., closed: bool = ..., ignored_properties: _Optional[_Iterable[str]] = ..., property: _Optional[_Iterable[_Union[PropertyShape, _Mapping]]] = ..., xone: _Optional[_Iterable[str]] = ..., severity: _Optional[_Union[ShaclSeverity, str]] = ..., message: _Optional[str] = ..., deactivated: bool = ..., **kwargs) -> None: ...

class AspectSpec(_message.Message):
    __slots__ = ("shape", "title", "intent")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    shape: NodeShape
    title: str
    intent: str
    def __init__(self, shape: _Optional[_Union[NodeShape, _Mapping]] = ..., title: _Optional[str] = ..., intent: _Optional[str] = ...) -> None: ...

class ProductSpec(_message.Message):
    __slots__ = ("shape", "title")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    shape: NodeShape
    title: str
    def __init__(self, shape: _Optional[_Union[NodeShape, _Mapping]] = ..., title: _Optional[str] = ...) -> None: ...

class ValidationResult(_message.Message):
    __slots__ = ("severity", "focus_node", "result_path", "value", "source_constraint_component", "source_shape", "result_message")
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    FOCUS_NODE_FIELD_NUMBER: _ClassVar[int]
    RESULT_PATH_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_CONSTRAINT_COMPONENT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SHAPE_FIELD_NUMBER: _ClassVar[int]
    RESULT_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    severity: ShaclSeverity
    focus_node: str
    result_path: str
    value: str
    source_constraint_component: str
    source_shape: str
    result_message: str
    def __init__(self, severity: _Optional[_Union[ShaclSeverity, str]] = ..., focus_node: _Optional[str] = ..., result_path: _Optional[str] = ..., value: _Optional[str] = ..., source_constraint_component: _Optional[str] = ..., source_shape: _Optional[str] = ..., result_message: _Optional[str] = ...) -> None: ...

class ValidationReport(_message.Message):
    __slots__ = ("conforms", "result")
    CONFORMS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    conforms: bool
    result: _containers.RepeatedCompositeFieldContainer[ValidationResult]
    def __init__(self, conforms: bool = ..., result: _Optional[_Iterable[_Union[ValidationResult, _Mapping]]] = ...) -> None: ...

class AspectBinding(_message.Message):
    __slots__ = ("aspect_id", "conforms", "claim_status", "report", "evidence_uri", "seal")
    ASPECT_ID_FIELD_NUMBER: _ClassVar[int]
    CONFORMS_FIELD_NUMBER: _ClassVar[int]
    CLAIM_STATUS_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_URI_FIELD_NUMBER: _ClassVar[int]
    SEAL_FIELD_NUMBER: _ClassVar[int]
    aspect_id: str
    conforms: bool
    claim_status: str
    report: ValidationReport
    evidence_uri: str
    seal: str
    def __init__(self, aspect_id: _Optional[str] = ..., conforms: bool = ..., claim_status: _Optional[str] = ..., report: _Optional[_Union[ValidationReport, _Mapping]] = ..., evidence_uri: _Optional[str] = ..., seal: _Optional[str] = ...) -> None: ...

class ProductHint(_message.Message):
    __slots__ = ("product_id", "peer", "title", "kind", "leaf", "table_identifier", "data_uri", "flow", "step", "agent_focus", "history", "spec_id", "aspects")
    PRODUCT_ID_FIELD_NUMBER: _ClassVar[int]
    PEER_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LEAF_FIELD_NUMBER: _ClassVar[int]
    TABLE_IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    DATA_URI_FIELD_NUMBER: _ClassVar[int]
    FLOW_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    AGENT_FOCUS_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    SPEC_ID_FIELD_NUMBER: _ClassVar[int]
    ASPECTS_FIELD_NUMBER: _ClassVar[int]
    product_id: str
    peer: str
    title: str
    kind: str
    leaf: str
    table_identifier: str
    data_uri: str
    flow: str
    step: str
    agent_focus: str
    history: str
    spec_id: str
    aspects: _containers.RepeatedCompositeFieldContainer[AspectBinding]
    def __init__(self, product_id: _Optional[str] = ..., peer: _Optional[str] = ..., title: _Optional[str] = ..., kind: _Optional[str] = ..., leaf: _Optional[str] = ..., table_identifier: _Optional[str] = ..., data_uri: _Optional[str] = ..., flow: _Optional[str] = ..., step: _Optional[str] = ..., agent_focus: _Optional[str] = ..., history: _Optional[str] = ..., spec_id: _Optional[str] = ..., aspects: _Optional[_Iterable[_Union[AspectBinding, _Mapping]]] = ...) -> None: ...

class ModelParallelism(_message.Message):
    __slots__ = ("tensor_parallel", "pipeline_parallel", "data_parallel")
    TENSOR_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    def __init__(self, tensor_parallel: _Optional[int] = ..., pipeline_parallel: _Optional[int] = ..., data_parallel: _Optional[int] = ...) -> None: ...

class ResourceFootprint(_message.Message):
    __slots__ = ("gpu", "vram_mib", "memory_mib", "vcore")
    GPU_FIELD_NUMBER: _ClassVar[int]
    VRAM_MIB_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MIB_FIELD_NUMBER: _ClassVar[int]
    VCORE_FIELD_NUMBER: _ClassVar[int]
    gpu: int
    vram_mib: int
    memory_mib: int
    vcore: int
    def __init__(self, gpu: _Optional[int] = ..., vram_mib: _Optional[int] = ..., memory_mib: _Optional[int] = ..., vcore: _Optional[int] = ...) -> None: ...

class KServeTarget(_message.Message):
    __slots__ = ("inference_service", "namespace", "serving_runtime")
    INFERENCE_SERVICE_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    SERVING_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    inference_service: str
    namespace: str
    serving_runtime: str
    def __init__(self, inference_service: _Optional[str] = ..., namespace: _Optional[str] = ..., serving_runtime: _Optional[str] = ...) -> None: ...

class WorkloadRequirements(_message.Message):
    __slots__ = ("backend", "parallelism", "footprint", "kserve")
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    PARALLELISM_FIELD_NUMBER: _ClassVar[int]
    FOOTPRINT_FIELD_NUMBER: _ClassVar[int]
    KSERVE_FIELD_NUMBER: _ClassVar[int]
    backend: ServingBackend
    parallelism: ModelParallelism
    footprint: ResourceFootprint
    kserve: KServeTarget
    def __init__(self, backend: _Optional[_Union[ServingBackend, str]] = ..., parallelism: _Optional[_Union[ModelParallelism, _Mapping]] = ..., footprint: _Optional[_Union[ResourceFootprint, _Mapping]] = ..., kserve: _Optional[_Union[KServeTarget, _Mapping]] = ...) -> None: ...

class OperatingProfile(_message.Message):
    __slots__ = ("capability", "thinking", "reasoning_effort", "note")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    THINKING_FIELD_NUMBER: _ClassVar[int]
    REASONING_EFFORT_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    capability: str
    thinking: bool
    reasoning_effort: str
    note: str
    def __init__(self, capability: _Optional[str] = ..., thinking: bool = ..., reasoning_effort: _Optional[str] = ..., note: _Optional[str] = ...) -> None: ...

class WorkloadOffer(_message.Message):
    __slots__ = ("peer", "model", "capabilities", "requirements", "resource_class", "queue", "methods", "profiles")
    PEER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_CLASS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    METHODS_FIELD_NUMBER: _ClassVar[int]
    PROFILES_FIELD_NUMBER: _ClassVar[int]
    peer: str
    model: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    requirements: WorkloadRequirements
    resource_class: ResourceClass
    queue: str
    methods: _containers.RepeatedScalarFieldContainer[str]
    profiles: _containers.RepeatedCompositeFieldContainer[OperatingProfile]
    def __init__(self, peer: _Optional[str] = ..., model: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ..., requirements: _Optional[_Union[WorkloadRequirements, _Mapping]] = ..., resource_class: _Optional[_Union[ResourceClass, str]] = ..., queue: _Optional[str] = ..., methods: _Optional[_Iterable[str]] = ..., profiles: _Optional[_Iterable[_Union[OperatingProfile, _Mapping]]] = ...) -> None: ...

class QueueHint(_message.Message):
    __slots__ = ("path", "resource_class", "gpu_guarantee", "gpu_max", "max_applications", "preemption_policy", "preemption_delay", "role", "examples")
    PATH_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_CLASS_FIELD_NUMBER: _ClassVar[int]
    GPU_GUARANTEE_FIELD_NUMBER: _ClassVar[int]
    GPU_MAX_FIELD_NUMBER: _ClassVar[int]
    MAX_APPLICATIONS_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_POLICY_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_DELAY_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    path: str
    resource_class: str
    gpu_guarantee: int
    gpu_max: int
    max_applications: int
    preemption_policy: str
    preemption_delay: str
    role: str
    examples: str
    def __init__(self, path: _Optional[str] = ..., resource_class: _Optional[str] = ..., gpu_guarantee: _Optional[int] = ..., gpu_max: _Optional[int] = ..., max_applications: _Optional[int] = ..., preemption_policy: _Optional[str] = ..., preemption_delay: _Optional[str] = ..., role: _Optional[str] = ..., examples: _Optional[str] = ...) -> None: ...

class LineageRequest(_message.Message):
    __slots__ = ("event_json", "event_type")
    EVENT_JSON_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    event_json: str
    event_type: str
    def __init__(self, event_json: _Optional[str] = ..., event_type: _Optional[str] = ...) -> None: ...

class LineageResponse(_message.Message):
    __slots__ = ("accepted", "error")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    error: str
    def __init__(self, accepted: bool = ..., error: _Optional[str] = ...) -> None: ...

class WatchWorkloadRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WorkloadIntent(_message.Message):
    __slots__ = ("capability", "alias", "model", "port", "gpu_ids", "backend", "warmup_seconds", "actual")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    GPU_IDS_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    WARMUP_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_FIELD_NUMBER: _ClassVar[int]
    capability: str
    alias: str
    model: str
    port: int
    gpu_ids: _containers.RepeatedScalarFieldContainer[int]
    backend: ServingBackend
    warmup_seconds: int
    actual: WorkloadStatus
    def __init__(self, capability: _Optional[str] = ..., alias: _Optional[str] = ..., model: _Optional[str] = ..., port: _Optional[int] = ..., gpu_ids: _Optional[_Iterable[int]] = ..., backend: _Optional[_Union[ServingBackend, str]] = ..., warmup_seconds: _Optional[int] = ..., actual: _Optional[_Union[WorkloadStatus, str]] = ...) -> None: ...

class WorkloadProfile(_message.Message):
    __slots__ = ("phase", "generation", "intents", "settled_at_unix_ms", "detail")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    INTENTS_FIELD_NUMBER: _ClassVar[int]
    SETTLED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    phase: WorkloadPhase
    generation: int
    intents: _containers.RepeatedCompositeFieldContainer[WorkloadIntent]
    settled_at_unix_ms: int
    detail: str
    def __init__(self, phase: _Optional[_Union[WorkloadPhase, str]] = ..., generation: _Optional[int] = ..., intents: _Optional[_Iterable[_Union[WorkloadIntent, _Mapping]]] = ..., settled_at_unix_ms: _Optional[int] = ..., detail: _Optional[str] = ...) -> None: ...

class PeerAnnounce(_message.Message):
    __slots__ = ("project", "engine_target", "surfaces", "ttl_seconds")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    ENGINE_TARGET_FIELD_NUMBER: _ClassVar[int]
    SURFACES_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    project: str
    engine_target: str
    surfaces: _containers.RepeatedCompositeFieldContainer[Surface]
    ttl_seconds: int
    def __init__(self, project: _Optional[str] = ..., engine_target: _Optional[str] = ..., surfaces: _Optional[_Iterable[_Union[Surface, _Mapping]]] = ..., ttl_seconds: _Optional[int] = ...) -> None: ...

class AnnounceAck(_message.Message):
    __slots__ = ("accepted", "ttl_seconds", "error")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    ttl_seconds: int
    error: str
    def __init__(self, accepted: bool = ..., ttl_seconds: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...
