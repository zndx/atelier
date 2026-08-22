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
    __slots__ = ("capability", "prompt", "system_prompt", "max_tokens", "temperature", "json_schema", "timezone", "clock_json")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    CLOCK_JSON_FIELD_NUMBER: _ClassVar[int]
    capability: str
    prompt: str
    system_prompt: str
    max_tokens: int
    temperature: float
    json_schema: str
    timezone: str
    clock_json: str
    def __init__(self, capability: _Optional[str] = ..., prompt: _Optional[str] = ..., system_prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., json_schema: _Optional[str] = ..., timezone: _Optional[str] = ..., clock_json: _Optional[str] = ...) -> None: ...

class CompleteResponse(_message.Message):
    __slots__ = ("text", "model", "prompt_tokens", "completion_tokens", "latency_ms", "reasoning_content", "finish_reason")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    REASONING_CONTENT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    reasoning_content: str
    finish_reason: str
    def __init__(self, text: _Optional[str] = ..., model: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., latency_ms: _Optional[float] = ..., reasoning_content: _Optional[str] = ..., finish_reason: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("id", "cron", "airflow_dag_id", "source", "enabled")
    ID_FIELD_NUMBER: _ClassVar[int]
    CRON_FIELD_NUMBER: _ClassVar[int]
    AIRFLOW_DAG_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    id: str
    cron: str
    airflow_dag_id: str
    source: str
    enabled: bool
    def __init__(self, id: _Optional[str] = ..., cron: _Optional[str] = ..., airflow_dag_id: _Optional[str] = ..., source: _Optional[str] = ..., enabled: bool = ...) -> None: ...

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

class ServerQueryRequest(_message.Message):
    __slots__ = ("kind", "ttl", "nonce", "origin_project", "note_id")
    KIND_FIELD_NUMBER: _ClassVar[int]
    TTL_FIELD_NUMBER: _ClassVar[int]
    NONCE_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_PROJECT_FIELD_NUMBER: _ClassVar[int]
    NOTE_ID_FIELD_NUMBER: _ClassVar[int]
    kind: ServerQueryKind
    ttl: int
    nonce: str
    origin_project: str
    note_id: str
    def __init__(self, kind: _Optional[_Union[ServerQueryKind, str]] = ..., ttl: _Optional[int] = ..., nonce: _Optional[str] = ..., origin_project: _Optional[str] = ..., note_id: _Optional[str] = ...) -> None: ...

class ServerQueryResponse(_message.Message):
    __slots__ = ("project", "remotes", "head", "peers", "schedules", "note", "surfaces", "queues", "workloads")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    REMOTES_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    PEERS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULES_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    SURFACES_FIELD_NUMBER: _ClassVar[int]
    QUEUES_FIELD_NUMBER: _ClassVar[int]
    WORKLOADS_FIELD_NUMBER: _ClassVar[int]
    project: str
    remotes: _containers.RepeatedCompositeFieldContainer[GitRemote]
    head: str
    peers: _containers.RepeatedCompositeFieldContainer[PeerHint]
    schedules: _containers.RepeatedCompositeFieldContainer[ScheduleHint]
    note: WikiNote
    surfaces: _containers.RepeatedCompositeFieldContainer[Surface]
    queues: _containers.RepeatedCompositeFieldContainer[QueueHint]
    workloads: _containers.RepeatedCompositeFieldContainer[WorkloadHint]
    def __init__(self, project: _Optional[str] = ..., remotes: _Optional[_Iterable[_Union[GitRemote, _Mapping]]] = ..., head: _Optional[str] = ..., peers: _Optional[_Iterable[_Union[PeerHint, _Mapping]]] = ..., schedules: _Optional[_Iterable[_Union[ScheduleHint, _Mapping]]] = ..., note: _Optional[_Union[WikiNote, _Mapping]] = ..., surfaces: _Optional[_Iterable[_Union[Surface, _Mapping]]] = ..., queues: _Optional[_Iterable[_Union[QueueHint, _Mapping]]] = ..., workloads: _Optional[_Iterable[_Union[WorkloadHint, _Mapping]]] = ...) -> None: ...

class WorkloadHint(_message.Message):
    __slots__ = ("wrk", "model", "capabilities", "tensor_parallel", "pipeline_parallel", "gpu_tokens")
    WRK_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    TENSOR_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    GPU_TOKENS_FIELD_NUMBER: _ClassVar[int]
    wrk: str
    model: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    tensor_parallel: int
    pipeline_parallel: int
    gpu_tokens: int
    def __init__(self, wrk: _Optional[str] = ..., model: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ..., tensor_parallel: _Optional[int] = ..., pipeline_parallel: _Optional[int] = ..., gpu_tokens: _Optional[int] = ...) -> None: ...

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
