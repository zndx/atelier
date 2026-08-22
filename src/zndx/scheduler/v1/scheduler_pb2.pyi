from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProjectionRoot(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROJECTION_ROOT_UNSPECIFIED: _ClassVar[ProjectionRoot]
    CURRENT: _ClassVar[ProjectionRoot]
    SCRATCH: _ClassVar[ProjectionRoot]
    ARCHIVE: _ClassVar[ProjectionRoot]

class QueueShareState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    QUEUE_SHARE_STATE_UNSPECIFIED: _ClassVar[QueueShareState]
    QUEUE_SHARE_RECORDED: _ClassVar[QueueShareState]
    QUEUE_SHARE_SUPERSEDED: _ClassVar[QueueShareState]
    QUEUE_SHARE_APPLIED: _ClassVar[QueueShareState]
    QUEUE_SHARE_REJECTED: _ClassVar[QueueShareState]
PROJECTION_ROOT_UNSPECIFIED: ProjectionRoot
CURRENT: ProjectionRoot
SCRATCH: ProjectionRoot
ARCHIVE: ProjectionRoot
QUEUE_SHARE_STATE_UNSPECIFIED: QueueShareState
QUEUE_SHARE_RECORDED: QueueShareState
QUEUE_SHARE_SUPERSEDED: QueueShareState
QUEUE_SHARE_APPLIED: QueueShareState
QUEUE_SHARE_REJECTED: QueueShareState

class PolicyDocument(_message.Message):
    __slots__ = ("media_type", "body")
    MEDIA_TYPE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    media_type: str
    body: str
    def __init__(self, media_type: _Optional[str] = ..., body: _Optional[str] = ...) -> None: ...

class ListPartitionsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PartitionInfo(_message.Message):
    __slots__ = ("name", "state", "site_id", "capacity", "used_capacity", "total_nodes", "preemption_enabled")
    class CapacityEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class UsedCapacityEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    SITE_ID_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    USED_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_ENABLED_FIELD_NUMBER: _ClassVar[int]
    name: str
    state: str
    site_id: str
    capacity: _containers.ScalarMap[str, int]
    used_capacity: _containers.ScalarMap[str, int]
    total_nodes: int
    preemption_enabled: bool
    def __init__(self, name: _Optional[str] = ..., state: _Optional[str] = ..., site_id: _Optional[str] = ..., capacity: _Optional[_Mapping[str, int]] = ..., used_capacity: _Optional[_Mapping[str, int]] = ..., total_nodes: _Optional[int] = ..., preemption_enabled: bool = ...) -> None: ...

class ListPartitionsResponse(_message.Message):
    __slots__ = ("partitions",)
    PARTITIONS_FIELD_NUMBER: _ClassVar[int]
    partitions: _containers.RepeatedCompositeFieldContainer[PartitionInfo]
    def __init__(self, partitions: _Optional[_Iterable[_Union[PartitionInfo, _Mapping]]] = ...) -> None: ...

class GetQueueTreeRequest(_message.Message):
    __slots__ = ("partition",)
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    partition: str
    def __init__(self, partition: _Optional[str] = ...) -> None: ...

class ResourceMap(_message.Message):
    __slots__ = ("quantities",)
    class QuantitiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    QUANTITIES_FIELD_NUMBER: _ClassVar[int]
    quantities: _containers.ScalarMap[str, int]
    def __init__(self, quantities: _Optional[_Mapping[str, int]] = ...) -> None: ...

class QueueNode(_message.Message):
    __slots__ = ("name", "status", "is_leaf", "is_managed", "parent", "max", "guaranteed", "allocated", "pending", "headroom", "abs_used_capacity", "max_running_apps", "running_apps", "sorting_policy", "preemption_enabled", "properties", "children")
    class AbsUsedCapacityEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class PropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    IS_LEAF_FIELD_NUMBER: _ClassVar[int]
    IS_MANAGED_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    GUARANTEED_FIELD_NUMBER: _ClassVar[int]
    ALLOCATED_FIELD_NUMBER: _ClassVar[int]
    PENDING_FIELD_NUMBER: _ClassVar[int]
    HEADROOM_FIELD_NUMBER: _ClassVar[int]
    ABS_USED_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    MAX_RUNNING_APPS_FIELD_NUMBER: _ClassVar[int]
    RUNNING_APPS_FIELD_NUMBER: _ClassVar[int]
    SORTING_POLICY_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_ENABLED_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    CHILDREN_FIELD_NUMBER: _ClassVar[int]
    name: str
    status: str
    is_leaf: bool
    is_managed: bool
    parent: str
    max: ResourceMap
    guaranteed: ResourceMap
    allocated: ResourceMap
    pending: ResourceMap
    headroom: ResourceMap
    abs_used_capacity: _containers.ScalarMap[str, float]
    max_running_apps: int
    running_apps: int
    sorting_policy: str
    preemption_enabled: bool
    properties: _containers.ScalarMap[str, str]
    children: _containers.RepeatedCompositeFieldContainer[QueueNode]
    def __init__(self, name: _Optional[str] = ..., status: _Optional[str] = ..., is_leaf: bool = ..., is_managed: bool = ..., parent: _Optional[str] = ..., max: _Optional[_Union[ResourceMap, _Mapping]] = ..., guaranteed: _Optional[_Union[ResourceMap, _Mapping]] = ..., allocated: _Optional[_Union[ResourceMap, _Mapping]] = ..., pending: _Optional[_Union[ResourceMap, _Mapping]] = ..., headroom: _Optional[_Union[ResourceMap, _Mapping]] = ..., abs_used_capacity: _Optional[_Mapping[str, float]] = ..., max_running_apps: _Optional[int] = ..., running_apps: _Optional[int] = ..., sorting_policy: _Optional[str] = ..., preemption_enabled: bool = ..., properties: _Optional[_Mapping[str, str]] = ..., children: _Optional[_Iterable[_Union[QueueNode, _Mapping]]] = ...) -> None: ...

class GetQueueTreeResponse(_message.Message):
    __slots__ = ("partition", "root")
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    partition: str
    root: QueueNode
    def __init__(self, partition: _Optional[str] = ..., root: _Optional[_Union[QueueNode, _Mapping]] = ...) -> None: ...

class GetQueueRequest(_message.Message):
    __slots__ = ("partition", "queue", "include_subtree")
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_SUBTREE_FIELD_NUMBER: _ClassVar[int]
    partition: str
    queue: str
    include_subtree: bool
    def __init__(self, partition: _Optional[str] = ..., queue: _Optional[str] = ..., include_subtree: bool = ...) -> None: ...

class GetQueueResponse(_message.Message):
    __slots__ = ("partition", "queue")
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    partition: str
    queue: QueueNode
    def __init__(self, partition: _Optional[str] = ..., queue: _Optional[_Union[QueueNode, _Mapping]] = ...) -> None: ...

class ListQueueApplicationsRequest(_message.Message):
    __slots__ = ("partition", "queue")
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    partition: str
    queue: str
    def __init__(self, partition: _Optional[str] = ..., queue: _Optional[str] = ...) -> None: ...

class ApplicationInfo(_message.Message):
    __slots__ = ("application_id", "state", "queue_name", "user", "used", "submission_time_ns")
    APPLICATION_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_NAME_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    USED_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    application_id: str
    state: str
    queue_name: str
    user: str
    used: ResourceMap
    submission_time_ns: int
    def __init__(self, application_id: _Optional[str] = ..., state: _Optional[str] = ..., queue_name: _Optional[str] = ..., user: _Optional[str] = ..., used: _Optional[_Union[ResourceMap, _Mapping]] = ..., submission_time_ns: _Optional[int] = ...) -> None: ...

class ListQueueApplicationsResponse(_message.Message):
    __slots__ = ("applications",)
    APPLICATIONS_FIELD_NUMBER: _ClassVar[int]
    applications: _containers.RepeatedCompositeFieldContainer[ApplicationInfo]
    def __init__(self, applications: _Optional[_Iterable[_Union[ApplicationInfo, _Mapping]]] = ...) -> None: ...

class ListNodesRequest(_message.Message):
    __slots__ = ("partition",)
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    partition: str
    def __init__(self, partition: _Optional[str] = ...) -> None: ...

class NodeInfo(_message.Message):
    __slots__ = ("node_id", "host_name", "rack_name", "capacity", "allocated", "available", "schedulable")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HOST_NAME_FIELD_NUMBER: _ClassVar[int]
    RACK_NAME_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    ALLOCATED_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    SCHEDULABLE_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    host_name: str
    rack_name: str
    capacity: ResourceMap
    allocated: ResourceMap
    available: ResourceMap
    schedulable: bool
    def __init__(self, node_id: _Optional[str] = ..., host_name: _Optional[str] = ..., rack_name: _Optional[str] = ..., capacity: _Optional[_Union[ResourceMap, _Mapping]] = ..., allocated: _Optional[_Union[ResourceMap, _Mapping]] = ..., available: _Optional[_Union[ResourceMap, _Mapping]] = ..., schedulable: bool = ...) -> None: ...

class ListNodesResponse(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeInfo]
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeInfo, _Mapping]]] = ...) -> None: ...

class GetPlacementPolicyRequest(_message.Message):
    __slots__ = ("partition",)
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    partition: str
    def __init__(self, partition: _Optional[str] = ...) -> None: ...

class PlacementRule(_message.Message):
    __slots__ = ("name", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GetPlacementPolicyResponse(_message.Message):
    __slots__ = ("rules",)
    RULES_FIELD_NUMBER: _ClassVar[int]
    rules: _containers.RepeatedCompositeFieldContainer[PlacementRule]
    def __init__(self, rules: _Optional[_Iterable[_Union[PlacementRule, _Mapping]]] = ...) -> None: ...

class GetDeclaredConfigRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetDeclaredConfigResponse(_message.Message):
    __slots__ = ("document",)
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    document: PolicyDocument
    def __init__(self, document: _Optional[_Union[PolicyDocument, _Mapping]] = ...) -> None: ...

class ValidateConfigRequest(_message.Message):
    __slots__ = ("document",)
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    document: PolicyDocument
    def __init__(self, document: _Optional[_Union[PolicyDocument, _Mapping]] = ...) -> None: ...

class ValidateConfigResponse(_message.Message):
    __slots__ = ("ok", "message", "errors")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ok: bool = ..., message: _Optional[str] = ..., errors: _Optional[_Iterable[str]] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheck(_message.Message):
    __slots__ = ("name", "succeeded", "description", "diagnosis")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DIAGNOSIS_FIELD_NUMBER: _ClassVar[int]
    name: str
    succeeded: bool
    description: str
    diagnosis: str
    def __init__(self, name: _Optional[str] = ..., succeeded: bool = ..., description: _Optional[str] = ..., diagnosis: _Optional[str] = ...) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "backend", "checks")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    CHECKS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    backend: str
    checks: _containers.RepeatedCompositeFieldContainer[HealthCheck]
    def __init__(self, healthy: bool = ..., backend: _Optional[str] = ..., checks: _Optional[_Iterable[_Union[HealthCheck, _Mapping]]] = ...) -> None: ...

class GetDashboardRequest(_message.Message):
    __slots__ = ("partition",)
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    partition: str
    def __init__(self, partition: _Optional[str] = ...) -> None: ...

class SchedulerSummary(_message.Message):
    __slots__ = ("name", "backend", "status", "total_nodes", "node_sort_policy", "total_applications", "total_tasks", "partition", "site_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
    NODE_SORT_POLICY_FIELD_NUMBER: _ClassVar[int]
    TOTAL_APPLICATIONS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TASKS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    SITE_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    backend: str
    status: str
    total_nodes: int
    node_sort_policy: str
    total_applications: int
    total_tasks: int
    partition: str
    site_id: str
    def __init__(self, name: _Optional[str] = ..., backend: _Optional[str] = ..., status: _Optional[str] = ..., total_nodes: _Optional[int] = ..., node_sort_policy: _Optional[str] = ..., total_applications: _Optional[int] = ..., total_tasks: _Optional[int] = ..., partition: _Optional[str] = ..., site_id: _Optional[str] = ...) -> None: ...

class StatusSlice(_message.Message):
    __slots__ = ("label", "count")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    label: str
    count: int
    def __init__(self, label: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class HistoryPoint(_message.Message):
    __slots__ = ("timestamp_ns", "value")
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    timestamp_ns: int
    value: int
    def __init__(self, timestamp_ns: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...

class UtilBucket(_message.Message):
    __slots__ = ("bucket_name", "num_nodes")
    BUCKET_NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_NODES_FIELD_NUMBER: _ClassVar[int]
    bucket_name: str
    num_nodes: int
    def __init__(self, bucket_name: _Optional[str] = ..., num_nodes: _Optional[int] = ...) -> None: ...

class ResourceUtilization(_message.Message):
    __slots__ = ("resource_type", "buckets")
    RESOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    resource_type: str
    buckets: _containers.RepeatedCompositeFieldContainer[UtilBucket]
    def __init__(self, resource_type: _Optional[str] = ..., buckets: _Optional[_Iterable[_Union[UtilBucket, _Mapping]]] = ...) -> None: ...

class GetDashboardResponse(_message.Message):
    __slots__ = ("summary", "application_status", "task_status", "application_history", "task_history", "node_utilizations", "healthy", "partition", "partitions")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    APPLICATION_STATUS_FIELD_NUMBER: _ClassVar[int]
    TASK_STATUS_FIELD_NUMBER: _ClassVar[int]
    APPLICATION_HISTORY_FIELD_NUMBER: _ClassVar[int]
    TASK_HISTORY_FIELD_NUMBER: _ClassVar[int]
    NODE_UTILIZATIONS_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    PARTITIONS_FIELD_NUMBER: _ClassVar[int]
    summary: SchedulerSummary
    application_status: _containers.RepeatedCompositeFieldContainer[StatusSlice]
    task_status: _containers.RepeatedCompositeFieldContainer[StatusSlice]
    application_history: _containers.RepeatedCompositeFieldContainer[HistoryPoint]
    task_history: _containers.RepeatedCompositeFieldContainer[HistoryPoint]
    node_utilizations: _containers.RepeatedCompositeFieldContainer[ResourceUtilization]
    healthy: bool
    partition: str
    partitions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, summary: _Optional[_Union[SchedulerSummary, _Mapping]] = ..., application_status: _Optional[_Iterable[_Union[StatusSlice, _Mapping]]] = ..., task_status: _Optional[_Iterable[_Union[StatusSlice, _Mapping]]] = ..., application_history: _Optional[_Iterable[_Union[HistoryPoint, _Mapping]]] = ..., task_history: _Optional[_Iterable[_Union[HistoryPoint, _Mapping]]] = ..., node_utilizations: _Optional[_Iterable[_Union[ResourceUtilization, _Mapping]]] = ..., healthy: bool = ..., partition: _Optional[str] = ..., partitions: _Optional[_Iterable[str]] = ...) -> None: ...

class SyncProjectionRequest(_message.Message):
    __slots__ = ("partition",)
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    partition: str
    def __init__(self, partition: _Optional[str] = ...) -> None: ...

class SyncProjectionResponse(_message.Message):
    __slots__ = ("notes_written", "config_path")
    NOTES_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    CONFIG_PATH_FIELD_NUMBER: _ClassVar[int]
    notes_written: int
    config_path: str
    def __init__(self, notes_written: _Optional[int] = ..., config_path: _Optional[str] = ...) -> None: ...

class GetProjectionIndexRequest(_message.Message):
    __slots__ = ("root", "archive_id")
    ROOT_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_ID_FIELD_NUMBER: _ClassVar[int]
    root: ProjectionRoot
    archive_id: str
    def __init__(self, root: _Optional[_Union[ProjectionRoot, str]] = ..., archive_id: _Optional[str] = ...) -> None: ...

class ProjectionNoteMeta(_message.Message):
    __slots__ = ("id", "title", "kind", "root", "relpath", "links")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    RELPATH_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    id: str
    title: str
    kind: str
    root: ProjectionRoot
    relpath: str
    links: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., title: _Optional[str] = ..., kind: _Optional[str] = ..., root: _Optional[_Union[ProjectionRoot, str]] = ..., relpath: _Optional[str] = ..., links: _Optional[_Iterable[str]] = ...) -> None: ...

class GetProjectionIndexResponse(_message.Message):
    __slots__ = ("notes", "total")
    NOTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    notes: _containers.RepeatedCompositeFieldContainer[ProjectionNoteMeta]
    total: int
    def __init__(self, notes: _Optional[_Iterable[_Union[ProjectionNoteMeta, _Mapping]]] = ..., total: _Optional[int] = ...) -> None: ...

class GetProjectionNoteRequest(_message.Message):
    __slots__ = ("root", "note_id", "archive_id")
    ROOT_FIELD_NUMBER: _ClassVar[int]
    NOTE_ID_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_ID_FIELD_NUMBER: _ClassVar[int]
    root: ProjectionRoot
    note_id: str
    archive_id: str
    def __init__(self, root: _Optional[_Union[ProjectionRoot, str]] = ..., note_id: _Optional[str] = ..., archive_id: _Optional[str] = ...) -> None: ...

class GetProjectionNoteResponse(_message.Message):
    __slots__ = ("id", "title", "kind", "body", "frontmatter_json", "links")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    FRONTMATTER_JSON_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    id: str
    title: str
    kind: str
    body: str
    frontmatter_json: str
    links: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., title: _Optional[str] = ..., kind: _Optional[str] = ..., body: _Optional[str] = ..., frontmatter_json: _Optional[str] = ..., links: _Optional[_Iterable[str]] = ...) -> None: ...

class WriteScratchConfigRequest(_message.Message):
    __slots__ = ("document", "rebuild_notes")
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    REBUILD_NOTES_FIELD_NUMBER: _ClassVar[int]
    document: PolicyDocument
    rebuild_notes: bool
    def __init__(self, document: _Optional[_Union[PolicyDocument, _Mapping]] = ..., rebuild_notes: bool = ...) -> None: ...

class WriteScratchConfigResponse(_message.Message):
    __slots__ = ("ok", "message")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    def __init__(self, ok: bool = ..., message: _Optional[str] = ...) -> None: ...

class DiffConfigRequest(_message.Message):
    __slots__ = ("include_live",)
    INCLUDE_LIVE_FIELD_NUMBER: _ClassVar[int]
    include_live: bool
    def __init__(self, include_live: bool = ...) -> None: ...

class DiffConfigResponse(_message.Message):
    __slots__ = ("unified_diff", "live_diff")
    UNIFIED_DIFF_FIELD_NUMBER: _ClassVar[int]
    LIVE_DIFF_FIELD_NUMBER: _ClassVar[int]
    unified_diff: str
    live_diff: str
    def __init__(self, unified_diff: _Optional[str] = ..., live_diff: _Optional[str] = ...) -> None: ...

class PromoteScratchRequest(_message.Message):
    __slots__ = ("dry_run", "archive_stamp")
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_STAMP_FIELD_NUMBER: _ClassVar[int]
    dry_run: bool
    archive_stamp: str
    def __init__(self, dry_run: bool = ..., archive_stamp: _Optional[str] = ...) -> None: ...

class PromoteScratchResponse(_message.Message):
    __slots__ = ("ok", "message", "archive_id", "applied", "validation")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_ID_FIELD_NUMBER: _ClassVar[int]
    APPLIED_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    archive_id: str
    applied: bool
    validation: ValidateConfigResponse
    def __init__(self, ok: bool = ..., message: _Optional[str] = ..., archive_id: _Optional[str] = ..., applied: bool = ..., validation: _Optional[_Union[ValidateConfigResponse, _Mapping]] = ...) -> None: ...

class ListArchivesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ArchiveInfo(_message.Message):
    __slots__ = ("id", "created_at", "note")
    ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    id: str
    created_at: str
    note: str
    def __init__(self, id: _Optional[str] = ..., created_at: _Optional[str] = ..., note: _Optional[str] = ...) -> None: ...

class ListArchivesResponse(_message.Message):
    __slots__ = ("archives",)
    ARCHIVES_FIELD_NUMBER: _ClassVar[int]
    archives: _containers.RepeatedCompositeFieldContainer[ArchiveInfo]
    def __init__(self, archives: _Optional[_Iterable[_Union[ArchiveInfo, _Mapping]]] = ...) -> None: ...

class RestoreArchiveToScratchRequest(_message.Message):
    __slots__ = ("archive_id",)
    ARCHIVE_ID_FIELD_NUMBER: _ClassVar[int]
    archive_id: str
    def __init__(self, archive_id: _Optional[str] = ...) -> None: ...

class RestoreArchiveToScratchResponse(_message.Message):
    __slots__ = ("ok", "message")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    def __init__(self, ok: bool = ..., message: _Optional[str] = ...) -> None: ...

class WorkloadIntent(_message.Message):
    __slots__ = ("wrk", "queue", "resource_class", "applications")
    WRK_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_CLASS_FIELD_NUMBER: _ClassVar[int]
    APPLICATIONS_FIELD_NUMBER: _ClassVar[int]
    wrk: str
    queue: str
    resource_class: str
    applications: int
    def __init__(self, wrk: _Optional[str] = ..., queue: _Optional[str] = ..., resource_class: _Optional[str] = ..., applications: _Optional[int] = ...) -> None: ...

class QueueShare(_message.Message):
    __slots__ = ("queue", "guaranteed", "max", "max_applications")
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    GUARANTEED_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MAX_APPLICATIONS_FIELD_NUMBER: _ClassVar[int]
    queue: str
    guaranteed: ResourceMap
    max: ResourceMap
    max_applications: int
    def __init__(self, queue: _Optional[str] = ..., guaranteed: _Optional[_Union[ResourceMap, _Mapping]] = ..., max: _Optional[_Union[ResourceMap, _Mapping]] = ..., max_applications: _Optional[int] = ...) -> None: ...

class QueueShareRequest(_message.Message):
    __slots__ = ("peer", "request_id", "valid_from_ns", "valid_until_ns", "reason", "supersedes_request_id", "workloads", "shares")
    PEER_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VALID_FROM_NS_FIELD_NUMBER: _ClassVar[int]
    VALID_UNTIL_NS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    SUPERSEDES_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    WORKLOADS_FIELD_NUMBER: _ClassVar[int]
    SHARES_FIELD_NUMBER: _ClassVar[int]
    peer: str
    request_id: str
    valid_from_ns: int
    valid_until_ns: int
    reason: str
    supersedes_request_id: str
    workloads: _containers.RepeatedCompositeFieldContainer[WorkloadIntent]
    shares: _containers.RepeatedCompositeFieldContainer[QueueShare]
    def __init__(self, peer: _Optional[str] = ..., request_id: _Optional[str] = ..., valid_from_ns: _Optional[int] = ..., valid_until_ns: _Optional[int] = ..., reason: _Optional[str] = ..., supersedes_request_id: _Optional[str] = ..., workloads: _Optional[_Iterable[_Union[WorkloadIntent, _Mapping]]] = ..., shares: _Optional[_Iterable[_Union[QueueShare, _Mapping]]] = ...) -> None: ...

class QueueShareResponse(_message.Message):
    __slots__ = ("accepted", "request_id", "state", "error")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    request_id: str
    state: QueueShareState
    error: str
    def __init__(self, accepted: bool = ..., request_id: _Optional[str] = ..., state: _Optional[_Union[QueueShareState, str]] = ..., error: _Optional[str] = ...) -> None: ...

class ListQueueShareRequestsRequest(_message.Message):
    __slots__ = ("peer", "queue", "since_ns", "limit")
    PEER_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    SINCE_NS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    peer: str
    queue: str
    since_ns: int
    limit: int
    def __init__(self, peer: _Optional[str] = ..., queue: _Optional[str] = ..., since_ns: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...

class QueueShareRecord(_message.Message):
    __slots__ = ("request", "recorded_at_ns", "state")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RECORDED_AT_NS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    request: QueueShareRequest
    recorded_at_ns: int
    state: QueueShareState
    def __init__(self, request: _Optional[_Union[QueueShareRequest, _Mapping]] = ..., recorded_at_ns: _Optional[int] = ..., state: _Optional[_Union[QueueShareState, str]] = ...) -> None: ...

class ListQueueShareRequestsResponse(_message.Message):
    __slots__ = ("records",)
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    records: _containers.RepeatedCompositeFieldContainer[QueueShareRecord]
    def __init__(self, records: _Optional[_Iterable[_Union[QueueShareRecord, _Mapping]]] = ...) -> None: ...
