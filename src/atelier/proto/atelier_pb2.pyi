from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "version")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    status: str
    version: str
    def __init__(self, status: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class AgentMetadata(_message.Message):
    __slots__ = ("id", "name", "description", "role", "tool_ids")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    TOOL_IDS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    role: str
    tool_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., role: _Optional[str] = ..., tool_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ListAgentsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListAgentsResponse(_message.Message):
    __slots__ = ("agents",)
    AGENTS_FIELD_NUMBER: _ClassVar[int]
    agents: _containers.RepeatedCompositeFieldContainer[AgentMetadata]
    def __init__(self, agents: _Optional[_Iterable[_Union[AgentMetadata, _Mapping]]] = ...) -> None: ...

class GetAgentRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class GetAgentResponse(_message.Message):
    __slots__ = ("agent",)
    AGENT_FIELD_NUMBER: _ClassVar[int]
    agent: AgentMetadata
    def __init__(self, agent: _Optional[_Union[AgentMetadata, _Mapping]] = ...) -> None: ...

class DataSource(_message.Message):
    __slots__ = ("id", "source_type", "source_uri", "display_name", "vocabulary_mode", "created_at", "metadata_json")
    ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_URI_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    VOCABULARY_MODE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    METADATA_JSON_FIELD_NUMBER: _ClassVar[int]
    id: str
    source_type: str
    source_uri: str
    display_name: str
    vocabulary_mode: str
    created_at: str
    metadata_json: str
    def __init__(self, id: _Optional[str] = ..., source_type: _Optional[str] = ..., source_uri: _Optional[str] = ..., display_name: _Optional[str] = ..., vocabulary_mode: _Optional[str] = ..., created_at: _Optional[str] = ..., metadata_json: _Optional[str] = ...) -> None: ...

class ListDataSourcesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListDataSourcesResponse(_message.Message):
    __slots__ = ("sources",)
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    sources: _containers.RepeatedCompositeFieldContainer[DataSource]
    def __init__(self, sources: _Optional[_Iterable[_Union[DataSource, _Mapping]]] = ...) -> None: ...

class ClassificationDataset(_message.Message):
    __slots__ = ("id", "name", "parquet_path", "description", "row_count", "source_id", "version_number", "is_active", "summary", "fsm_run_id", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARQUET_PATH_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_NUMBER_FIELD_NUMBER: _ClassVar[int]
    IS_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    FSM_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    parquet_path: str
    description: str
    row_count: int
    source_id: str
    version_number: int
    is_active: bool
    summary: str
    fsm_run_id: str
    created_at: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., parquet_path: _Optional[str] = ..., description: _Optional[str] = ..., row_count: _Optional[int] = ..., source_id: _Optional[str] = ..., version_number: _Optional[int] = ..., is_active: bool = ..., summary: _Optional[str] = ..., fsm_run_id: _Optional[str] = ..., created_at: _Optional[str] = ...) -> None: ...

class ListDatasetsRequest(_message.Message):
    __slots__ = ("source_id",)
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    def __init__(self, source_id: _Optional[str] = ...) -> None: ...

class ListDatasetsResponse(_message.Message):
    __slots__ = ("datasets",)
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    datasets: _containers.RepeatedCompositeFieldContainer[ClassificationDataset]
    def __init__(self, datasets: _Optional[_Iterable[_Union[ClassificationDataset, _Mapping]]] = ...) -> None: ...

class FSMStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FSMStatusResponse(_message.Message):
    __slots__ = ("run_id", "state", "started_at", "updated_at", "progress_json", "error")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    state: str
    started_at: str
    updated_at: str
    progress_json: str
    error: str
    def __init__(self, run_id: _Optional[str] = ..., state: _Optional[str] = ..., started_at: _Optional[str] = ..., updated_at: _Optional[str] = ..., progress_json: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class StartClassificationRequest(_message.Message):
    __slots__ = ("connection_name", "database", "sample_size", "source_id")
    CONNECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DATABASE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_SIZE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    connection_name: str
    database: str
    sample_size: int
    source_id: str
    def __init__(self, connection_name: _Optional[str] = ..., database: _Optional[str] = ..., sample_size: _Optional[int] = ..., source_id: _Optional[str] = ...) -> None: ...

class StartClassificationResponse(_message.Message):
    __slots__ = ("run_id", "started", "error")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    started: bool
    error: str
    def __init__(self, run_id: _Optional[str] = ..., started: bool = ..., error: _Optional[str] = ...) -> None: ...
