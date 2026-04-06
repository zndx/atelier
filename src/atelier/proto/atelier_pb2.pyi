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

class ClassificationDataset(_message.Message):
    __slots__ = ("id", "name", "parquet_path", "description", "row_count")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARQUET_PATH_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    parquet_path: str
    description: str
    row_count: int
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., parquet_path: _Optional[str] = ..., description: _Optional[str] = ..., row_count: _Optional[int] = ...) -> None: ...

class ListDatasetsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListDatasetsResponse(_message.Message):
    __slots__ = ("datasets",)
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    datasets: _containers.RepeatedCompositeFieldContainer[ClassificationDataset]
    def __init__(self, datasets: _Optional[_Iterable[_Union[ClassificationDataset, _Mapping]]] = ...) -> None: ...
