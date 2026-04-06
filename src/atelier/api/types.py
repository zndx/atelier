"""Application-level enums and types."""

from enum import Enum


class AgentType(str, Enum):
    """Types of keystone agents."""

    CLASSIFIER = "classifier"
    EVIDENCE_FUSER = "evidence_fuser"
    VISUALIZATION_DIRECTOR = "visualization_director"


class DatasetType(str, Enum):
    """Classification dataset source types."""

    PARQUET = "parquet"
    SIGNALS = "signals"
