# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
