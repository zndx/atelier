# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""CDP Governance SDK — Atlas and Ranger REST clients for CDP environments.

Provides AtlasClient and RangerClient for direct REST access to Apache
Atlas v2 and Apache Ranger v2 APIs.  GovernanceClient is a convenience
facade exposing both through a single configuration.

Designed as a standalone SDK with only ``requests`` as a dependency,
integrated into Atelier via HOCON config.
"""

from atelier.governance.client import GovernanceClient, ClientConfig
from atelier.governance.atlas import AtlasClient, ClassificationTag, QualifiedName
from atelier.governance.ranger import RangerClient, RangerRole

__all__ = [
    "GovernanceClient", "ClientConfig",
    "AtlasClient", "ClassificationTag", "QualifiedName",
    "RangerClient", "RangerRole",
]
